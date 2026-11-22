#!/usr/bin/env python3
"""stock_decision_v1 teacher BACKENDS: each teacher through its own released inference path (see teacher_pins.json).

  stock   google/gemma-4-31B-it @842da379 through the published Garas engine code (feea8bb) with NO adapter: the R2 prompt
          (garas_core.build_prompt, noul_defs=True), the engine's hash-derived non-identity choice order, the tied letter
          readout (exact_letter_logits, bf16, Gemma softcap), K > 52 by the engine's two rounds. Score questions (the engine
          refuses them) are rendered by the same build_prompt as ordered levels in level order (as R2 renders score).
  garas   the same engine with the published adapter (plantwaterer/garas @feea8bb, md5 56a0b787, autocast_adapter_dtype=False):
          GarasEngine(base, adapter).__call__ unchanged. Score: unsupported (its engine raises Unsupported).
  d3      vllm-sr/d3 @fc2d74bd: d3_runtime.D3.from_pretrained(snapshot, device, batch_size=8, verify="fast").system_one(...):
          noncausal full attention, last-token 255-code readout (fp32), T = 1, one question per request.
  pplx    perplexity-ai/pplx-decider-v1.1-27b @14c9d901: source/src/autojev DecisionModel(snapshot).predict(rows, batch_size=8):
          noncausal full attention, last-token 255-code readout, T = decision_config.json temperature (applied once).
          Its own 8,192-token limit: a longer prompt is refused (checked one row at a time before batching; never truncated).
  olmo    allenai/Olmo-3.1-32B-Think @97ade57a (a different lineage: Ai2, pretrained from scratch on Dolma 3): reason with the
          model's chat template (sampling = its generation_config: T 0.6, top-p 0.95, a per-decision seed, at most
          max_think tokens, stopped at "</think>"), then READ the answer distribution: the next-token probabilities of the
          option letters " A".." z" after reasoning + "</think>\\n\\nAnswer:", renormalised over the K valid letters. Fleet
          backend: SGLang only (python3 -m sglang.launch_server with a named attention backend, input-token logprobs, as
          trainers/claude_think.py); transformers only in laptop tests and the dry's read-out parity. K <= 52; prompts <= max_prompt_tokens.
Every backend: predict_batch(decisions) -> [sdv1_teacher.Result]; provenance() -> dict. Heavy imports happen in __init__.
"""
from __future__ import annotations

import hashlib
import os
import string
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sdv1_teacher as st  # noqa: E402

LETTERS = string.ascii_uppercase + string.ascii_lowercase


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _check_code(pin, roots):
    """the teacher's own code files must be the pinned ones (md5), wherever the job put them."""
    got = {}
    for rel, want in (pin.get("code_md5") or {}).items():
        p = next((os.path.join(r, rel) for r in roots if r and os.path.exists(os.path.join(r, rel))), None)
        if p is None:
            raise SystemExit(f"pinned code file {rel} not found under {roots}")
        m = _md5(p)
        if m != want:
            raise SystemExit(f"{rel}: md5 {m} != pinned {want}")
        got[rel] = m
    return got


def _oom(e):
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _free_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class Backend:
    batch_size = 1

    def predict(self, d):
        raise NotImplementedError

    def predict_batch(self, ds):
        out = []
        for d in ds:
            try:
                out.append(self.predict(d))
            except Exception as e:  # noqa: BLE001 - every failure becomes an explicit status, never a distribution
                if _oom(e):
                    _free_cuda()
                    out.append(st.Result("failure_oom", detail=f"{type(e).__name__}: {e}"))
                else:
                    out.append(st.Result("failure", detail=f"{type(e).__name__}: {e}"))
        return out

    def provenance(self):
        return dict(self.prov)


# ----------------------------------------------------------------------------------------------------- Garas engine family
def _stock_engine_class(ge, core):
    class StockGemmaEngine(ge.GarasEngine):
        """GarasEngine with the adapter step removed (everything else is the published engine's own code)."""
        name = "stock-gemma-4-31b-it"

        def _load(self, base, adapter, device):
            if adapter not in (None, "", "none"):
                raise SystemExit("the stock teacher takes no adapter")
            import torch
            from transformers import AutoConfig, AutoTokenizer
            mcfg = AutoConfig.from_pretrained(base)
            _, model_cls = core.model_loader(mcfg, "auto")
            self.chat = core.chat_preset(mcfg.model_type, False)
            tok = AutoTokenizer.from_pretrained(base)
            tok.padding_side = "right"
            ids = [tok(c, add_special_tokens=False)["input_ids"] for c in core.LABELS]
            if any(len(x) != 1 for x in ids):
                raise SystemExit("a letter is not a single token")
            label_ids = torch.tensor([x[0] for x in ids])
            dt = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
            m = model_cls.from_pretrained(base, dtype=dt, device_map=device).eval()
            self.tok, self.model, self.device = tok, m, device
            lab = label_ids.to(device)

            def letter_logits(texts):
                with torch.no_grad():
                    return core.exact_letter_logits(m, tok, texts, lab, device, "pad").float().cpu().numpy()

            self._logits = letter_logits
            self._sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
            import transformers
            self.provenance = {"kind": "in-process", "base": base, "adapter": None, "dtype": str(dt).replace("torch.", ""),
                               "transformers": transformers.__version__, "torch": torch.__version__,
                               "core_md5": _md5(core.__file__), "engine_md5": _md5(ge.__file__)}

        def __call__(self, state, questions):
            if all(q.get("type") != "score" for q in questions.values()):
                return super().__call__(state, questions)
            if len(questions) != 1:
                raise ge.Unsupported("score questions are scored one per request")
            (qkey, q), = questions.items()
            levels = [core.as_text(x) for x in (q.get("criteria") or [])]
            if not 2 <= len(levels) <= core.MAX_K:
                raise ge.Unsupported(f"score with {len(levels)} levels (2..{core.MAX_K})")
            u = {"state": core.render_state(state, "text"), "instructions": core.as_text(q.get("instructions")),
                 "qtype": "score", "option_keys": [str(i) for i in range(len(levels))], "option_texts": levels,
                 "noul_criteria": None}
            order = list(range(len(levels)))           # ordered levels keep their order (as the R2 renderer)
            text = core.build_prompt(self.tok, u, order, self.max_len, chat=self.chat, noul_defs=True)
            if text is None or u["state"] not in text:
                raise ge.Unsupported(f"maximum context length: question {qkey!r} does not fit {self.max_len} tokens "
                                     f"without cutting the state (no truncation)")
            qid = "score|" + qkey
            p = self.run_prompts([(qid, order, text)])[qid]
            return {"model": self.name, "answers": {qkey: {"type": "score", "probabilities": core.norm(dict(zip(u["option_keys"], p)))}}}, None

    return StockGemmaEngine


class GarasEngineBackend(Backend):
    QKEY = "q"

    def __init__(self, teacher, pin, snapshot, adapter, device, options):
        code_dir, kit_dir = options["code_dir"], options["kit_dir"]
        for p in (kit_dir, code_dir):
            if p not in sys.path:
                sys.path.insert(0, p)
        self.code_md5 = _check_code(pin, [code_dir])
        import garas_core as core
        import garas_engine as ge
        self.teacher, self.ge, self.core = teacher, ge, core
        kw = dict(device=device, max_len=int(pin["inference"]["max_len"]), tokens_per_batch=int(pin["inference"]["tokens_per_batch"]))
        if teacher == "garas":
            if not adapter:
                raise SystemExit("garas needs --adapter (the pinned adapter dir)")
            self.engine = ge.GarasEngine(base=snapshot, adapter=adapter, **kw)
        else:
            self.engine = _stock_engine_class(ge, core)(base=snapshot, adapter=None, **kw)
        self._prompts = []
        orig = self.engine.run_prompts

        def recorded(items):
            for _, _, t in items:
                self._prompts.append((st.sha256_text(t), len(self.engine.tok(t, add_special_tokens=False)["input_ids"])))
            return orig(items)

        self.engine.run_prompts = recorded
        self.prov = {"teacher": teacher, "engine": dict(self.engine.provenance), "code_md5": self.code_md5, "kw": kw,
                     "question_key": self.QKEY}

    def predict(self, d):
        q = st.systemone_question(d)
        req = {"state": d["state"], "questions": {self.QKEY: q}}
        if d["kind"] == "score" and self.teacher == "garas":
            return st.Result("unsupported_kind", detail="the Garas engine refuses score questions (its released behaviour)",
                             request=req)
        self._prompts = []
        try:
            resp, _ = self.engine(d["state"], {self.QKEY: q})
        except self.ge.Unsupported as e:
            msg = str(e)
            status = ("unsupported_too_long" if "context length" in msg else "unsupported_kind" if "type" in msg
                      else "unsupported_menu")
            return st.Result(status, detail=msg, request=req)
        ans = resp["answers"][self.QKEY]
        extra = {"prompt_sha256": [h for h, _ in self._prompts], "n_prompts": len(self._prompts)}
        try:
            probs = st.map_systemone_answer(d, ans)
        except st.InvalidOutput as e:
            return st.Result("invalid_output", detail=str(e), native_answer=ans, request=req, extra=extra)
        return st.Result("ok", probabilities=probs, native_answer=ans, request=req,
                         n_input_tokens=sum(n for _, n in self._prompts), extra=extra)


# ---------------------------------------------------------------------------------------------------------------------- d3
class D3Backend(Backend):
    QKEY = "q"
    ERR = {"max_length_exceeded": "unsupported_too_long", "invalid_question": "unsupported_menu",
           "invalid_model_output": "invalid_output"}

    def __init__(self, teacher, pin, snapshot, adapter, device, options):
        if snapshot not in sys.path:
            sys.path.insert(0, snapshot)
        self.code_md5 = _check_code(pin, [snapshot])
        import d3_runtime
        dev = "cuda:0" if device == "cuda" else device
        inf = pin["inference"]
        self.model = d3_runtime.D3.from_pretrained(snapshot, device=dev, batch_size=int(inf["batch_size"]),
                                                   verify=inf.get("verify", "fast"),
                                                   permutation_average=bool(inf.get("permutation_average", False)))
        self.rt = d3_runtime
        self.prov = {"teacher": teacher, "code_md5": self.code_md5, "runtime": self.model.runtime_info(),
                     "loaded_seconds": round(self.model.loaded_seconds, 1), "fast_skipped": self.model.fast_skipped,
                     "kernels": self.model.kernels, "question_key": self.QKEY}

    def predict(self, d):
        q = st.systemone_question(d)
        req = {"state": d["state"], "questions": {self.QKEY: q}}
        if d["kind"] == "score" and not 2 <= len(d["options"]) <= self.rt.SCORE_LEVELS[1]:
            return st.Result("unsupported_menu", detail=f"d3 score takes {self.rt.SCORE_LEVELS} levels", request=req)
        if d["kind"] == "choice" and len(d["options"]) > self.rt.MAX_OPTIONS:
            return st.Result("unsupported_menu", detail=f"d3 takes at most {self.rt.MAX_OPTIONS} options", request=req)
        resp = self.model.system_one(state=d["state"], questions={self.QKEY: q})
        ans = resp["answers"][self.QKEY]
        if "error" in ans:
            return st.Result(self.ERR.get(ans["error"], "failure"), detail=f"{ans['error']}: {ans.get('message', '')}",
                             native_answer=ans, request=req)
        try:
            probs = st.map_systemone_answer(d, ans)
        except st.InvalidOutput as e:
            return st.Result("invalid_output", detail=str(e), native_answer=ans, request=req)
        return st.Result("ok", probabilities=probs, native_answer=ans, request=req,
                         n_input_tokens=int(resp.get("usage", {}).get("input_tokens") or 0))


# -------------------------------------------------------------------------------------------------------------------- pplx
class PplxBackend(Backend):
    batch_size = 64            # rows handed over per call; predict() itself runs them 8 at a time, sorted by length

    def __init__(self, teacher, pin, snapshot, adapter, device, options):
        src = os.path.join(snapshot, "source", "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        self.code_md5 = _check_code(pin, [snapshot])
        from autojev import model as am
        self.am = am
        self.inner = int(pin["inference"]["batch_size"])
        self.max_tokens = int(pin["inference"]["max_tokens"])
        self.m = am.DecisionModel(snapshot, device=device)
        self.prov = {"teacher": teacher, "code_md5": self.code_md5, "temperature": self.m.temperature,
                     "attention_mode": self.m.attention_mode, "pooling": self.m.pooling, "device": self.m.device_name,
                     "inner_batch_size": self.inner, "sorted_by_length": True}

    def _row(self, d):
        return {"state": d["state"], "question": st.systemone_question(d)}

    def predict_batch(self, ds):
        out = [None] * len(ds)
        ok = []
        for i, d in enumerate(ds):
            row = self._row(d)
            try:
                n = self.m.prepare([row], max_length=self.max_tokens).input_tokens
            except ValueError as e:
                msg = str(e)
                status = "unsupported_too_long" if "token limit" in msg else "unsupported_menu" if "option" in msg else "unsupported_input"
                out[i] = st.Result(status, detail=msg, request=row)
                continue
            except Exception as e:  # noqa: BLE001
                out[i] = st.Result("failure", detail=f"{type(e).__name__}: {e}", request=row)
                continue
            ok.append((n, i, row))
        ok.sort(key=lambda x: (x[0], x[1]))
        for j in range(0, len(ok), self.inner):
            grp = ok[j:j + self.inner]
            try:
                probs = self.m.predict([r for _, _, r in grp], batch_size=self.inner)
            except Exception as e:  # noqa: BLE001
                if _oom(e):
                    _free_cuda()
                status = "failure_oom" if _oom(e) else "failure"
                for n, i, r in grp:
                    out[i] = st.Result(status, detail=f"{type(e).__name__}: {e}", request=r, n_input_tokens=n)
                continue
            for (n, i, r), p in zip(grp, probs):
                d = ds[i]
                # pplx's own option order: choice = the criteria keys (= semantic keys, presented order); noul = ["false",
                # "true"] (= the semantic keys); score = level indices "0".."K-1" -> the semantic level keys by index
                keys = self.am.options(r["question"])[0]
                native = st.keys_of(d) if d["kind"] == "score" else keys
                try:
                    pr = st.map_index_probs(d, p, native)
                except st.InvalidOutput as e:
                    out[i] = st.Result("invalid_output", detail=str(e), native_answer=p, request=r, n_input_tokens=n)
                    continue
                out[i] = st.Result("ok", probabilities=pr, native_answer=[float(x) for x in p], request=r, n_input_tokens=n)
        return out


# ------------------------------------------------------------------------------------------------- reasoning teacher (olmo)
class ReasoningBackend(Backend):
    """v2 (2026-10-14, after the od1 dry): SGLang ONLY on the fleet (backend 'sglang'; the transformers path is for laptop tests).
    The v1 dry fell back to transformers because SGLang auto-picked the fa3 attention backend, which asserts SM 80-90 (the RTX PRO
    6000 is SM 12.0); v2 names the attention backend and walks a start-up chain (pin sglang_attempts: flashinfer, flashinfer
    without CUDA graphs, triton without CUDA graphs). KV is SGLang's paged pool (bf16) with chunked prefill and a running-request cap;
    reasoning is capped at max_think tokens. The server runs with the IMAGE's packages (PYTHONPATH from server_pythonpath, default
    unset), so the job-local transformers env never reaches it. readout_parity() (dry only) re-reads the letter distribution of a
    few SGLang continuations with transformers (the >= 5.17 per-layer-type RoPE: YaRN on full-attention layers only, as SGLang)."""
    batch_size = 64

    @property
    def streaming(self):
        """(v5) SGLang: the harness keeps `concurrency` requests in flight across the whole shard (predict_one); v4 handed over
        64 rows at a time and waited for the slowest (the od1 log showed 5 running requests at a chunk tail)."""
        return self.backend == "sglang"

    def predict_one(self, d):
        try:
            return self._sglang_one(d)
        except Exception as e:  # noqa: BLE001
            return st.Result("failure", detail=f"{type(e).__name__}: {e}")

    def __init__(self, teacher, pin, snapshot, adapter, device, options):
        import threading
        from transformers import AutoTokenizer
        inf = pin["inference"]
        self.inf, self.options = inf, options
        self.snapshot, self.device, self.teacher = snapshot, device, teacher
        self.max_think = int(options.get("max_think", inf["max_think_tokens"]))
        self.max_prompt = int(inf["max_prompt_tokens"])
        self.temperature, self.top_p = float(inf["temperature"]), float(inf["top_p"])
        self.answer_suffix = inf["answer_suffix"]
        self.tok = AutoTokenizer.from_pretrained(snapshot)
        self.letter_ids = []
        for c in LETTERS:
            ids = self.tok.encode(" " + c, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"' {c}' is not one token in this tokenizer")
            self.letter_ids.append(ids[0])
        self.backend = options.get("backend", inf.get("backend", "sglang"))
        self.concurrency = int(options.get("concurrency", inf.get("concurrency", 48)))
        self.timeout = float(options.get("timeout_s", inf.get("timeout_s", 1800)))
        self.server, self.model = None, None
        self._local = threading.local()
        self.kept = []                       # (decision_uid, cont, n_cont_tokens, keys, z) of recent ok rows (readout parity)
        self.prov = {"teacher": teacher, "max_think": self.max_think, "max_prompt": self.max_prompt, "temperature": self.temperature,
                     "top_p": self.top_p, "answer_suffix": self.answer_suffix, "prompt_template": inf["prompt_template_id"],
                     "requested_backend": self.backend}
        if self.backend == "sglang":
            self._start_sglang()
        elif self.backend == "hf":
            self._load_hf()
        else:
            raise SystemExit(f"backend {self.backend!r}: sglang | hf")
        self.prov["backend"] = self.backend

    # ---- prompt
    def prompt(self, d):
        lines = st.option_lines_texts(d)
        body = (st.state_text(d["state"]) + "\n\nQuestion: " + (d["instructions"] or "Choose the best option.") + "\n\nOptions:\n"
                + "\n".join(f"{LETTERS[j]}) {t}" for j, (_, t) in enumerate(lines))
                + "\n\nThink it through, then give the letter of the single best option.")
        text = self.tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
        return text, [k for k, _ in lines]

    def seed(self, d):
        return int(st.sha256_text("sdv1-reason-seed|" + d["decision_uid"] + "|" + st.canon(st.keys_of(d)) + "|"
                                  + st.sha256_text(st.state_text(d["state"])))[:8], 16)

    def _pre(self, d):
        if len(d["options"]) > len(LETTERS):
            return None, st.Result("unsupported_menu", detail=f"K={len(d['options'])} > {len(LETTERS)} letters",
                                   request={"state": d["state"], "question": st.systemone_question(d)})
        text, keys = self.prompt(d)
        req = {"prompt": text}
        n = len(self.tok(text, add_special_tokens=False)["input_ids"])
        if n > self.max_prompt:
            return None, st.Result("unsupported_too_long", detail=f"{n} prompt tokens > {self.max_prompt}", request=req,
                                   n_input_tokens=n)
        return (text, keys, n, req), None

    def _cont(self, text, reasoning, truncated):
        return text + reasoning + ("\n" if truncated else "") + "</think>" + self.answer_suffix

    def _finish(self, d, text, keys, n, req, reasoning, truncated, z, n_think, finish, cont=None, n_cont=None):
        import math
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        probs = {k: v / s for k, v in zip(keys, e)}
        extra = {"reasoning": reasoning, "think_tokens": n_think, "truncated": bool(truncated), "finish": finish,
                 "seed": self.seed(d), "backend": self.backend}
        if cont is not None:
            self.kept.append((d["decision_uid"], cont, n_cont, list(keys), list(z)))
            del self.kept[:-400]
        return st.Result("ok", probabilities=probs, native_answer={"letter_logprobs": z}, request=req, n_input_tokens=n,
                         extra=extra)

    # ---- SGLang
    def sglang_args(self, attempt):
        inf = self.inf
        port = str(int(self.options.get("port", inf.get("sglang_port", 30071))))
        args = [sys.executable, "-m", "sglang.launch_server", "--model-path", self.snapshot, "--tp-size", "1",
                "--dtype", "bfloat16", "--mem-fraction-static", str(inf.get("sglang_mem_fraction", 0.88)),
                "--context-length", str(int(inf["context_length"])), "--chunked-prefill-size", str(int(inf["chunked_prefill_size"])),
                "--max-running-requests", str(int(inf["max_running_requests"])), "--random-seed", "0",
                "--host", "127.0.0.1", "--port", port, "--attention-backend", attempt["attention_backend"]]
        if attempt.get("disable_cuda_graph"):
            args.append("--disable-cuda-graph")
        return args + list(inf.get("sglang_extra_args", []))

    def server_env(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if self.options.get("server_pythonpath"):
            env["PYTHONPATH"] = self.options["server_pythonpath"]
        return env

    def _start_sglang(self):
        import atexit
        import requests
        inf = self.inf
        self.url = f"http://127.0.0.1:{int(self.options.get('port', inf.get('sglang_port', 30071)))}"
        try:
            if requests.get(self.url + "/health", timeout=5).ok:
                raise SystemExit(f"a server already answers on {self.url}; refusing to share it")
        except requests.RequestException:
            pass
        atexit.register(self._stop_sglang)
        tried = []
        for attempt in inf["sglang_attempts"]:
            args = self.sglang_args(attempt)
            with open(self.options.get("sglang_log", "sglang_server.log"), "ab") as log:
                log.write(f"\n===== sdv1 start-up attempt {len(tried) + 1}: {attempt}\n".encode())
                log.flush()
                # same process group as the job: a runner that kills the job's group also takes the server down
                self.server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=self.server_env())
            t0, why = time.time(), None
            while time.time() - t0 < float(inf.get("sglang_start_timeout_s", 1500)):
                if self.server.poll() is not None:
                    why = f"exited {self.server.returncode} during start-up"
                    break
                try:
                    if requests.get(self.url + "/health", timeout=5).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(float(self.options.get("health_poll_s", 10)))
            else:
                why = "not healthy in time"
            tried.append({"attempt": attempt, "result": why or "healthy", "seconds": round(time.time() - t0, 1)})
            if why is None:
                self.prov.update({"sglang_args": args[1:], "sglang_start_s": round(time.time() - t0, 1), "sglang_attempts": tried})
                try:
                    out = subprocess.run([sys.executable, "-c", "import sglang, flashinfer; print(sglang.__version__, "
                                          "flashinfer.__version__)"], env=self.server_env(), capture_output=True, text=True,
                                         timeout=120).stdout.split()
                    self.prov["sglang"], self.prov["flashinfer"] = (out + ["?", "?"])[:2]
                except Exception:  # noqa: BLE001
                    pass
                return
            self._stop_sglang()
        self.prov["sglang_attempts"] = tried
        raise RuntimeError(f"SGLang did not start with any of {len(tried)} attempts: {tried}")

    def _stop_sglang(self):
        """SIGTERM (SGLang's launcher then ends its own process tree), SIGKILL after 60 s."""
        if self.server is not None and self.server.poll() is None:
            try:
                self.server.terminate()
                self.server.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    self.server.kill()
                    self.server.wait(timeout=30)
                except Exception:  # noqa: BLE001
                    pass
        self.server = None

    def close(self):
        self._stop_sglang()

    def _session(self):
        import requests
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
        return s

    def _sglang_one(self, d):
        import requests
        pre, bad = self._pre(d)
        if bad:
            return bad
        text, keys, n, req = pre
        s = self._session()
        try:
            r = s.post(self.url + "/generate", json={
                "text": text, "sampling_params": {"max_new_tokens": self.max_think, "temperature": self.temperature,
                                                  "top_p": self.top_p, "stop": ["</think>"], "sampling_seed": self.seed(d)}},
                timeout=self.timeout).json()
            if "text" not in r:
                return st.Result("failure", detail=f"generate: {str(r)[:300]}", request=req, n_input_tokens=n)
            reasoning = r["text"]
            fin = r.get("meta_info", {}).get("finish_reason") or {}
            truncated = isinstance(fin, dict) and fin.get("type") == "length"
            cont = self._cont(text, reasoning, truncated)
            ids = self.tok(cont, add_special_tokens=False)["input_ids"]
            z = []
            for lid in self.letter_ids[:len(keys)]:
                # the letter's log-probability as an INPUT token of ids + [letter] (claude_think.py's measured recipe: the first
                # entry of the requested span comes back null, so the span starts one token earlier)
                r2 = s.post(self.url + "/generate", json={
                    "input_ids": ids + [lid], "sampling_params": {"max_new_tokens": 0}, "return_logprob": True,
                    "logprob_start_len": len(ids) - 1}, timeout=600).json()
                v = r2["meta_info"]["input_token_logprobs"][-1][0]
                if v is None:
                    return st.Result("invalid_output", detail="null input logprob for an answer letter", request=req)
                z.append(float(v))
        except requests.Timeout as e:
            return st.Result("timeout", detail=str(e), request=req, n_input_tokens=n)
        return self._finish(d, text, keys, n, req, reasoning, truncated, z,
                            int(r.get("meta_info", {}).get("completion_tokens") or 0), fin, cont=cont, n_cont=len(ids))

    # ---- transformers (laptop tests; the dry's readout parity)
    def _load_hf(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        dt = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(self.snapshot, dtype=dt, device_map=self.device,
                                                          attn_implementation="sdpa").eval()
        self.hf_budget = int(self.options.get("hf_token_budget", self.inf.get("hf_token_budget", 24576)))
        self.prov.update({"transformers": transformers.__version__, "torch": torch.__version__, "dtype": str(dt),
                          "hf_token_budget": self.hf_budget})

    def _hf_groups(self, pre):
        """length-sorted groups with (rows x (longest prompt + max_think)) <= the token budget (a 10k-token prompt runs alone)."""
        pre = sorted(pre, key=lambda x: (x[0], x[1]))
        out, cur = [], []
        for x in pre:
            if cur and (len(cur) + 1) * (x[0] + self.max_think) > self.hf_budget:
                out.append(cur)
                cur = []
            cur.append(x)
        return out + ([cur] if cur else [])

    def _hf_run(self, ds, grp, out):
        torch = self.torch
        dev = self.model.device
        self.tok.padding_side = "left"
        enc = self.tok([p[0] for _, _, p in grp], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        torch.manual_seed(self.seed(ds[grp[0][1]]))
        with torch.no_grad():
            gen = self.model.generate(**enc, do_sample=True, temperature=self.temperature, top_p=self.top_p,
                                      max_new_tokens=self.max_think, stop_strings=["</think>"], tokenizer=self.tok,
                                      pad_token_id=self.tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        del gen, enc
        for (n, i, (text, keys, _, req)), row in zip(grp, new):
            ids = [t for t in row.tolist() if t != self.tok.pad_token_id]
            s = self.tok.decode(ids, skip_special_tokens=False)
            cut = s.find("</think>")
            truncated = cut < 0
            reasoning = s if truncated else s[:cut]
            cont = self._cont(text, reasoning, truncated)
            z = self.hf_letter_logprobs(cont, len(keys))
            out[i] = self._finish(ds[i], text, keys, n, req, reasoning, truncated, z, len(ids),
                                  {"type": "length" if truncated else "stop"})

    def hf_letter_logprobs(self, cont, k):
        """one row, no padding: the next-token log-probabilities of the first k letters after cont."""
        torch = self.torch
        ids = torch.tensor([self.tok(cont, add_special_tokens=False)["input_ids"]], device=self.model.device)
        with torch.no_grad():
            lg = self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(lg, -1)
        return [float(lp[lid]) for lid in self.letter_ids[:k]]

    def _hf_batch(self, ds):
        out = [None] * len(ds)
        pre = []
        for i, d in enumerate(ds):
            p, bad = self._pre(d)
            if bad:
                out[i] = bad
            else:
                pre.append((p[2], i, p))
        todo = self._hf_groups(pre)
        while todo:
            grp = todo.pop(0)
            try:
                self._hf_run(ds, grp, out)
            except Exception as e:  # noqa: BLE001
                if _oom(e):
                    _free_cuda()
                    if len(grp) > 1:                     # split and retry: an OOM is never a status while a smaller batch fits
                        h = len(grp) // 2
                        todo[:0] = [grp[:h], grp[h:]]
                        continue
                for n, i, p in grp:
                    out[i] = st.Result("failure_oom" if _oom(e) else "failure", detail=f"{type(e).__name__}: {e}", request=p[3])
        return out

    def predict_batch(self, ds):
        if self.backend == "hf":
            return self._hf_batch(ds)
        from concurrent.futures import ThreadPoolExecutor

        def one(d):
            try:
                return self._sglang_one(d)
            except Exception as e:  # noqa: BLE001
                return st.Result("failure", detail=f"{type(e).__name__}: {e}")
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(one, ds))

    def readout_parity(self, n=8, gpu_free_mib=4096):
        """dry only: stop the server, load the model with transformers, re-read the letter distribution of n kept SGLang
        continuations (the longest one included), compare. -> dict (TV per row, argmax agreement)."""
        if self.backend != "sglang" or not self.kept:
            return {"ran": False, "why": f"backend {self.backend}, {len(self.kept)} kept rows"}
        rows = sorted(self.kept, key=lambda x: x[2])
        pick = [rows[-1]] + [rows[int(j * (len(rows) - 1) / max(n - 1, 1))] for j in range(n - 1)]
        seen, chosen = set(), []
        for r in pick:
            if r[0] not in seen:
                seen.add(r[0])
                chosen.append(r)
        self._stop_sglang()
        t0 = time.time()
        while time.time() - t0 < 180:
            try:
                used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                                          capture_output=True, text=True, timeout=10).stdout.split()[0])
            except Exception:  # noqa: BLE001
                break
            if used <= gpu_free_mib:
                break
            time.sleep(5)
        self.backend = "hf-parity"
        self._load_hf()
        import math
        res = []
        for uid, cont, ncont, keys, z in chosen:
            zh = self.hf_letter_logprobs(cont, len(keys))

            def norm(v):
                m = max(v)
                e = [math.exp(x - m) for x in v]
                s = sum(e)
                return [x / s for x in e]
            a, b = norm(z), norm(zh)
            res.append({"decision_uid": uid, "cont_tokens": ncont, "tv": 0.5 * sum(abs(x - y) for x, y in zip(a, b)),
                        "argmax_same": a.index(max(a)) == b.index(max(b)), "p_sglang_max": max(a)})
        tvs = sorted(r["tv"] for r in res)
        return {"ran": True, "rows": res, "n": len(res), "argmax_agreement": sum(r["argmax_same"] for r in res) / len(res),
                "median_tv": tvs[len(tvs) // 2], "max_tv": tvs[-1], "transformers": self.prov.get("transformers")}


BACKENDS = {"stock": GarasEngineBackend, "garas": GarasEngineBackend, "d3": D3Backend, "pplx": PplxBackend,
            "olmo": ReasoningBackend}


def load_backend(teacher, pin, snapshot, adapter=None, device="cuda", options=None):
    if teacher not in BACKENDS:
        raise SystemExit(f"no backend for {teacher!r}")
    if pin.get("family") != teacher:
        raise SystemExit(f"pin family {pin.get('family')!r} != {teacher!r}")
    return BACKENDS[teacher](teacher, pin, snapshot, adapter, device, dict(options or {}))
