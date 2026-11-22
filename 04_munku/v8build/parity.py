# renderer parity: rebuild existing v7.5 rows (5 families) from their parsed state / instructions / options with build_manifest_v3's
# unit + claude_lmscore.build_prompt (the v2e copy) and the E1 tokenizer; prompts must be byte-identical.
import json, sys, collections
sys.path[:0] = ["/tmp/c2e/esbar_code_v2e/trainers", "/data/local/v8build/scripts"]
import claude_lmscore as clm, build_manifest_v3 as BM
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data/local/v8build/tok")
fams = {"train_newyorker": 0, "train_hover": 0, "train_winogrande_xl": 0, "train_isarcasm": 0, "train_esci": 0}
ok = collections.Counter(); bad = collections.Counter()
for l in open("/data/local/v75build/esbar_stage1_v75.jsonl"):
    if not any(f'"source_family":"{f}"' in l or f'"source_family": "{f}"' in l for f in fams): continue
    r = json.loads(l); f = r["source_family"]
    if fams[f] >= 40: continue
    fams[f] += 1
    p = r["prompt"]; io_ = p.rfind("\n\nOptions:\n"); iq = p.rfind("\n\nQuestion: ", 0, io_)
    st = p[p.index("<|turn>user\n") + 12:iq]; ins = p[iq + 12:io_]
    import re
    opts = [m.group(1) for m in re.finditer(r"^[A-Za-z]\) (.*)$", p[io_ + 11:p.rfind("\n\nAnswer with")], re.M)]
    keys = list(r["option_order"])                      # shown order; render with identity order over these keys
    unit = {"state": st, "instructions": ins, "qtype": "choice", "option_keys": keys, "noul_criteria": None, "option_texts": opts}
    res = clm.build_prompt(tok, unit, list(range(len(keys))), max_len=32768, noul_defs=True)
    q = res[0] if isinstance(res, tuple) else res
    (ok if q == p else bad)[f] += 1
    if q != p and bad[f] == 1:
        i = next(i for i in range(min(len(p), len(q))) if p[i] != q[i]) if p != q else -1
        print("DIFF", f, i, repr(p[max(0, i-60):i+60]), "||", repr(q[max(0, i-60):i+60]))
print("PARITY ok", dict(ok), "bad", dict(bad))
