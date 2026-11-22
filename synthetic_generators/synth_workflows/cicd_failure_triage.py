"""CI/CD failure triage: a build engineer or release captain reads a failed pipeline and decides what it means.

Six schemas: a failed pull-request check, a red main branch, a halted progressive deploy, a nightly benchmark
regression alert, a broken dependency update, and a request to quarantine a flaky test. The actors are human
engineers and CI systems (runners, merge queues, canary analysers, dependency bots, flake trackers). Each schema
samples a `regime` first and lets it steer the factors, so most cases carry a flake rate that is small but not
zero, an error that could be the runner or the code, a delta just past the noise band, or a missing rerun or a
truncated log. Change-board approvals live in it_change_approval; this module is about diagnosing failures and
gating merges and releases. There are no security-alert or "is this malicious" questions.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    fmt_ts,
    hexid,
    kv_lines,
    maybe,
    near,
    noisy,
    noul,
    person,
    pick,
    pick_w,
    prune,
    rand_ts,
    regime,
    score,
    show,
    table,
)

# =========================================================================== shared data
# Every repo, team and test is invented. Per language: each test's repo, how critical the code it covers is,
# the code it covers (subject), a file next to it (neighbour), a file far away (unrelated), the owning team,
# and the assertion / exception lines a real run of that test would print.
TESTS = {
    "python": [
        {"id": "tests/integration/test_order_sync.py::test_retry_on_conflict", "repo": "northgate/ingest-pipeline",
         "crit": "core_feature", "kind": "integration", "team": "Order Sync", "subject": "ingest/order_sync.py",
         "neighbour": "ingest/retry_policy.py", "unrelated": "ranker/features/geo.py",
         "assertion": "AssertionError: assert 2 == 3 (len(attempts))", "exception": "KeyError: 'etag'"},
        {"id": "tests/unit/test_scoring.py::test_tie_break_is_stable", "repo": "northgate/ranker",
         "crit": "secondary_feature", "kind": "unit", "team": "Search Relevance", "subject": "ranker/scoring.py",
         "neighbour": "ranker/features/freshness.py", "unrelated": "ingest/cli.py",
         "assertion": "AssertionError: ['sku-118', 'sku-042'] != ['sku-042', 'sku-118']",
         "exception": "ZeroDivisionError: float division by zero"},
    ],
    "go": [
        {"id": "TestReconcileLedger/partial_batch", "repo": "ledgerline/api-core", "crit": "revenue_path",
         "kind": "unit", "team": "Ledger", "subject": "internal/ledger/reconcile.go",
         "neighbour": "internal/ledger/batch.go", "unrelated": "cmd/admin/flags.go",
         "assertion": "reconcile_test.go:88: got 41 entries, want 42",
         "exception": "panic: runtime error: invalid memory address or nil pointer dereference"},
        {"id": "TestSettlementWindow_DST", "repo": "ledgerline/settlement-worker", "crit": "revenue_path",
         "kind": "integration", "team": "Settlement", "subject": "internal/window/window.go",
         "neighbour": "internal/clock/clock.go", "unrelated": "internal/httpapi/health.go",
         "assertion": "window_test.go:131: window end = 01:00 UTC, want 02:00 UTC",
         "exception": "panic: time: missing Location in call to Time.In"},
    ],
    "typescript": [
        {"id": "checkout/discounts.spec.ts > applies stacked discounts", "repo": "harborfront/checkout-ui",
         "crit": "revenue_path", "kind": "unit", "team": "Checkout", "subject": "src/checkout/discounts.ts",
         "neighbour": "src/checkout/cart-store.ts", "unrelated": "src/account/avatar-upload.tsx",
         "assertion": "expect(received).toBe(expected) Expected: 1840 Received: 1875",
         "exception": "TypeError: Cannot read properties of undefined (reading 'rate')"},
        {"id": "e2e/guest-checkout.spec.ts > completes purchase as guest", "repo": "harborfront/storefront-web",
         "crit": "revenue_path", "kind": "e2e", "team": "Checkout", "subject": "src/checkout/guest-flow.tsx",
         "neighbour": "src/checkout/address-form.tsx", "unrelated": "src/blog/post-list.tsx",
         "assertion": "expect(locator).toHaveText(expected) Expected: \"Order confirmed\" Received: \"Something went wrong\"",
         "exception": "TypeError: Cannot destructure property 'id' of 'order' as it is undefined."},
    ],
    "java": [
        {"id": "SubscriptionRenewalIT > renewsWithProration()", "repo": "tidewater/subscriptions",
         "crit": "revenue_path", "kind": "integration", "team": "Subscriptions",
         "subject": "src/main/java/com/tidewater/subs/RenewalService.java",
         "neighbour": "src/main/java/com/tidewater/subs/ProrationCalculator.java",
         "unrelated": "src/main/java/com/tidewater/admin/AuditExportController.java",
         "assertion": "org.opentest4j.AssertionFailedError: expected: <29.97> but was: <30.00>",
         "exception": "java.lang.NullPointerException: Cannot invoke \"Plan.getTier()\" because \"plan\" is null"},
    ],
    "kotlin": [
        {"id": "SyncWorkerTest > retriesAfterNetworkLoss", "repo": "quarry/field-app", "crit": "core_feature",
         "kind": "unit", "team": "Mobile Sync", "subject": "app/src/main/kotlin/sync/SyncWorker.kt",
         "neighbour": "app/src/main/kotlin/sync/Backoff.kt",
         "unrelated": "app/src/main/kotlin/settings/ThemePicker.kt",
         "assertion": "java.lang.AssertionError: expected attempts:<3> but was:<1>",
         "exception": "java.lang.IllegalStateException: WorkManager is not initialized properly."},
    ],
    "rust": [
        {"id": "cache::lru::tests::evicts_under_pressure", "repo": "emberline/edge-cache", "crit": "core_feature",
         "kind": "unit", "team": "Edge Platform", "subject": "src/cache/lru.rs", "neighbour": "src/cache/mod.rs",
         "unrelated": "src/cli/args.rs", "assertion": "assertion `left == right` failed: left: 511, right: 512",
         "exception": "called `Option::unwrap()` on a `None` value"},
    ],
}
RUN_CMD = {"python": "pytest -q -n 4 --maxfail=20", "go": "go test ./... -race -count=1",
           "typescript": "npx jest --ci --shard={shard}", "java": "./gradlew integrationTest --no-daemon",
           "kotlin": "./gradlew :app:testDebugUnitTest", "rust": "cargo test --locked --all-features"}
RUNNER_POOLS = ["gha-ubuntu-22.04-8core", "bk-linux-x64-large", "ci-arm64-medium", "sh-docker-c6i-4xl"]
INFRA_KINDS = ("dns", "disk_full")


def _repo_and_test(rng, langs=None):
    pool = [(lang, t) for lang, ts in TESTS.items() for t in ts if langs is None or lang in langs]
    lang, t = pick(rng, pool)
    return t["repo"], lang, t


def _name_unlike(rng, *taken):
    """A fresh fictional name whose first name differs from every name in `taken` (chat handles stay distinct)."""
    firsts = {t.split()[0].lower() for t in taken if t}
    n = person(rng)
    for _ in range(20):
        if n.split()[0].lower() not in firsts:
            break
        n = person(rng)
    return n


def _sha(rng):
    return hexid(rng, 7)


def _stem(path):
    parts = path.split("/")
    stem = parts[-1].split(".")[0]
    return parts[-2] if stem in ("mod", "index", "__init__") and len(parts) > 1 else stem


def _dur(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _log_ts(t, ms):
    return t.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms % 1000:03d}Z"


def _error_line(kind, lang, t, port=5432):
    """The line a runner prints for one failure kind, in the test framework's own dialect."""
    if kind == "assertion":
        return t["assertion"]
    if kind == "exception":
        return t["exception"]
    if kind == "timeout":
        return {"go": "panic: test timed out after 10m0s",
                "typescript": "thrown: \"Exceeded timeout of 30000 ms for a test.\"",
                "rust": f"test {t['id']} has been running for over 60 seconds; job killed at 900s",
                "python": "Failed: Timeout >300.0s"}.get(lang, "Test worker exceeded the 600s limit and was stopped")
    if kind == "conn_refused":
        return {"go": f"dial tcp 127.0.0.1:{port}: connect: connection refused",
                "python": f"ConnectionRefusedError: [Errno 111] Connection refused ('127.0.0.1', {port})",
                "typescript": f"Error: connect ECONNREFUSED 127.0.0.1:{port}",
                "rust": "Os { code: 111, kind: ConnectionRefused, message: \"Connection refused\" }"
                }.get(lang, f"java.net.ConnectException: Connection refused (localhost:{port})")
    if kind == "oom":
        return "Killed (signal 9); container memory limit 7.5GiB reached"
    if kind == "dns":
        return {"python": "socket.gaierror: [Errno -3] Temporary failure in name resolution (artifacts.ci.example.net)",
                "go": "dial tcp: lookup artifacts.ci.example.net: i/o timeout"
                }.get(lang, "getaddrinfo EAI_AGAIN artifacts.ci.example.net")
    return {"python": "OSError: [Errno 28] No space left on device: '/tmp/pytest-of-runner'",
            "go": "write /tmp/go-build3301/b412/_pkg_.a: no space left on device"
            }.get(lang, "ENOSPC: no space left on device, write '/home/runner/work/_temp/cache'")


def _exit_code(kind, lang):
    if kind == "oom":
        return 137
    if kind == "timeout":
        return 124 if lang in ("rust", "java", "kotlin") else 1
    if lang == "rust":
        return 101
    if kind == "exception" and lang == "go":
        return 2
    return 1


def _fail_block(lang, test_id, err, secs):
    """How the framework reports one failed test (list of lines)."""
    if lang == "python":
        return [f"FAILED {test_id} - {err}"]
    if lang == "go":
        return [f"--- FAIL: {test_id} ({secs:.2f}s)", f"    {err}", "FAIL"]
    if lang == "typescript":
        return [f"  ● {test_id}", f"    {err}"]
    if lang == "rust":
        return [f"---- {test_id} stdout ----", f"thread '{test_id}' panicked: {err}", "test result: FAILED."]
    return [f"{test_id} FAILED", f"    {err}"]


# =========================================================================== schema 1
class PrCheckFailure(Schema):
    name = "pr_check_failure"
    description = "A required CI check failed on a pull request: what broke it, and may the pull request merge?"
    surfaces = {"json": 0.45, "checks_summary": 0.3, "ci_log": 0.25}

    NOTES = {
        "says_unrelated": ["this can't be me, I only touched {f}", "pretty sure this is the usual flake, my change "
                           "doesn't go near it", "unrelated to this PR afaict - can someone override the check?"],
        "suspects_own": ["hmm, might be mine, the {stem} change could affect ordering",
                         "looking into it, possibly my refactor of {stem}"],
        "asks_rerun": ["can we just re-run this?", "no idea what this is, can someone re-run it"],
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        repo, lang, t = _repo_and_test(rng)
        err = pick_w(rng, {"assertion": 3, "exception": 2, "timeout": 3, "conn_refused": 2, "oom": 1, "dns": 1,
                           "disk_full": 1})
        overlap = pick_w(rng, {"subject": 3, "neighbour": 3, "unrelated": 3, None: 1})
        per50 = pick_w(rng, {0: 3, 1: 2, 2: 2, 3: 1, 5: 1})       # failures per 50 main-branch runs
        rerun = pick_w(rng, {None: 3, "passed": 3, "failed_same": 2, "failed_other": 1})
        main = pick_w(rng, {"green": 6, "red_same_test": 1, "red_other_job": 1, None: 1})
        pool_fail = pick(rng, [0, 0, 0, 1, 2, 4])
        truncated = chance(rng, 0.15)
        note = pick_w(rng, {None: 3, "says_unrelated": 3, "suspects_own": 1, "asks_rerun": 1})
        twist = None
        if reg == "clear":
            arche = pick(rng, ["code", "infra", "flake"])
            if arche == "code":
                err, overlap, per50, rerun, main = pick(rng, ["assertion", "exception"]), "subject", 0, "failed_same", "green"
                pool_fail, truncated = 0, False
                note = pick(rng, [None, "suspects_own"])
            elif arche == "infra":
                err, overlap, rerun, per50 = pick(rng, list(INFRA_KINDS)), "unrelated", "passed", 0
                pool_fail = pick(rng, [6, 11, 17])
            else:
                err, overlap, per50, rerun, main = "timeout", "unrelated", pick(rng, [7, 9, 12]), "passed", "green"
        elif reg == "borderline":
            per50 = pick(rng, [1, 2, 2, 3])
            err = pick(rng, ["timeout", "conn_refused", "oom", "assertion"])
            overlap = pick_w(rng, {"neighbour": 3, "subject": 1, "unrelated": 1})
            rerun = pick_w(rng, {None: 3, "passed": 2, "failed_other": 1})
        else:
            twist = pick(rng, ["flaky_but_touched", "author_denies", "main_already_red", "noisy_pool_code_error"])
            if twist == "flaky_but_touched":
                per50, overlap, err, rerun = pick(rng, [4, 6, 8]), "subject", pick(rng, ["assertion", "timeout"]), pick(rng, [None, "passed"])
            elif twist == "author_denies":
                overlap, err, note, rerun = "subject", "exception", "says_unrelated", pick(rng, ["passed", None])
            elif twist == "main_already_red":
                main, overlap, err, per50 = "red_same_test", pick(rng, ["subject", "neighbour"]), "assertion", 0
            else:
                pool_fail, err = pick(rng, [5, 8, 13]), pick(rng, ["assertion", "conn_refused"])
                overlap = pick(rng, ["neighbour", "subject"])
            truncated = chance(rng, 0.4)
        if t["kind"] == "unit":            # a unit test does not open sockets or resolve hosts
            err = {"conn_refused": "timeout", "dns": "disk_full"}.get(err, err)
        window = pick(rng, [50, 50, 100, 200])
        fails = 0 if per50 == 0 else max(1, round(per50 * window / 50 * rng.uniform(0.8, 1.2)))
        history_known = reg == "clear" or chance(rng, 0.85)
        lines_changed = pick(rng, [4, 12, 38, 96, 240, 610])
        if err == "timeout":
            job_secs = {"go": 600, "python": 300}.get(lang, 900) + rng.randint(5, 40)
        else:
            job_secs = rng.randint(90, 1400)
        return {"regime": reg, "twist": twist, "repo": repo, "language": lang, "test": t["id"],
                "test_kind": t["kind"], "test_team": t["team"], "error_kind": err, "diff_overlap": overlap,
                "fails_on_main": fails if history_known else None, "history_window": window if history_known else None,
                "rerun": rerun, "main_status": main, "pool_infra_failures_today": pool_fail,
                "log_truncated": truncated, "error_visible": not (truncated and chance(rng, 0.5)),
                "author_note": note, "hours_to_release_cut": maybe(rng, pick(rng, [3, 8, 20, 48]), 0.6),
                "pr": rng.randint(1200, 9800), "author": person(rng), "lines_changed": lines_changed,
                "job_seconds": job_secs, "runner": f"{pick(rng, RUNNER_POOLS)}-{rng.randint(1, 64):03d}",
                "shard": pick(rng, ["1/4", "2/4", "3/6", "5/6", "2/8"]), "port": pick(rng, [5432, 6379, 9092, 8081]),
                "started": rand_ts(rng, hours=(7, 22)), "head_sha": _sha(rng)}

    def _parts(self, f, rng):
        t = next(x for x in TESTS[f["language"]] if x["id"] == f["test"])
        path = t[f["diff_overlap"]] if f["diff_overlap"] else None
        files = None
        if path:
            files = [path]
            if f["diff_overlap"] == "subject" and chance(rng, 0.5):
                files.append(t["neighbour"])
            if chance(rng, 0.4):
                files.append(pick(rng, ["CHANGELOG.md", "README.md", ".github/CODEOWNERS"]))
        stem = _stem(path) if path else "the module"
        title = pick(rng, [f"Refactor {stem} error handling", f"Handle empty input in {stem}",
                           f"{stem}: small cleanup", f"Tidy up {stem} logging", f"Speed up {stem}"])
        note = None
        if f["author_note"]:
            note = noisy(pick(rng, self.NOTES[f["author_note"]]).format(f=path or "one file", stem=stem), rng, 0.15)
        err = _error_line(f["error_kind"], f["language"], t, f["port"]) if f["error_visible"] else None
        job = {"unit": "unit-tests", "integration": f"integration ({f['shard']})", "e2e": "e2e-chromium"}[f["test_kind"]]
        return t, files, title, note, err, job

    def _rerun_text(self, f):
        return {None: "not re-run yet", "passed": "re-run #1 passed", "failed_same": "re-run #1 failed the same way",
                "failed_other": "re-run #1 failed in a different test"}[f["rerun"]]

    def _main_text(self, f):
        return {"green": "green", "red_same_test": "red on this same test",
                "red_other_job": "red on a different job", None: "not checked"}[f["main_status"]]

    def render(self, f, rng, surface):
        t, files, title, note, err, job = self._parts(f, rng)
        code = _exit_code(f["error_kind"], f["language"])
        hist = (f"{f['fails_on_main']} failures in last {f['history_window']} main runs"
                if f["fails_on_main"] is not None else None)
        if surface == "json":
            return dumps(prune({
                "pull_request": {"repo": f["repo"], "number": f["pr"], "title": title, "author": f["author"],
                                 "head_sha": f["head_sha"], "lines_changed": f["lines_changed"],
                                 "files_changed": files, "author_comment": note},
                "failed_check": {"job": job, "runner": f["runner"], "exit_code": code,
                                 "duration_s": f["job_seconds"], "test": f["test"], "error_excerpt": err,
                                 "log_truncated": True if f["log_truncated"] else None},
                "test_history": ({"main_runs": f["history_window"], "failures": f["fails_on_main"]}
                                 if f["fails_on_main"] is not None else None),
                "rerun": {"result": f["rerun"]} if f["rerun"] else {"attempted": False},
                "context": {"main_branch": f["main_status"],
                            "infra_failures_on_runner_pool_today": f["pool_infra_failures_today"],
                            "hours_to_release_branch_cut": f["hours_to_release_cut"]},
            }))
        if surface == "checks_summary":
            others = [("lint", rng.randint(25, 70)), ("build", rng.randint(120, 400)), ("typecheck", rng.randint(40, 150))]
            ok = pick(rng, ["✓", "pass", "[ok]"])
            bad = {"✓": "✗", "pass": "FAIL", "[ok]": "[x]"}[ok]
            lines = [f"PR #{f['pr']} · {title}",
                     f"{f['repo']} · {f['author']} · {f['lines_changed']} lines changed · head {f['head_sha']}", "",
                     f"Checks: 1 failing, {len(others)} passing",
                     f"  {bad} {job} (required) - failed after {_dur(f['job_seconds'])} on {f['runner']}"]
            lines += [f"  {ok} {n} - {_dur(s)}" for n, s in others]
            lines += ["", f"Failed test: {f['test']}",
                      f"  {err}" if err else "  (no error captured - log was truncated before the failure)",
                      f"  exit code {code}", ""]
            lines.append(f"Files: {', '.join(files)}" if files else "Files: diff not loaded")
            lines.append(f"Flake tracker: {hist}" if hist else "Flake tracker: no data for this test")
            lines.append(f"Re-run: {self._rerun_text(f)}")
            lines.append(f"main branch: {self._main_text(f)}")
            if f["pool_infra_failures_today"]:
                lines.append(f"Runner pool: {f['pool_infra_failures_today']} jobs lost to infra errors today")
            if f["hours_to_release_cut"]:
                lines.append(f"Release branch cut in {f['hours_to_release_cut']} h")
            if note:
                lines += ["", f"{f['author'].split()[0].lower()} commented: {note}"]
            return "\n".join(lines)
        # ci_log
        t0 = f["started"]
        ms = rng.randrange(1000)
        out = [f"{_log_ts(t0, ms)} ##[group]Run {RUN_CMD[f['language']].format(shard=f['shard'])}",
               f"{_log_ts(t0 + timedelta(seconds=1), ms + 7)} runner: {f['runner']}  sha: {f['head_sha']}  pr: #{f['pr']}"]
        if f["log_truncated"]:
            out.append(f"[... {rng.randint(1200, 9000):,} lines omitted: log exceeded 4 MB ...]")
        tf = t0 + timedelta(seconds=f["job_seconds"] - 2)
        if err:
            out += [f"{_log_ts(tf, ms + 311)} {x}" for x in _fail_block(f["language"], f["test"], err,
                                                                   600.0 if f["error_kind"] == "timeout"
                                                                   else f["job_seconds"] * 0.4)]
        else:
            out.append(f"{_log_ts(tf, ms + 311)} {f['test']} ... (output cut)")
        out.append(f"{_log_ts(tf + timedelta(seconds=2), ms + 402)} ##[error]Process completed with exit code {code}.")
        tail = kv_lines([("flake-tracker", hist or "no data"), ("retry", self._rerun_text(f)),
                         ("main branch", self._main_text(f)),
                         ("pool infra errors today", f["pool_infra_failures_today"] or None),
                         ("changed files", ", ".join(files) if files else None),
                         ("release cut", f"in {f['hours_to_release_cut']} h" if f["hours_to_release_cut"] else None)],
                        rng, style=pick(rng, ["colon", "dash"]))
        s = "\n".join(out) + "\n\n" + tail
        if note:
            s += f"\n\nPR comment ({f['author']}): {note}"
        return s

    def questions(self):
        return {
            "failure_origin": choice("What most likely made this check fail?", {
                "change_under_test": "The pull request's own code change broke the behaviour the test checks.",
                "known_flaky_test": "The test fails intermittently by itself, whatever change is under test.",
                "ci_infrastructure": "The runner, network, disk or another piece of CI infrastructure failed "
                                     "underneath the job.",
                "already_broken_on_main": "The main branch already has this failure and the pull request inherited it.",
                "undetermined": "The evidence at hand does not single out any one cause yet.",
            }),
            "merge_gate": choice("What should happen to this pull request's merge right now?", {
                "allow_merge": "Let it merge; the red check is not attributable to this change.",
                "rerun_then_decide": "Re-run the failed job and decide on the strength of that result.",
                "block_for_fix": "Keep it blocked until the author changes the code.",
                "hold_for_main": "Park it until the main branch is repaired, then rebase and test again.",
            }),
            "change_implication": score("How strongly does the evidence tie the failure to this pull request's diff?", [
                "Not at all: the diff has nothing to do with what failed.",
                "Weakly: a loose connection, but other causes fit better.",
                "Moderately: the diff plausibly reaches the failing code, though other explanations remain.",
                "Strongly: the diff and the failure line up closely.",
            ]),
            "diagnosis_confidence": score("How confidently can a build engineer diagnose this failure from what is here?", [
                "Guesswork: the error text is missing, or two or more of the rerun result, flake history and "
                "main-branch status are missing.",
                "Tentative: exactly one of the rerun result, flake history or main-branch status is missing.",
                "Reasonable: the error, rerun, flake history and main-branch status are all there, but they point "
                "different ways.",
                "Firm: all of that evidence is there and tells one story (own code, a known flake, infrastructure, "
                "or main already broken).",
            ]),
            "rerun_would_pass": noul(
                "Running the same commit again would most likely turn this check green.",
                yes="The failure looks intermittent or environmental, so an identical run would probably pass.",
                no="The failure looks deterministic for this commit and would come back on a rerun."),
        }


# =========================================================================== schema 2
TEAMS = ["Checkout", "Ledger", "Search Relevance", "Mobile Sync", "Edge Platform", "Identity", "Messaging",
         "Data Ingest", "Growth", "Internal Tools", "Settlement", "Order Sync", "Subscriptions"]
BUMPS = {"go": ("github.com/quillsoft/retry", "v1.8.2", "v1.9.0"), "python": ("pydantix", "2.6.4", "2.7.0"),
         "typescript": ("tzfmt", "3.2.1", "3.3.0"), "java": ("com.lattice:json-bind", "4.1.0", "4.2.0"),
         "kotlin": ("org.quillsoft:retry4", "4.0.3", "4.1.0"), "rust": ("bytepool", "0.9.4", "0.10.0")}
UNRELATED_MSGS = ["Add metrics for export latency", "Bump log retention in dev config", "Fix typo in onboarding copy",
                  "Remove dead feature flag old_nav", "Update CODEOWNERS for the admin area",
                  "Add pagination to admin list view", "Reword error message on login form",
                  "Document the local setup for arm64 laptops", "Add dark-mode colours to the settings page",
                  "Log request ids on 4xx responses", "Split the admin routes file in two",
                  "Raise the default page size for exports", "Drop unused analytics events"]
PLATFORM_CHANGES = ["runner image ubuntu-22.04 {v} rolled out to the pool at {hm}",
                    "shared build cache moved to a new storage bucket at {hm}",
                    "service-container sidecar upgraded (postgres 15 -> 16) at {hm}"]


class MainBranchRed(Schema):
    name = "main_branch_red"
    description = "The main branch went red after a batch of merges: find what broke it, who owns it, and how to get green."
    surfaces = {"json": 0.45, "buildcop_chat": 0.3, "status_page": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        repo, lang, t = _repo_and_test(rng)
        cause = pick_w(rng, {"code": 5, "dependency": 2, "ci_platform": 2, "flaky": 2})
        n = pick(rng, [2, 3, 4, 5, 7, 9])
        bisect = pick_w(rng, {"not_started": 3, "one_commit": 3, "two_commits": 2, "inconclusive": 2})
        reruns = pick(rng, [(2, 2), (3, 3), (1, 2), (2, 3), (1, 3), None])
        also_release = maybe(rng, chance(rng, 0.15), 0.4)
        ci_change = chance(rng, 0.2)
        prior = pick(rng, [0, 0, 1, 2, 4, 9])
        err = pick_w(rng, {"assertion": 4, "exception": 3, "timeout": 2, "conn_refused": 1, "oom": 1})
        revert = pick_w(rng, {"clean": 3, "conflicts": 2, "not_tried": 3})
        author_status = pick_w(rng, {"online": 5, "offline_timezone": 3, "on_leave": 1})
        hotfix = chance(rng, 0.15)
        twist = None
        if reg == "clear":
            arche = pick(rng, ["code", "code", "platform", "flaky"])
            if arche == "code":
                cause, bisect, reruns, also_release, ci_change, prior = "code", "one_commit", (3, 3), False, False, 0
                err, revert = pick(rng, ["assertion", "exception"]), "clean"
            elif arche == "platform":
                cause, also_release, ci_change, bisect = "ci_platform", True, True, "inconclusive"
                err, reruns = pick(rng, ["conn_refused", "oom", "timeout"]), (3, 3)
            else:
                cause, reruns, prior, bisect, err, ci_change = "flaky", (1, 3), 9, "inconclusive", "timeout", False
        elif reg == "borderline":
            bisect = pick(rng, ["two_commits", "not_started", "two_commits"])
            reruns, prior, n = pick(rng, [(2, 3), (1, 2), None]), pick(rng, [1, 2, 2]), pick(rng, [4, 5, 7])
        else:
            twist = pick(rng, ["bisect_vs_diff", "platform_and_code", "revert_conflicts", "flaky_test_touched"])
            if twist == "bisect_vs_diff":
                cause, bisect, n = "code", "one_commit", max(n, 3)
            elif twist == "platform_and_code":
                cause, ci_change, reruns, also_release = pick(rng, ["code", "ci_platform"]), True, (3, 3), None
            elif twist == "revert_conflicts":
                cause, bisect, revert, author_status, hotfix = "code", "one_commit", "conflicts", "on_leave", True
            else:
                cause, prior, reruns = "flaky", pick(rng, [4, 6, 9]), (2, 2)
        if t["kind"] == "unit" and err == "conn_refused":
            err = "timeout"
        # the commits between the last green run and the first red one
        culprit = rng.randrange(n)
        commits = []
        for i in range(n):
            touches = "unrelated"
            if i == culprit and cause == "code":
                touches = "subject" if chance(rng, 0.7) else "neighbour"
            elif i == culprit and cause == "dependency":
                touches = "lockfile"
            elif i == culprit and twist in ("flaky_test_touched", "platform_and_code"):
                touches = "subject"
            commits.append({"sha": _sha(rng), "author": person(rng), "touches": touches,
                            "team": t["team"] if touches == "subject" and chance(rng, 0.6) else pick(rng, TEAMS)})
        if cause in ("ci_platform", "flaky") and bisect == "one_commit" and reg != "clear":
            target = [rng.randrange(n)]                      # an intermittent failure bisects to a bystander
        elif bisect == "one_commit":
            target = [culprit]
            if twist == "bisect_vs_diff":
                target = [(culprit + 1 + rng.randrange(n - 1)) % n]
        elif bisect == "two_commits" and n >= 2:
            target = sorted([culprit, (culprit + 1 + rng.randrange(n - 1)) % n])
        else:
            target = []
            if bisect == "two_commits":
                bisect = "inconclusive"
        suspect = target[0] if target else next((i for i, c in enumerate(commits) if c["touches"] != "unrelated"), None)
        if suspect is None:
            revert, author_status = "not_tried", None
        start = rand_ts(rng, hours=(6, 20))
        return {"regime": reg, "twist": twist, "cause": cause, "repo": repo, "language": lang, "test": t["id"],
                "test_kind": t["kind"], "test_team": t["team"], "error_kind": err, "commits": commits,
                "culprit_index": culprit, "bisect": bisect, "bisect_target": target, "suspect_index": suspect,
                "reruns": reruns, "fails_on_release_branch": also_release, "ci_platform_change": ci_change,
                "prior_failures_100": prior, "revert_attempt": revert, "suspect_author_status": author_status,
                "hours_red": pick(rng, [0.5, 1.5, 3, 6, 11, 19]), "merge_queue": pick(rng, [0, 2, 5, 11, 24]),
                "release_train_blocked": chance(rng, 0.3), "hotfix_waiting": hotfix,
                "last_green": _sha(rng), "run_id": rng.randint(40000, 99999), "red_since": start,
                "platform_change_text": pick(rng, PLATFORM_CHANGES).format(
                    v=f"{start.year}{start.month:02d}{start.day:02d}.{rng.randint(1, 4)}",
                    hm=(start - timedelta(minutes=rng.randint(10, 90))).strftime("%H:%M")),
                "buildcop": _name_unlike(rng, *[c["author"] for c in commits])}

    def _message(self, c, f, rng, spare):
        t = next(x for x in TESTS[f["language"]] if x["id"] == f["test"])
        if c["touches"] == "subject":
            stem = _stem(t["subject"])
            return pick(rng, [f"Refactor {stem} to share one client", f"{stem}: handle partial batches",
                              f"Cache lookups in {stem}", f"Rework ordering in {stem}"])
        if c["touches"] == "neighbour":
            return pick(rng, [f"Clean up {_stem(t['neighbour'])}", f"Rename helpers in {_stem(t['neighbour'])}"])
        if c["touches"] == "lockfile":
            pkg, a, b = BUMPS[f["language"]]
            return f"chore(deps): bump {pkg} from {a} to {b}"
        return spare.pop()

    def _texts(self, f, rng):
        t = next(x for x in TESTS[f["language"]] if x["id"] == f["test"])
        err = _error_line(f["error_kind"], f["language"], t)
        spare = rng.sample(UNRELATED_MSGS, len(UNRELATED_MSGS))
        msgs = [self._message(c, f, rng, spare) for c in f["commits"]]
        shas = [f["commits"][i]["sha"] for i in f["bisect_target"]]
        bis = {"not_started": "not started", "inconclusive": "inconclusive (runs disagreed)",
               "one_commit": f"points to {shas[0]}" if shas else "",
               "two_commits": f"narrowed to {' / '.join(shas)}"}[f["bisect"]]
        rr = f"{f['reruns'][0]} of {f['reruns'][1]} reruns failed" if f["reruns"] else None
        s = f["suspect_index"]
        sus = f["commits"][s] if s is not None else None
        rev = None
        if sus is not None and f["revert_attempt"] != "not_tried":
            rev = (f"revert of {sus['sha']} applies cleanly" if f["revert_attempt"] == "clean"
                   else f"revert of {sus['sha']} conflicts with two later commits")
        return err, msgs, bis, rr, sus, rev

    def render(self, f, rng, surface):
        err, msgs, bis, rr, sus, rev = self._texts(f, rng)
        job = {"unit": "unit-tests", "integration": "integration", "e2e": "e2e-chromium"}[f["test_kind"]]
        status = {"online": "online", "offline_timezone": "offline (their night)", "on_leave": "on leave until Monday",
                  None: None}[f["suspect_author_status"]]
        platform = f["platform_change_text"] if f["ci_platform_change"] else None
        if surface == "json":
            return dumps(prune({
                "build": {"repo": f["repo"], "branch": "main", "pipeline_run": f["run_id"], "failing_job": job,
                          "failing_test": f["test"], "test_owner": f["test_team"], "error": err,
                          "red_since_utc": fmt_ts(f["red_since"], style="iso"), "hours_red": f["hours_red"],
                          "last_green_sha": f["last_green"]},
                "commits_since_green": [{"sha": c["sha"], "author": c["author"], "team": c["team"], "message": m}
                                        for c, m in zip(f["commits"], msgs)],
                "investigation": {"bisect": bis, "reruns": rr, "fails_on_release_branch": f["fails_on_release_branch"],
                                  "test_failures_prior_100_main_runs": f["prior_failures_100"],
                                  "ci_platform_changes": platform, "revert": rev,
                                  "suspect_author": status},
                "impact": {"merge_queue_waiting": f["merge_queue"],
                           "release_train_blocked": f["release_train_blocked"],
                           "hotfix_waiting_on_green": f["hotfix_waiting"] or None},
            }))
        if surface == "buildcop_chat":
            sus_author = sus["author"] if sus is not None else None
            cop = f["buildcop"].split()[0].lower()
            others = [_name_unlike(rng, f["buildcop"], sus_author)]
            others.append(_name_unlike(rng, f["buildcop"], sus_author, others[0]))
            t = f["red_since"]
            msgs_out = [("ci-bot", f"main is red: {job} failing since {f['commits'][-1]['sha']} "
                                   f"({f['test']}). last green {f['last_green']}, {len(f['commits'])} commits in range")]
            msgs_out.append((cop, f"looking. error is `{err}`"))
            if rr:
                msgs_out.append((cop, rr))
            if f["prior_failures_100"]:
                msgs_out.append((cop, f"flake tracker has this test failing {f['prior_failures_100']}x in the "
                                      f"100 runs before today"))
            if f["bisect"] != "not_started":
                msgs_out.append((cop, f"bisect {bis}" + (f" ({sus['author']}, \"{msgs[f['suspect_index']]}\")"
                                                         if f["bisect"] == "one_commit" and sus else "")))
            if platform:
                msgs_out.append((others[0].split()[0].lower(), f"fyi platform side: {platform}"))
            if f["fails_on_release_branch"] is not None:
                msgs_out.append((cop, "release branch run " + ("is also failing on it" if f["fails_on_release_branch"]
                                                               else "is green")))
            if rev:
                msgs_out.append((cop, rev))
            if sus is not None and f["suspect_author_status"] == "online":
                msgs_out.append((sus["author"].split()[0].lower(),
                                 pick(rng, ["looking, not obvious to me how my change hits that test",
                                            "ugh, could be mine. give me 30 min", "I can revert if you want"])))
            elif sus is not None and status:
                msgs_out.append((cop, f"{sus['author']} is {status}"))
            if f["merge_queue"] >= 5:
                msgs_out.append((others[1].split()[0].lower(), f"{f['merge_queue']} PRs stuck in the merge queue, ETA?"))
            if f["release_train_blocked"]:
                msgs_out.append(("release-bot", "today's release train is waiting for a green main"))
            if f["hotfix_waiting"]:
                msgs_out.append((others[0].split()[0].lower(), "we have a hotfix that needs main green to ship"))
            out, day = [], t.date()
            for k, (who, text) in enumerate(msgs_out):
                t += timedelta(minutes=rng.randint(1, 12))
                if k == 1:                    # the thread picks up again in the last hour or so
                    t += timedelta(minutes=max(0, int(f["hours_red"] * 60) - 75))
                stamp = t.strftime("%H:%M") if t.date() == day else t.strftime("%b ") + str(t.day) + t.strftime(" %H:%M")
                day = t.date()
                out.append(f"[{stamp}] {who}: {noisy(text, rng, 0.08) if who not in ('ci-bot', 'release-bot') else text}")
            return f"#main-is-red  ({f['repo']}, red for {f['hours_red']} h)\n" + "\n".join(out)
        # status_page
        rows = [[c["sha"], c["author"], c["team"], m, "<- bisect" if i in f["bisect_target"] else ""]
                for i, (c, m) in enumerate(zip(f["commits"], msgs))]
        head = [f"## main is red: {f['repo']}", "",
                f"Failing `{job}` / `{f['test']}` (owner: {f['test_team']}) since "
                f"{fmt_ts(f['red_since'], rng)}, {f['hours_red']} h ago. Last green: `{f['last_green']}`.", "",
                f"    {err}", "", table(["sha", "author", "team", "message", "note"], rows, rng), ""]
        info = [f"Bisect: {bis}", f"Reruns: {show(rr, rng)}",
                f"Same test failed {f['prior_failures_100']} of the previous 100 main runs",
                f"Release branch: {show(f['fails_on_release_branch'], rng, lambda v: 'failing too' if v else 'green')}",
                f"Platform changes today: {platform}" if platform else None, f"Revert: {rev}" if rev else None,
                f"Suspect author: {status}" if status else None,
                f"Merge queue: {f['merge_queue']} waiting" + ("; release train blocked" if f["release_train_blocked"] else "")
                + ("; a hotfix is waiting" if f["hotfix_waiting"] else "")]
        return "\n".join(head) + "\n" + bullets([x for x in info if x], rng)

    def questions(self):
        return {
            "build_fix_path": choice("What is the best next move to get the main branch green again?", {
                "revert_suspect": "Revert the one commit bisect points to, since nothing stops a revert, and confirm "
                                  "main recovers.",
                "forward_fix": "Leave the history alone and have the author push a corrective commit, because "
                               "reverting the commit bisect points to conflicts with later work.",
                "quarantine_test": "Take the failing test out of the gate and track it separately: it was already "
                                   "failing now and then before these commits, and reruns disagree.",
                "rerun_and_watch": "Re-run the pipeline and watch: a CI platform change landed and the release branch "
                                   "fails too, so the commits may not be at fault.",
                "bisect_further": "Hold off on any change until bisect pins the culprit down to a single commit.",
            }),
            "route_to_team": choice("Which group should own fixing this red build?", {
                "suspect_commit_team": "The team that merged the commit most likely to have caused it.",
                "test_owners": "The team that owns the failing test and the code it exercises.",
                "ci_platform": "The CI platform group that runs the runners, images and shared build services.",
                "build_tooling": "The build tooling group that owns lockfiles, toolchains and dependency versions.",
            }),
            "culprit_certainty": score("How firmly does the evidence pin the breakage on a single commit?", [
                "Not at all: nothing points at any commit in the range.",
                "Loosely: several commits remain plausible.",
                "Partly: the field is down to one or two commits, but they are not told apart.",
                "Fairly: one commit stands out, but a test or signal disagrees.",
                "Firmly: bisect, reruns and the diff all point at the same commit.",
            ]),
            "pipeline_blockage": score("How badly is the red main branch holding up other work?", [
                "Barely: few merges waiting and nothing needs to ship from main today.",
                "Somewhat: a handful of pull requests are queued behind it.",
                "Heavily: the merge queue is backing up and engineers are working around it.",
                "Critically: a release train or a hotfix cannot ship until main is green.",
            ]),
            "clean_revert_available": noul(
                "Reverting the suspect commit would bring main back to green without breaking anything else.",
                yes="A single identified commit is to blame and its revert applies cleanly.",
                no="The culprit is unclear, the revert conflicts, or the cause lies outside the commits."),
        }


# =========================================================================== schema 3
# (service, tier 1 = most critical, owning team)
DEPLOY_SERVICES = [("checkout-api", 1, "Checkout"), ("session-auth", 1, "Identity"), ("ledger-writer", 1, "Ledger"),
                   ("catalog-search", 2, "Discovery"), ("notification-fanout", 2, "Messaging"),
                   ("partner-webhooks", 2, "Integrations"), ("media-resizer", 3, "Media"),
                   ("admin-console", 3, "Internal Tools")]
# metric -> (label, unit, usual baseline range, decimals, score points lost per unit of canary/baseline ratio)
CANARY_METRICS = {"error_rate": ("5xx ratio", "%", (0.05, 0.6), 2, 60),
                  "p99_latency": ("p99 latency", "ms", (120, 900), 0, 150),
                  "restarts": ("container restarts", "/h", (0.1, 0.5), 2, 40),
                  "cpu": ("CPU utilisation", "%", (30, 60), 0, 200)}
EXTERNAL_EVENTS = {"downstream_degraded": "status page of the {dep} provider shows degraded performance since {hm}",
                   "baseline_pod_restarted": "one baseline pod was rescheduled at {hm}, halfway through the window",
                   "zone_network_blip": "cloud provider reported packet loss in one availability zone at {hm}"}
MIGRATIONS = {
    "none": "no schema migration in this release",
    "expand_only": "expand-only migration applied before the rollout (new nullable column, old code ignores it)",
    "destructive": "contract migration dropped the legacy_status column before the rollout; the previous release "
                   "still maps that column in its ORM model",
    "lock_timeout": "step 3/5 ALTER TABLE timed out waiting for a lock; nothing committed",
    "constraint_violation": "backfill stopped part-way: NOT NULL violation on a handful of rows",
    "backfill_timeout": "backfill hit the 30 min step timeout part-way through",
}
HEALTH_FAILURES = ["CrashLoopBackOff: exit 1, missing required env PARTNER_API_BASE",
                   "OOMKilled at the 1.5Gi memory limit",
                   "readiness probe failed: GET /healthz timed out after 2s (5 of 5 attempts)"]


class CanaryDeployHalt(Schema):
    name = "canary_deploy_halt"
    description = "A progressive deploy stopped itself (canary analysis, migration step or health check): what next?"
    surfaces = {"json": 0.45, "deploy_log": 0.3, "halt_report": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        svc, tier, team = pick(rng, DEPLOY_SERVICES)
        halted_by = pick_w(rng, {"canary_analysis": 6, "migration_step": 2, "health_check": 2})
        metric = pick(rng, list(CANARY_METRICS))
        pass_mark = pick(rng, [70, 75, 80])
        sc = rng.randint(max(0, pass_mark - 35), pass_mark - 1)
        requests = pick(rng, [900, 6000, 40000, 220000])
        external = pick_w(rng, {None: 6, "downstream_degraded": 2, "baseline_pod_restarted": 1, "zone_network_blip": 1})
        migration = pick_w(rng, {"none": 5, "expand_only": 3, "destructive": 1})
        flag_guarded = chance(rng, 0.25)
        halts, overridden = pick(rng, [(0, 0), (1, 0), (1, 1), (3, 1), (4, 3), (6, 5)])
        twist = None
        if reg == "clear":
            if chance(rng, 0.5):
                halted_by, metric, sc, requests, external, halts, overridden = (
                    "canary_analysis", "error_rate", rng.randint(5, 30), 220000, None, 0, 0)
                flag_guarded = False
            else:
                halted_by, external, sc, halts, overridden = ("canary_analysis", "downstream_degraded",
                                                              pass_mark - rng.randint(1, 4), 5, 4)
        elif reg == "borderline":
            sc = pass_mark - rng.randint(1, 9)
            requests = pick(rng, [6000, 40000])
        else:
            twist = pick(rng, ["external_but_severe", "tiny_sample", "one_way_migration", "flag_dark"])
            if twist == "external_but_severe":
                external, sc, halted_by = "downstream_degraded", rng.randint(5, 25), "canary_analysis"
            elif twist == "tiny_sample":
                requests, sc, halted_by = pick(rng, [300, 650, 900]), rng.randint(10, 35), "canary_analysis"
            elif twist == "one_way_migration":
                migration, sc = "destructive", rng.randint(15, pass_mark - 5)
                halted_by = pick(rng, ["canary_analysis", "health_check"])
            else:
                flag_guarded, sc, halted_by = True, pass_mark - rng.randint(3, 20), "canary_analysis"
        if halted_by == "migration_step":
            migration = pick(rng, ["lock_timeout", "constraint_violation", "backfill_timeout"])
            if reg == "clear":
                migration = "lock_timeout"
            if external == "baseline_pod_restarted":        # no canary window before the migrations run
                external = None
        stage = 0 if halted_by == "migration_step" else pick(rng, [1, 5, 10, 25, 50])
        if reg == "clear" and halted_by == "canary_analysis":
            stage = pick(rng, [5, 10])
        lo, hi = CANARY_METRICS[metric][2]
        usual = round(rng.uniform(lo, hi), 3)
        base_elev = external == "downstream_degraded" and twist != "external_but_severe"
        baseline = round(usual * (rng.uniform(1.4, 2.0) if base_elev else rng.uniform(0.9, 1.1)), 3)
        ratio = round(1 + (100 - sc) / CANARY_METRICS[metric][4] * rng.uniform(0.8, 1.2), 2)
        if twist == "tiny_sample":
            stage = 1
        requests = max(120, int(round(requests * stage / 5, -1)))
        start = rand_ts(rng, hours=(8, 22))
        return {"regime": reg, "twist": twist, "service": svc, "tier": tier, "team": team, "halted_by": halted_by,
                "stage_pct": stage, "metric": metric, "usual_baseline": usual, "baseline": baseline,
                "canary_value": round(baseline * ratio, 3), "score": sc, "pass_mark": pass_mark,
                "canary_requests": requests if halted_by == "canary_analysis" else None,
                "external_event": external, "migration": migration,
                "migration_progress_pct": rng.randint(20, 90) if migration in ("constraint_violation", "backfill_timeout") else None,
                "writes_blocked_s": rng.randint(4, 80) if migration == "lock_timeout" else None,
                "health_failure": pick(rng, HEALTH_FAILURES) if halted_by == "health_check" else None,
                "crashing_pods": rng.randint(1, 4) if halted_by == "health_check" else None,
                "flag_guarded": flag_guarded, "config_change": chance(rng, 0.35), "commits": pick(rng, [3, 8, 14, 31]),
                "halts_30d": halts, "halts_overridden_30d": overridden,
                "peak_traffic": chance(rng, 0.4), "minutes_since_halt": pick(rng, [4, 12, 25, 50]),
                "release": f"v{rng.randint(2, 9)}.{rng.randint(10, 60)}.{rng.randint(1, 9)}",
                "deployer": person(rng), "dependency": pick(rng, ["payments", "SMS", "geocoding", "email delivery"]),
                "started": start}

    def _fmt(self, f, v):
        label, unit, _, d, _ = CANARY_METRICS[f["metric"]]
        return f"{v:.{d}f}{unit}" if unit == "%" else f"{v:.{d}f} {unit}"

    def _external_text(self, f):
        if not f["external_event"]:
            return None
        hm = (f["started"] + timedelta(minutes=6)).strftime("%H:%M")
        return EXTERNAL_EVENTS[f["external_event"]].format(dep=f["dependency"], hm=hm)

    def _migration_text(self, f):
        m = MIGRATIONS[f["migration"]]
        if f["migration_progress_pct"]:
            m += f" ({f['migration_progress_pct']}% of rows done)"
        if f["writes_blocked_s"]:
            m += f"; writes to the table queued for {f['writes_blocked_s']}s while it waited"
        return m

    def _stage(self, f):
        return "pre-deploy migration" if f["stage_pct"] == 0 else f"canary {f['stage_pct']}%"

    def render(self, f, rng, surface):
        label = CANARY_METRICS[f["metric"]][0]
        ext, mig = self._external_text(f), self._migration_text(f)
        prev = f["release"].rsplit(".", 1)[0] + "." + str(max(0, int(f["release"].rsplit(".", 1)[1]) - 1))
        is_canary = f["halted_by"] == "canary_analysis"
        if surface == "json":
            return dumps(prune({
                "deployment": {"service": f["service"], "tier": f["tier"], "owner": f["team"], "release": f["release"],
                               "previous_release": prev, "deployed_by": f["deployer"], "commits": f["commits"],
                               "includes_config_change": f["config_change"], "new_code_behind_flag": f["flag_guarded"],
                               "halted_at": self._stage(f), "halted_by": f["halted_by"],
                               "minutes_since_halt": f["minutes_since_halt"]},
                "canary_analysis": ({"metric": label, "baseline": self._fmt(f, f["baseline"]),
                                     "baseline_typical_7d": self._fmt(f, f["usual_baseline"]),
                                     "canary": self._fmt(f, f["canary_value"]), "score": f["score"],
                                     "pass_mark": f["pass_mark"], "canary_requests": f["canary_requests"],
                                     "min_requests_for_verdict": 1000, "window_minutes": 15} if is_canary else None),
                "health": ({"failing_pods": f"{f['crashing_pods']} of {f['crashing_pods'] + rng.randint(1, 3)} new pods",
                            "last_failure": f["health_failure"]} if f["health_failure"] else None),
                "migration": mig,
                "environment": {"traffic": "peak" if f["peak_traffic"] else "off-peak", "concurrent_event": ext},
                "history": {"halts_last_30d": f["halts_30d"], "halts_overridden_last_30d": f["halts_overridden_30d"]},
            }))
        if surface == "deploy_log":
            t = f["started"]
            lines = []
            if f["migration"] != "none" and f["halted_by"] != "migration_step":
                lines.append((t - timedelta(seconds=rng.randint(60, 900)), "INFO ", "migrate", mig))
            lines.append((t, "INFO ", "rollout", f"{f['service']} {f['release']} (prev {prev}): starting "
                                                 f"{'migrations' if f['stage_pct'] == 0 else 'stage ' + self._stage(f)}"))
            if f["halted_by"] == "migration_step":
                t += timedelta(seconds=rng.randint(20, 400))
                lines.append((t, "ERROR", "migrate", mig))
            if is_canary:
                t += timedelta(minutes=15, seconds=rng.randint(0, 40))
                lines.append((t, "INFO ", "analysis", f"15m window: {label} baseline={self._fmt(f, f['baseline'])} "
                                                      f"(7d typical {self._fmt(f, f['usual_baseline'])}) "
                                                      f"canary={self._fmt(f, f['canary_value'])} (canary n={f['canary_requests']:,})"))
                lines.append((t, "ERROR", "analysis", f"score {f['score']} < pass mark {f['pass_mark']} -> FAIL "
                                                      f"(verdicts need >= 1,000 canary requests)"))
            elif f["health_failure"]:
                t += timedelta(minutes=rng.randint(2, 6))
                lines.append((t, "ERROR", "rollout", f"{f['crashing_pods']} new pod(s) unhealthy: {f['health_failure']}"))
            lines.append((t + timedelta(seconds=1), "WARN ", "rollout",
                          f"halted at {self._stage(f)}; tier-{f['tier']} policy: waiting for an operator"))
            out = [f"{fmt_ts(ts, style='log')} {lvl} [{src}] {msg}" for ts, lvl, src, msg in lines]
            extra = [f"release notes: {f['commits']} commits"
                     + (", includes a config change" if f["config_change"] else "")
                     + (", new code path behind a flag (default off)" if f["flag_guarded"] else ""),
                     f"traffic: {'peak' if f['peak_traffic'] else 'off-peak'}",
                     f"this service halted {f['halts_30d']} time(s) in 30 days, {f['halts_overridden_30d']} overridden"]
            if ext:
                extra.append(f"on-call note: {ext}")
            return "\n".join(out) + "\n\n" + "\n".join(f"# {x}" for x in extra)
        # halt_report
        head = [f"### Deploy halted: {f['service']} {f['release']} at {self._stage(f)}",
                f"Owner {f['team']} · tier {f['tier']} · deployed by {f['deployer']} · halted "
                f"{f['minutes_since_halt']} min ago", ""]
        if is_canary:
            other = [m for m in CANARY_METRICS if m != f["metric"]][:2]
            rows = [[label, self._fmt(f, f["baseline"]), self._fmt(f, f["canary_value"]), "FAIL"]]
            for m in other:
                lo, hi = CANARY_METRICS[m][2]
                v = rng.uniform(lo, hi)
                g = dict(f, metric=m)
                rows.append([CANARY_METRICS[m][0], self._fmt(g, v), self._fmt(g, v * rng.uniform(0.97, 1.06)), "pass"])
            head += [table(["metric", "baseline", "canary", "verdict"], rows, rng, style="markdown"), "",
                     f"Score {f['score']}/100 (pass mark {f['pass_mark']}); {f['canary_requests']:,} canary requests "
                     f"over 15 min (the analyser wants at least 1,000 for a verdict). Baseline {label} is usually "
                     f"around {self._fmt(f, f['usual_baseline'])}.", ""]
        elif f["health_failure"]:
            head += [f"Health check: {f['crashing_pods']} new pod(s) failing: {f['health_failure']}", ""]
        notes = [f"Release: {f['commits']} commits; config change: {'yes' if f['config_change'] else 'no'}; "
                 f"new code behind a flag: {'yes' if f['flag_guarded'] else 'no'}",
                 f"Migration: {mig}", f"Traffic: {'peak' if f['peak_traffic'] else 'off-peak'}",
                 f"Other events: {ext}" if ext else None,
                 f"Halts on this service in 30 days: {f['halts_30d']} ({f['halts_overridden_30d']} later overridden)"]
        return "\n".join(head) + "\n" + bullets(notes, rng)

    def questions(self):
        return {
            "rollout_next_step": choice("What should the release captain do with the halted rollout?", {
                "roll_back": "Return every instance to the previous release: the new release is clearly failing and "
                             "the old one can still run against the data.",
                "hold_and_investigate": "Keep the rollout frozen while someone digs in: the score missed only narrowly, "
                                        "or the new code is dark behind a flag, and nothing outside explains it.",
                "resume_rollout": "Override the halt: the score missed the pass mark by five points or less and an "
                                  "outside event that hit the baseline too explains it.",
                "retry_stage": "Run the failed stage again as it is: a locked table, an outside event or too little "
                               "canary traffic, not the release, caused the halt.",
                "roll_forward_fix": "Keep it paused and push a small corrective release, because a migration has "
                                    "already changed data the previous release cannot run against, or a backfill "
                                    "stopped part-way.",
            }),
            "investigation_owner": choice("Who should lead the investigation into this halt?", {
                "service_team": "The team that owns the service and wrote the changes in this release.",
                "deploy_platform": "The deployment platform team that runs the rollout tooling and the canary analyser.",
                "database_team": "The database team that looks after schema migrations and the data stores.",
                "dependency_owner": "The owners of another system the service depends on.",
            }),
            "regression_likelihood": score("How likely is it that the new release itself is what went wrong?", [
                "Very unlikely: something outside the release clearly explains the halt.",
                "Unlikely: an outside cause fits better, though the release is not ruled out.",
                "Even odds: the signal is weak or has another plausible explanation.",
                "Likely: the signal is real and points at the release, with some doubt left.",
                "Near certain: the release is clearly misbehaving.",
            ]),
            "user_exposure": score("How much have users been affected by this rollout so far?", [
                "None: no user traffic reached the new code, or nothing measurable moved.",
                "Slight: a small slice of traffic saw marginally worse behaviour.",
                "Noticeable: a real share of requests saw errors or slowness.",
                "Serious: many users hit failures, or writes were blocked.",
            ]),
            "rollback_restores_state": noul(
                "Rolling back to the previous release would put the service fully back to where it was before.",
                yes="Nothing irreversible has run, and the old release can work with the data as it now stands.",
                no="A migration, backfill or data change has already run that the old release cannot simply undo."),
        }


# =========================================================================== schema 4
# (benchmark, unit, baseline, lower is better, owner, who waits on it, repo, a deliberate trade-off a commit could make)
BENCHES = [
    ("json_decode/large_payload", "ns/op", 48200, True, "Platform Libraries", "hot_path", "ledgerline/api-core",
     "validate UTF-8 up front instead of lazily (safer, a little slower)"),
    ("checkout_flow_e2e/p99", "ms", 410, True, "Checkout", "hot_path", "harborfront/storefront-web",
     "load the fraud-score widget synchronously so the pay button waits for it"),
    ("ingest_throughput/parquet_1gb", "MB/s", 620, False, "Data Ingest", "batch", "northgate/ingest-pipeline",
     "checksum every row group on read"),
    ("index_build/peak_rss", "MiB", 3100, True, "Search Relevance", "batch", "northgate/ranker",
     "keep decoded postings in memory to halve build time"),
    ("cold_start/app_launch", "ms", 850, True, "Mobile Sync", "startup", "quarry/field-app",
     "precompute lookup tables at launch (slower start, faster scrolling)"),
    ("lru_get_hit", "ns/op", 38.5, True, "Edge Platform", "hot_path", "emberline/edge-cache",
     "take a read lock on hits so eviction stats stay exact"),
    ("renewal_batch/10k_subscriptions", "s", 94, True, "Subscriptions", "batch", "tidewater/subscriptions",
     "write an audit row per renewal"),
]
ENV_CHANGES = {"runner_image": ("bench-runner image {a}", "bench-runner image {b}"),
               "instance_type": ("c6i.4xlarge (dedicated)", "c7i.4xlarge (dedicated)"),
               "shared_host": ("dedicated bench pool", "shared general pool (dedicated pool under maintenance)")}
TOOLCHAIN = {"ledgerline/api-core": "Bump Go toolchain to 1.23.4", "emberline/edge-cache": "Bump rustc to 1.82.0",
             "harborfront/storefront-web": "Upgrade Node to 22.11", "northgate/ingest-pipeline": "Move CI to Python 3.13",
             "northgate/ranker": "Move CI to Python 3.13", "quarry/field-app": "Bump Kotlin to 2.1.0",
             "tidewater/subscriptions": "Move to JDK 21"}


class NightlyPerfRegression(Schema):
    name = "nightly_perf_regression"
    description = "A nightly benchmark crossed its regression threshold: is it real, and should it hold the release?"
    surfaces = {"json": 0.45, "alert_email": 0.25, "bench_dashboard": 0.3}

    def sample_factors(self, rng):
        reg = regime(rng)
        bench = pick(rng, BENCHES)
        th = pick(rng, [3, 5, 5, 10])
        cv = round(th * rng.uniform(0.3, 1.2), 1)
        delta = round(th * rng.uniform(1.05, 2.5), 1)
        samples = pick(rng, [3, 5, 10, 30])
        shape = pick_w(rng, {"step": 5, "gradual": 2, "single_night": 3})
        env = pick_w(rng, {None: 6, "runner_image": 1, "instance_type": 1, "shared_host": 1, "toolchain": 1})
        tradeoff = chance(rng, 0.12)
        bisect = pick_w(rng, {"found": 3, "not_run": 3, "inconclusive": 2})
        step_len = pick(rng, [1, 2, 3, 4])
        twist = None
        if reg == "clear":
            if chance(rng, 0.55):
                delta, cv, shape, env, samples = round(th * rng.uniform(2.5, 4), 1), round(th * 0.3, 1), "step", None, pick(rng, [10, 30])
                bisect, tradeoff, step_len = "found", False, pick(rng, [3, 4, 5])
            else:
                delta, cv, shape, env, samples = round(th * rng.uniform(1.05, 1.3), 1), round(th * rng.uniform(1.2, 2), 1), "single_night", "shared_host", 3
                bisect, tradeoff = "inconclusive", False
        elif reg == "borderline":
            delta = near(rng, th * 1.2, th * 0.25, lo=th * 1.02, digits=1)
            cv, samples, shape = round(th * rng.uniform(0.5, 0.9), 1), pick(rng, [3, 5, 5, 10]), pick(rng, ["step", "single_night"])
        else:
            twist = pick(rng, ["env_confound", "slow_creep", "intended_tradeoff", "toolchain"])
            if twist == "env_confound":
                delta, env, shape = round(th * rng.uniform(2, 3.5), 1), pick(rng, ["runner_image", "instance_type"]), "step"
            elif twist == "slow_creep":
                delta, cv, shape, env, samples = round(th * rng.uniform(1.1, 1.5), 1), round(th * 0.3, 1), "gradual", None, 10
            elif twist == "intended_tradeoff":
                delta, tradeoff, shape = round(th * rng.uniform(1.6, 3), 1), True, "step"
            else:
                delta, env, shape = round(th * rng.uniform(1.5, 3), 1), "toolchain", "step"
        # eight nights of medians, oldest first; relative deviation in the "worse" direction
        devs = []
        for i in range(8):
            if shape == "gradual":
                d = delta * (i + 1) / 8 + rng.gauss(0, cv * 0.15)
            elif i >= 8 - (1 if shape == "single_night" else step_len):
                d = delta * rng.uniform(0.9, 1.1)
            else:                                   # ordinary nights stay inside the alert band
                d = max(-th * 0.9, min(th * 0.9, rng.gauss(0, cv * 0.4)))
            devs.append(d)
        devs[7] = delta
        nights = 0
        for d in reversed(devs):
            if d < th:
                break
            nights += 1
        name, unit, base, lower_better, team, path, repo, trade = bench
        vals = [base * (1 + d / 100 if lower_better else 1 - d / 100) for d in devs]
        vals = [round(v, 1) if base < 100 else int(round(v)) for v in vals]
        start = rand_ts(rng, hours=(1, 4)).date()
        f = {"regime": reg, "twist": twist, "benchmark": name, "unit": unit, "baseline": base,
                "lower_is_better": lower_better, "owner": team, "user_path": path, "repo": repo, "threshold_pct": th,
                "noise_cv_pct": cv, "delta_pct": delta, "samples_per_night": samples, "shape": shape,
                "nights_flagged": nights, "series": vals, "env_change": env,
                "env_disclosed": env is not None and (reg == "clear" or chance(rng, 0.7)),
                "tradeoff_commit": trade if tradeoff else None, "bisect": bisect,
                "commits_in_range": pick(rng, [4, 11, 23, 40, 65]),
                "other_benchmarks_flagged": pick(rng, [0, 0, 1, 3, 9]) if env else pick(rng, [0, 0, 0, 1, 2]),
                "release_cut_days": pick(rng, [1, 2, 4, 7, 12]), "last_night": start,
                "image_a": f"{start.year}.{start.month:02d}.1", "image_b": f"{start.year}.{start.month:02d}.2",
                "bisect_sha": _sha(rng), "commenter": person(rng),
                "comment": pick_w(rng, {None: 4, "jumpy": 2, "expected": 1, "worried": 1})}
        # a batch job's spare time in its nightly window, in % of its usual run time
        f["window_headroom_pct"] = pick(rng, [4, 8, 15, 30, 60]) if path == "batch" else None
        return f

    PATH_TEXT = {"hot_path": "on the request path of a customer-facing service",
                 "startup": "app launch time that every user sees",
                 "batch": "a scheduled nightly batch job"}

    def _path(self, f):
        t = self.PATH_TEXT[f["user_path"]]
        if f["window_headroom_pct"] is not None:
            t += f"; it currently finishes with {f['window_headroom_pct']}% of its run time to spare in its window"
        return t

    def _setup(self, f, ra, rb):
        if ra:
            return f"{ra} for the baseline, {rb} last night"
        if f["env_change"] in (None, "toolchain"):
            return "same runner set-up as the baseline runs"
        return None

    def _derived(self, f, rng):
        fmt = (lambda v: f"{v:,.1f}") if f["baseline"] < 100 else (lambda v: f"{v:,.0f}")
        sign = "+" if f["lower_is_better"] else "-"
        change = f"{sign}{f['delta_pct']:.1f}%"
        runner_a = runner_b = None
        if f["env_disclosed"] and f["env_change"] in ENV_CHANGES:
            a, b = ENV_CHANGES[f["env_change"]]
            runner_a, runner_b = a.format(a=f["image_a"], b=f["image_b"]), b.format(a=f["image_a"], b=f["image_b"])
        notable = []
        if f["env_change"] == "toolchain" and (f["env_disclosed"] or chance(rng, 0.5)):
            notable.append(TOOLCHAIN[f["repo"]])
        if f["tradeoff_commit"]:
            notable.append(f["tradeoff_commit"][0].upper() + f["tradeoff_commit"][1:])
        bis = {"found": f"auto-bisect: first bad commit {f['bisect_sha']}"
                        + (f" ({notable[-1]})" if notable and chance(rng, 0.6) else ""),
               "not_run": None, "inconclusive": "auto-bisect: inconclusive (neighbouring commits overlap within noise)"
               }[f["bisect"]]
        comment = None
        if f["comment"]:
            comment = noisy({"jumpy": "this benchmark is always jumpy, I'd rerun before anyone panics",
                             "expected": "we knew this change would cost something here, question is how much is ok",
                             "worried": ("this is the path customers actually hit, I don't love shipping it like this"
                                         if f["user_path"] != "batch" else
                                         "this job already runs close to its nightly window, any slower and it overruns")
                             }[f["comment"]], rng, 0.12)
        return fmt, change, runner_a, runner_b, notable, bis, comment

    def render(self, f, rng, surface):
        fmt, change, ra, rb, notable, bis, comment = self._derived(f, rng)
        better = "lower" if f["lower_is_better"] else "higher"
        if surface == "json":
            return dumps(prune({
                "alert": {"benchmark": f["benchmark"], "repo": f["repo"], "unit": f["unit"], "better": better,
                          "baseline_median": f["baseline"], "last_night": f["series"][-1], "change": change,
                          "alert_threshold_pct": f["threshold_pct"], "consecutive_nights_over_threshold": f["nights_flagged"],
                          "night_of": f["last_night"].isoformat()},
                "measurement": {"samples_per_night": f["samples_per_night"], "historical_cv_pct": f["noise_cv_pct"],
                                "noise_band_pct": round(2 * f["noise_cv_pct"], 1),
                                "runner_setup": self._setup(f, ra, rb)},
                "series_last_8_nights": f["series"],
                "suspects": {"commits_in_range": f["commits_in_range"], "bisect": bis, "notable_commits": notable or None},
                "context": {"owner": f["owner"], "what_waits_on_it": self._path(f),
                            "release_branch_cut_in_days": f["release_cut_days"],
                            "other_benchmarks_flagged_same_night": f["other_benchmarks_flagged"],
                            "comment": comment},
            }))
        if surface == "alert_email":
            lines = ["From: perf-sentinel@ci.example.net",
                     f"To: {f['owner'].lower().replace(' ', '-')}-team@example.net",
                     f"Subject: [perf] {f['benchmark']} regressed {change} ({f['repo']})", "",
                     f"Nightly benchmark {f['benchmark']} crossed its {f['threshold_pct']}% alert threshold "
                     f"on the night of {fmt_date(f['last_night'], rng)} ({better} is better).", "",
                     kv_lines([("baseline median", f"{fmt(f['baseline'])} {f['unit']}"),
                               ("last night", f"{fmt(f['series'][-1])} {f['unit']} ({change})"),
                               ("nights over threshold", f"{f['nights_flagged']} in a row" if f["nights_flagged"] > 1
                                else "last night only"),
                               ("samples per night", f"{f['samples_per_night']} (historical CV {f['noise_cv_pct']}%, "
                                                     f"noise band +/-{round(2 * f['noise_cv_pct'], 1)}%)"),
                               ("runner", self._setup(f, ra, rb)), ("what waits on it", self._path(f))],
                              rng, style="aligned"), "",
                     f"Commits in range: {f['commits_in_range']}." + (f" {bis[0].upper() + bis[1:]}." if bis
                                                                         else " Bisect not run."),
                     ]
            if notable:
                lines.append("Commits worth a look: " + "; ".join(notable))
            lines += [f"Other benchmarks flagged the same night: {f['other_benchmarks_flagged']}.",
                      f"Release branch cut: in {f['release_cut_days']} day(s)."]
            if comment:
                lines += ["", f"> {f['commenter']} replied: {comment}"]
            return "\n".join(lines)
        # bench_dashboard
        rows = []
        for i, v in enumerate(f["series"]):
            day = f["last_night"] - timedelta(days=7 - i)
            dev = (v / f["baseline"] - 1) * 100
            rows.append([fmt_date(day, style="iso"), fmt(v), f"{dev:+.1f}%", "over" if abs(dev) >= f["threshold_pct"]
                         and ((dev > 0) == f["lower_is_better"]) else ""])
        out = [f"## {f['benchmark']} ({f['repo']}): nightly medians, {f['unit']}, {better} is better",
               f"Baseline {fmt(f['baseline'])} {f['unit']}; alert at {f['threshold_pct']}%; {f['samples_per_night']} samples/night; "
               f"run-to-run CV {f['noise_cv_pct']}% (noise band +/-{round(2 * f['noise_cv_pct'], 1)}%)",
               f"Used by: {self._path(f)}", "",
               table(["night", "median", "vs baseline", "flag"], rows, rng), ""]
        extra = [f"Runner: {self._setup(f, ra, rb)}" if self._setup(f, ra, rb) else None,
                 f"{f['commits_in_range']} commits since the last clean night", bis,
                 ("Notable: " + "; ".join(notable)) if notable else None,
                 f"{f['other_benchmarks_flagged']} other benchmark(s) flagged the same night",
                 f"Owner: {f['owner']}; release cut in {f['release_cut_days']} day(s)",
                 f"{f['commenter']}: \"{comment}\"" if comment else None]
        return "\n".join(out) + "\n" + bullets([x for x in extra if x], rng)

    def questions(self):
        return {
            "release_gate_call": choice("What should the release captain do about this benchmark alert?", {
                "block_release": "Hold the release cut until the slowdown is fixed or explained: it looks real and "
                                 "users or a deadline-bound job would feel it.",
                "ship_and_track": "Let the release go ahead and file a ticket: the slowdown looks real but is too "
                                  "small for anyone to feel.",
                "rerun_first": "Re-run the benchmark under controlled conditions first: the shift is within noise, "
                               "seen on one night only, or measured on a changed runner set-up.",
                "accept_as_intended": "Accept the new numbers as the known cost of a deliberate change named among "
                                      "the commits, and move the baseline.",
            }),
            "signal_strength": score("How clearly does the flagged result stand out from the benchmark's normal variation?", [
                "Not at all: the shift is inside the noise band (twice the run-to-run CV).",
                "Faintly: past the noise band, but on one night only.",
                "Clearly: past the noise band on two or more nights in a row.",
                "Unmistakably: at least twice the noise band, on three or more nights in a row.",
            ]),
            "user_felt_impact": score("If the slowdown is real, how much would users or downstream jobs feel it?", [
                "Not at all: a batch job with at least twice the slowdown to spare in its window.",
                "Slightly: a user-facing path slowed by under 5%, or a batch job that still fits its window.",
                "Noticeably: a user-facing path or app launch slowed by 5 to 15%.",
                "Badly: a user-facing path slowed by more than 15%, or a batch job pushed past the end of its window.",
            ]),
            "slowdown_is_real": noul(
                "The flagged numbers reflect a genuine performance change in the code, not noise or a changed setup.",
                yes="The shift clears the noise band, persists, and no change in how it was measured explains it.",
                no="Noise, a one-night blip, or a change of hardware, image or neighbours explains the numbers."),
            "measurement_conditions_differ": noul(
                "The flagged runs were measured under different conditions from the baseline runs.",
                yes="The alert shows a different runner image, instance type or host pool for the flagged runs than "
                    "for the baseline.",
                no="The alert shows the same runner set-up for both, or names no change in how the runs were measured."),
        }


# =========================================================================== schema 5
# language -> (ecosystem, lockfile, invented packages)
DEP_ECOSYSTEMS = {
    "python": ("PyPI", "poetry.lock", ["pydantix", "tzparse", "httpcore-lite", "orjsonic"]),
    "go": ("Go modules", "go.sum", ["github.com/quillsoft/retry", "github.com/lattice-io/yamlkit"]),
    "typescript": ("npm", "pnpm-lock.yaml", ["tzfmt", "@lattice/forms", "fetchwire", "zodiac-schema"]),
    "java": ("Maven", "gradle.lockfile", ["com.lattice:json-bind", "io.brightwire:http-client"]),
    "kotlin": ("Maven", "gradle.lockfile", ["org.quillsoft:retry4", "io.brightwire:http-client"]),
    "rust": ("crates.io", "Cargo.lock", ["bytepool", "serde-lattice", "hyperion-http"]),
}
DEP_REASONS = {"routine": "routine weekly update", "eol": "current version reaches end of support",
               "blocks_other": "another pending upgrade needs this version", "fixes_bug": "fixes a bug we hit in production"}


# package -> (the build error an API change would cause, the first failing assertion a behaviour change would cause)
PKG_ERRORS = {
    "pydantix": ("ImportError: cannot import name 'BaseSettings' from 'pydantix'",
                 "ValidationError: 1 validation error for OrderEvent: extra fields not permitted (source)"),
    "tzparse": ("AttributeError: module 'tzparse' has no attribute 'parse_zone'",
                "assert '2026-03-29T01:00:00+00:00' == '2026-03-29T02:00:00+01:00'"),
    "httpcore-lite": ("TypeError: Client.__init__() got an unexpected keyword argument 'retries'",
                      "AssertionError: expected 3 attempts on HTTP 503, saw 1"),
    "orjsonic": ("ImportError: cannot import name 'OPT_NAIVE_UTC' from 'orjsonic'",
                 "assert b'{\"amount\":12.5}' == b'{\"amount\":12.50}'"),
    "github.com/quillsoft/retry": ("internal/client/retry.go:41:12: undefined: retry.WithJitter",
                                   "client_test.go:77: attempts = 1, want 3"),
    "github.com/lattice-io/yamlkit": ("internal/config/load.go:23:18: too many arguments in call to yamlkit.Unmarshal",
                                      "config_test.go:52: key \"on\" decoded as bool true, want string \"on\""),
    "tzfmt": ("src/lib/dates.ts(18,22): error TS2339: Property 'parseZone' does not exist on type 'typeof import(\"tzfmt\")'",
              "expect(received).toBe(expected) Expected: \"29 Mar, 02:00\" Received: \"29 Mar, 01:00\""),
    "@lattice/forms": ("error TS2305: Module '\"@lattice/forms\"' has no exported member 'useFieldArray'.",
                       "expected onSubmit to have been called 1 time, but it was called 0 times"),
    "fetchwire": ("src/api/client.ts(44,17): error TS2554: Expected 1 arguments, but got 2.",
                  "expected header content-type 'application/json', received 'application/json; charset=utf-8'"),
    "zodiac-schema": ("error TS2339: Property 'nonempty' does not exist on type 'ZodiacString'.",
                      "expected parse to fail for an empty string, but it succeeded"),
    "com.lattice:json-bind": ("error: cannot find symbol: method bindStrict(String) in class JsonBinder",
                              "org.opentest4j.AssertionFailedError: expected: <12.50> but was: <12.5>"),
    "io.brightwire:http-client": ("error: incompatible types: Duration cannot be converted to long (RenewalClient.java:57)",
                                  "AssertionFailedError: expected: <3> retry attempts but was: <1>"),
    "org.quillsoft:retry4": ("e: SyncWorker.kt:88:21 Unresolved reference: withJitter",
                             "java.lang.AssertionError: expected attempts:<3> but was:<1>"),
    "bytepool": ("error[E0061]: this function takes 2 arguments but 1 argument was supplied --> src/io/pool.rs:88:17",
                 "assertion `left == right` failed: left: 4096, right: 8192 (pool chunk size)"),
    "serde-lattice": ("error[E0432]: unresolved import `serde_lattice::de::Legacy`",
                      "called `Result::unwrap()` on an `Err` value: unknown field `legacy_id`"),
    "hyperion-http": ("error[E0599]: no method named `http1_only` found for struct `ClientBuilder`",
                      "assertion failed: resp.version() == Version::HTTP_11"),
}


def _dep_error(kind, lang, pkg, frm, to, eco):
    if kind == "compile_error":
        return PKG_ERRORS[pkg][0]
    if kind == "test_failures":
        return PKG_ERRORS[pkg][1]
    if kind == "resolution_conflict":
        return {"typescript": f"ERESOLVE unable to resolve dependency tree: peer {pkg}@\"^{frm.split('.')[0]}\" "
                              f"required by @lattice/forms-adapter@2.1.0",
                "python": f"ResolutionImpossible: httpcore-lite 1.4.0 depends on {pkg}<{to}",
                "go": f"go: {pkg}@{to} requires go >= 1.24 (running go 1.23.4; GOTOOLCHAIN=local)",
                "rust": f"failed to select a version for `{pkg}`: required by `edge-cache v0.1.0` and by `hyperion-http v0.8.2`"
                }.get(lang, f"Could not resolve {pkg}:{to}: conflict with {frm} required by io.brightwire:http-client")
    if kind == "lock_checksum":
        return {"python": f"THESE PACKAGES DO NOT MATCH THE HASHES FROM THE LOCK FILE: {pkg}=={to} (expected sha256:9f2c41e0..., got sha256:41ab07d3...)",
                "go": f"verifying {pkg}@{to}: checksum mismatch (go.sum h1:7LpQ2..., downloaded h1:Xq3vA...)",
                "typescript": f"ERR_PNPM_TARBALL_INTEGRITY Got unexpected checksum for {pkg}@{to}",
                "rust": f"checksum for `{pkg} v{to}` changed between lock files"
                }.get(lang, f"Dependency verification failed for {pkg}:{to}: checksum does not match the lockfile")
    return f"GET https://registry-mirror.ci.example.net/{eco.lower().replace(' ', '-')}/{pkg}/{to}: 503 Service Unavailable (attempt 3/3)"


class DependencyUpdateBreakage(Schema):
    name = "dependency_update_breakage"
    description = "A dependency or lockfile update turned the build red: adapt, pin, wait or retry?"
    surfaces = {"json": 0.45, "bot_pr": 0.3, "build_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        repo, lang, _ = _repo_and_test(rng)
        eco, lockfile, pkgs = DEP_ECOSYSTEMS[lang]
        pkg = pick(rng, pkgs)
        bump = pick_w(rng, {"patch": 3, "minor": 4, "major": 2})
        kind = pick_w(rng, {"compile_error": 3, "test_failures": 4, "resolution_conflict": 2, "registry_fetch_error": 2,
                            "lock_checksum": 2})
        changelog = pick_w(rng, {"breaking_listed": 2, "no_mention": 4, None: 2})
        upstream = pick_w(rng, {None: 5, "open_acknowledged": 2, "open_no_reply": 2, "closed_wontfix": 1})
        usage = pick_w(rng, {"public_api": 6, "internal_module": 2, "deprecated_api": 2})
        call_sites = pick(rng, [2, 5, 9, 18, 44])
        driver = pick_w(rng, {"routine": 5, "eol": 1, "blocks_other": 2, "fixes_bug": 2})
        batch = pick(rng, [1, 1, 3, 5, 8])
        twist = None
        if reg == "clear":
            if chance(rng, 0.55):
                bump, changelog, kind, usage, call_sites, upstream, driver = (
                    "major", "breaking_listed", "compile_error", "public_api", pick(rng, [18, 44]), None, "routine")
            else:
                kind, bump, changelog, upstream, usage = "registry_fetch_error", "patch", "no_mention", None, "public_api"
        elif reg == "borderline":
            bump, changelog, usage = "minor", "no_mention", "public_api"
            kind = pick(rng, ["test_failures", "test_failures", "lock_checksum", "registry_fetch_error"])
            upstream, call_sites = pick(rng, [None, "open_no_reply"]), pick(rng, [3, 5, 9])
        else:
            twist = pick(rng, ["patch_breaks", "our_misuse", "eol_pressure", "resolution"])
            if twist == "patch_breaks":
                bump, kind, changelog, upstream = "patch", pick(rng, ["compile_error", "test_failures"]), "no_mention", "open_acknowledged"
            elif twist == "our_misuse":
                usage, upstream, kind = "internal_module", "closed_wontfix", pick(rng, ["compile_error", "test_failures"])
            elif twist == "eol_pressure":
                bump, changelog, driver, call_sites, kind = "major", "breaking_listed", pick(rng, ["eol", "blocks_other"]), 44, "compile_error"
            else:
                kind, driver = "resolution_conflict", pick(rng, ["fixes_bug", "blocks_other", "routine"])
        major, minor, patch = rng.randint(1, 5), rng.randint(0, 14), rng.randint(0, 9)
        frm = f"{major}.{minor}.{patch}"
        to = {"patch": f"{major}.{minor}.{patch + rng.randint(1, 3)}", "minor": f"{major}.{minor + 1}.0",
              "major": f"{major + 1}.0.0"}[bump]
        if lang == "go":
            frm, to = "v" + frm, "v" + to
        return {"regime": reg, "twist": twist, "repo": repo, "language": lang, "ecosystem": eco, "lockfile": lockfile,
                "package": pkg, "from_version": frm, "to_version": to, "bump": bump,
                "direct": kind != "resolution_conflict" or chance(rng, 0.5), "failure_kind": kind,
                "failing_tests": rng.randint(1, 3) if reg == "borderline" else pick(rng, [1, 2, 4, 11, 37]),
                "changelog": changelog, "upstream_issue": upstream, "our_usage": usage, "call_sites": call_sites,
                "deferral_driver": driver, "eol_days": pick(rng, [14, 30, 60]) if driver == "eol" else None,
                "batch_size": batch, "retried": kind == "registry_fetch_error" and chance(rng, 0.3) or chance(rng, 0.15),
                "local_repro": pick_w(rng, {"reproduces": 2, "passes_locally": 1, None: 3}),
                "opened_by": pick_w(rng, {"depbot": 6, "lockfile_refresh": 2, "manual": 2}),
                "pr": rng.randint(1200, 9800), "engineer": person(rng), "started": rand_ts(rng, hours=(6, 20))}

    def _texts(self, f, rng):
        err = _dep_error(f["failure_kind"], f["language"], f["package"], f["from_version"], f["to_version"], f["ecosystem"])
        step = {"compile_error": "build", "test_failures": "test", "resolution_conflict": "install",
                "registry_fetch_error": "install", "lock_checksum": "install"}[f["failure_kind"]]
        cl = {"breaking_listed": f"release notes for {f['to_version']} list breaking changes",
              "no_mention": f"release notes for {f['to_version']} mention no breaking changes", None: None}[f["changelog"]]
        up = {None: None, "open_acknowledged": "an upstream issue reports the same error; a maintainer says a fix is coming",
              "open_no_reply": "an upstream issue reports something similar; no maintainer reply yet",
              "closed_wontfix": "upstream closed a similar report: the module involved is internal and not covered by "
                                "semver"}[f["upstream_issue"]]
        use = {"public_api": None, "internal_module": f"code imports {f['package']} from an internal (underscore) module",
               "deprecated_api": "code still calls APIs deprecated two minor versions ago"}[f["our_usage"]]
        why = DEP_REASONS[f["deferral_driver"]] + (f" in {f['eol_days']} days" if f["eol_days"] else "")
        others = None
        if f["batch_size"] > 1:
            others = f"grouped update of {f['batch_size']} packages" + (
                "" if f["failure_kind"] in ("registry_fetch_error", "lock_checksum")
                else f"; the other {f['batch_size'] - 1} install and build cleanly")
        if f["failure_kind"] == "lock_checksum":
            up = (up + "; " if up else "") + (f"the registry shows {f['to_version']} was re-published about 40 minutes "
                                              f"after it first appeared")
        return err, step, cl, up, use, why, others

    def render(self, f, rng, surface):
        err, step, cl, up, use, why, others = self._texts(f, rng)
        who = {"depbot": "depbot", "lockfile_refresh": "lockfile-refresh job", "manual": f["engineer"]}[f["opened_by"]]
        repro = {"reproduces": "reproduces on a developer machine", "passes_locally": "passes on a developer machine",
                 None: None}[f["local_repro"]]
        if surface == "json":
            return dumps(prune({
                "update": {"repo": f["repo"], "pull_request": f["pr"], "opened_by": who, "ecosystem": f["ecosystem"],
                           "package": f["package"], "from": f["from_version"], "to": f["to_version"], "bump": f["bump"],
                           "direct_dependency": f["direct"], "lockfile": f["lockfile"], "batch": others, "reason": why},
                "ci": {"failed_step": step, "error": err,
                       "failing_tests": f["failing_tests"] if f["failure_kind"] == "test_failures" else None,
                       "retried": f["retried"], "local": repro},
                "upstream": {"release_notes": cl, "issue_tracker": up},
                "codebase": {"call_sites": f["call_sites"], "note": use},
            }))
        if surface == "bot_pr":
            title = f"Bump {f['package']} from {f['from_version']} to {f['to_version']}"
            lines = [f"#{f['pr']} {title}" + (f" (+{f['batch_size'] - 1} more)" if f["batch_size"] > 1 else ""),
                     f"{f['repo']} · opened by {who}", "",
                     f"Updates {f['package']} ({'direct' if f['direct'] else 'transitive'}, {f['ecosystem']}) "
                     f"{f['from_version']} -> {f['to_version']} ({f['bump']}). Reason: {why}.",
                     (cl[0].upper() + cl[1:] + ".") if cl else "No release notes published for this version.", "",
                     f"CI: {step} failed", f"    {err}"]
            if f["failure_kind"] == "test_failures":
                lines.append(f"    ({f['failing_tests']} test(s) failing)")
            if f["retried"]:
                lines.append("    (retried once, same result)")
            if others:
                lines.append(f"Note: {others}.")
            comments = []
            if up:
                comments.append(f"{f['engineer'].split()[0]}: checked upstream, {up}")
            if use:
                comments.append(f"{_name_unlike(rng, f['engineer']).split()[0]}: fwiw our {use}")
            if repro:
                comments.append(f"{f['engineer'].split()[0]}: " + ("reproduces locally for me" if f["local_repro"] ==
                                                                   "reproduces" else "passes locally for me, odd"))
            comments.append(f"{_name_unlike(rng, f['engineer']).split()[0]}: {f['package']} is used at "
                            f"{f['call_sites']} call site(s)")
            return "\n".join(lines + ["", "Comments:"] + [f"  {noisy(c, rng, 0.1)}" for c in comments])
        # build_log
        t0 = f["started"]
        cmd = {"install": {"python": "poetry install --sync", "go": "go mod download",
                           "typescript": "pnpm install --frozen-lockfile", "java": "./gradlew dependencies",
                           "kotlin": "./gradlew dependencies", "rust": "cargo fetch --locked"},
               "build": {"python": "python -m compileall -q src && pytest --collect-only -q", "go": "go build ./...",
                         "typescript": "pnpm tsc --noEmit", "java": "./gradlew compileJava",
                         "kotlin": "./gradlew compileDebugKotlin", "rust": "cargo build --locked"},
               "test": {**RUN_CMD, "typescript": "npx jest --ci"}}[step][f["language"]]
        log = [f"{_log_ts(t0, 11)} ##[group]{step}: {cmd}",
               f"{_log_ts(t0 + timedelta(seconds=4), 204)} resolving {f['package']}@{f['to_version']} (was {f['from_version']})"]
        t1 = t0 + timedelta(seconds=rng.randint(20, 300))
        log.append(f"{_log_ts(t1, 877)} {err}")
        if f["failure_kind"] == "test_failures":
            log.append(f"{_log_ts(t1, 902)} {f['failing_tests']} failed, {rng.randint(300, 2400)} passed")
        log.append(f"{_log_ts(t1 + timedelta(seconds=1), 3)} ##[error]Process completed with exit code 1.")
        notes = kv_lines([("lockfile", f["lockfile"]), ("bump", f"{f['bump']} ({'direct' if f['direct'] else 'transitive'})"),
                          ("why", why), ("batch", others), ("release notes", cl), ("upstream", up), ("usage", use),
                          ("call sites", f["call_sites"]), ("retried", "yes" if f["retried"] else None),
                          ("local", repro)], rng, style=pick(rng, ["colon", "dash"]))
        return "\n".join(log) + f"\n\n{f['repo']} #{f['pr']} ({who})\n" + notes

    def questions(self):
        return {
            "upgrade_path": choice("How should the team handle this broken dependency update?", {
                "pin_current": "Pin the dependency at its current version and close the update for now.",
                "adapt_code": "Take the new version and change this repository's code to work with it.",
                "wait_for_upstream": "Hold the update until the upstream project ships a fix, then try again.",
                "regenerate_and_retry": "Regenerate the lockfile or retry the install; the new code is probably not the problem.",
                "split_batch": "Land the rest of the grouped update and carve this package out into its own change.",
            }),
            "adoption_effort": score("How much work would it take to adopt the new version?", [
                "Trivial: nothing, or a one-line change.",
                "Small: a handful of call sites or tests to touch.",
                "Substantial: coordinated changes across many call sites or modules.",
                "Major: a redesign, or waiting on another team or on upstream.",
            ]),
            "deferral_cost": score("How costly is it to stay on the current version until next month's dependency sweep?", [
                "Negligible: nothing is waiting on the new version.",
                "Minor: some drift that makes the next upgrade a little bigger.",
                "Significant: it holds up other upgrades or leaves a known bug in place.",
                "Severe: the current version loses support or breaks before then.",
            ]),
            "upstream_at_fault": noul(
                "The breakage comes from a change in the dependency's behaviour, not from how this repository uses it.",
                yes="The new version changed or broke documented, public behaviour that this code relied on correctly.",
                no="This code relied on internal, deprecated or undocumented behaviour, or the failure is not about "
                   "the new version at all."),
            "lockfile_only_issue": noul(
                "Regenerating the lockfile or retrying the job would turn the build green without any source change.",
                yes="The job failed while fetching or verifying packages (a registry outage, or a checksum that no "
                    "longer matches a re-published release), not because of the new version's code.",
                no="The new version fails to compile, fails tests or cannot be resolved alongside other packages, so "
                   "source or version-constraint changes are needed."),
        }


# =========================================================================== schema 6
CRIT_TEXT = {"revenue_path": "money path (pricing, payments or ledger)", "core_feature": "core product behaviour",
             "secondary_feature": "secondary feature (ranking tie-breaks)"}
MODE_THEORY = {"timing": "a sleep that is too short on busy runners", "order_dependent": "state leaking from another test",
               "external_service": "the partner sandbox being flaky", "resource_contention": "port clashes between shards",
               "value_varies": "nondeterministic ordering in the test data"}


def _mode_error(mode, lang, t, port):
    if mode == "timing":
        return _error_line("timeout", lang, t)
    if mode == "order_dependent":
        return t["assertion"] + " (seen only when an earlier test ran in the same worker)"
    if mode == "external_service":
        return "sandbox.partner-api.example.com: 504 Gateway Timeout"
    if mode == "resource_contention":
        return f"bind: address already in use (127.0.0.1:{port})"
    return t["assertion"]


class FlakyTestQuarantine(Schema):
    name = "flaky_test_quarantine"
    description = "Someone asks to pull an intermittently failing test out of the merge gate: grant it or not?"
    surfaces = {"json": 0.45, "quarantine_ticket": 0.3, "ci_health_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        pool = [(lang, t) for lang, ts in TESTS.items() for t in ts]
        lang, t = pick(rng, pool)
        runs = pick(rng, [100, 200, 400])
        rate = pick(rng, [0.5, 1, 2, 4, 8])                         # % of runs that fail
        retry_pass = pick(rng, [55, 82, 90, 96, 100])               # % of failures that pass on an immediate retry
        mode = pick_w(rng, {"timing": 3, "order_dependent": 2, "external_service": 2, "resource_contention": 1,
                            "value_varies": 2})
        onset_change = maybe(rng, chance(rng, 0.3), 0.3)
        dup = pick_w(rng, {"covered_elsewhere": 2, "partial": 3, "none": 2, None: 1})
        role = pick_w(rng, {"owner_team": 3, "other_team": 3, "build_cop": 2})
        incidents = pick(rng, [0, 0, 0, 1, 2])
        ticket = pick_w(rng, {None: 3, "open_unassigned": 3, "assigned": 2})
        q_count, q_oldest = pick(rng, [(2, 9), (6, 25), (14, 60), (31, 190)])
        twist = None
        if reg == "clear":
            if chance(rng, 0.55):
                lang, t = pick(rng, [p for p in pool if p[1]["crit"] != "revenue_path"])
                mode, retry_pass, rate, onset_change = pick(rng, ["timing", "external_service"]), 100, pick(rng, [4, 8]), False
                dup, role, incidents, ticket = "covered_elsewhere", "owner_team", 0, "assigned"
            else:
                lang, t = pick(rng, [p for p in pool if p[1]["crit"] == "revenue_path"])
                mode, retry_pass, onset_change, dup, incidents = "value_varies", 55, True, "none", 2
                role = "other_team"
        elif reg == "borderline":
            rate, retry_pass = pick(rng, [1, 2]), pick(rng, [75, 88])
            mode, dup = pick(rng, ["order_dependent", "resource_contention", "timing"]), "partial"
        else:
            twist = pick(rng, ["critical_but_classic", "value_varies_low_rate", "other_team_pressure", "quarantine_debt"])
            if twist == "critical_but_classic":
                lang, t = pick(rng, [p for p in pool if p[1]["crit"] == "revenue_path"])
                mode, retry_pass, dup, onset_change = "timing", 100, "none", False
            elif twist == "value_varies_low_rate":
                mode, rate, retry_pass, incidents = "value_varies", 0.5, 100, 0
            elif twist == "other_team_pressure":
                role, ticket, rate, runs = "other_team", None, pick(rng, [4, 8]), 400
            else:
                (q_count, q_oldest), ticket = pick(rng, [(38, 240), (52, 410)]), "open_unassigned"
        if t["kind"] == "unit":
            mode = {"external_service": "timing", "resource_contention": "order_dependent"}.get(mode, mode)
        fails = max(1, round(runs * rate / 100))
        retry_passes = round(fails * retry_pass / 100)          # keep the share achievable with this many failures
        theory = mode
        if twist == "value_varies_low_rate" or (reg != "clear" and chance(rng, 0.25)):
            theory = pick(rng, [m for m in MODE_THEORY if m != mode])   # the requester's guess can be wrong
        return {"regime": reg, "twist": twist, "language": lang, "test": t["id"], "repo": t["repo"],
                "owner": t["team"], "criticality": t["crit"], "window_runs": runs, "failures": fails,
                "retry_passes": retry_passes, "pass_on_retry_pct": round(100 * retry_passes / fails),
                "failure_mode": mode, "requester_theory": theory,
                "flaky_since_days": pick(rng, [3, 9, 21, 60]), "onset_near_change": onset_change,
                "duplicate_coverage": dup, "requester_role": role, "requester": person(rng),
                "requester_team": t["team"] if role == "owner_team" else pick(rng, [x for x in TEAMS if x != t["team"]]),
                "prod_incidents_90d": incidents, "fix_ticket": ticket,
                # a red gate blocks at most one PR per failure; the window is roughly the last month of runs
                "blocked_prs_7d": min(max(1, fails // 3), pick(rng, [2, 6, 14, 31]) if twist != "other_team_pressure"
                                      else pick(rng, [12, 18, 24])),
                "owner_responded": role == "owner_team" or (twist != "other_team_pressure" and chance(rng, 0.5)),
                "quarantined_now": q_count, "oldest_quarantine_days": q_oldest,
                "policy_max_days": pick(rng, [14, 30]), "request_id": f"QRN-{rng.randint(100, 999)}",
                "port": pick(rng, [5432, 6379, 8081]), "opened": rand_ts(rng, hours=(8, 19)),
                # who may pull a test out of the gate, under this repository's quarantine policy
                "signoff_policy": pick(rng, ["money_paths", "blocked_prs", "owner_silent"])}

    SIGNOFF = {"money_paths": "the owning team must agree before a money-path test is quarantined; any other test "
                              "may be pulled by the build cop",
               "blocked_prs": "the build cop may pull a test without the owners once it has blocked 10 or more PRs "
                              "in 7 days; otherwise the owning team must agree",
               "owner_silent": "the owning team must agree, unless it has not responded on the request, in which case "
                               "the build cop decides"}
    CAP = 30

    def _policy(self, f):
        return (f"Quarantine policy: {self.SIGNOFF[f['signoff_policy']]}; at most {self.CAP} tests in quarantine, "
                f"each for {f['policy_max_days']} days at most.")

    def _bits(self, f, rng):
        t = next(x for x in TESTS[f["language"]] if x["id"] == f["test"])
        err = _mode_error(f["failure_mode"], f["language"], t, f["port"])
        rate = f"{f['failures']}/{f['window_runs']} ({100 * f['failures'] / f['window_runs']:.1f}%)"
        onset = {True: "a change to the covered code merged the day before the first flaky failure",
                 False: "no change to the covered code around when the flakiness started", None: None}[f["onset_near_change"]]
        dup = {"covered_elsewhere": "the same behaviour is also checked by two other suites",
               "partial": "other tests cover the happy path only", "none": "no other test covers this behaviour",
               None: None}[f["duplicate_coverage"]]
        ticket = {None: None, "open_unassigned": "fix ticket open, unassigned",
                  "assigned": "fix ticket assigned, owner expects a fix next sprint"}[f["fix_ticket"]]
        th = MODE_THEORY[f["requester_theory"]]
        if f["requester_role"] == "owner_team":
            opts = [f"Our test, known issue ({th}). We want it out of the gate while we fix it.",
                    f"Ours. We think it's {th}; asking to quarantine while we work on it."]
        else:
            opts = [f"This keeps failing PRs, looks like {th}.", f"Pretty sure it's {th}; please quarantine until fixed."]
            if f["blocked_prs_7d"] >= 6:
                opts.append(f"Blocking merges all week. Probably {th}.")
        reason = noisy(pick(rng, opts), rng, 0.15)
        return err, rate, onset, dup, ticket, reason

    def render(self, f, rng, surface):
        err, rate, onset, dup, ticket, reason = self._bits(f, rng)
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "test": f["test"], "repo": f["repo"], "requested_by": f["requester"],
                            "requester_team": f["requester_team"], "reason": reason},
                "flake_stats": {"window_runs": f["window_runs"], "failures": f["failures"],
                                "pass_on_immediate_retry_pct": f["pass_on_retry_pct"],
                                "flaky_since_days": f["flaky_since_days"], "sample_failure": err},
                "test": {"owner": f["owner"], "covers": CRIT_TEXT[f["criticality"]], "overlapping_coverage": dup},
                "context": {"covered_code_changes_near_onset": onset, "prod_incidents_same_area_90d": f["prod_incidents_90d"],
                            "prs_blocked_last_7d": f["blocked_prs_7d"], "fix_ticket": ticket,
                            "owner_team_has_responded": f["owner_responded"]},
                "quarantine_list": {"currently_quarantined": f["quarantined_now"],
                                    "oldest_days": f["oldest_quarantine_days"], "policy_max_days": f["policy_max_days"]},
                "policy": self._policy(f),
            }))
        if surface == "quarantine_ticket":
            rows = [("Request", f["request_id"]), ("Opened", fmt_ts(f["opened"], rng)), ("Test", f["test"]),
                    ("Repository", f["repo"]), ("Owning team", f["owner"]),
                    ("Requested by", f"{f['requester']} ({f['requester_team']})"),
                    ("Covers", CRIT_TEXT[f["criticality"]]), ("Failure rate", rate),
                    ("Pass on retry", f"{f['retry_passes']} of {f['failures']} failures"),
                    ("Flaky since", f"{f['flaky_since_days']} days ago"), ("Other coverage", show(dup, rng)),
                    ("Fix ticket", show(ticket, rng)), ("PRs blocked (7d)", f["blocked_prs_7d"]),
                    ("Quarantine list", f"{f['quarantined_now']} tests, oldest {f['oldest_quarantine_days']} days "
                                        f"(policy max {f['policy_max_days']})")]
            body = ["FLAKY TEST QUARANTINE REQUEST", kv_lines(rows, rng), "", "Reason:", reason, "",
                    "Sample failure:", f"    {err}", "", self._policy(f)]
            notes = [onset, f"{f['prod_incidents_90d']} production incident(s) in this area in the last 90 days"
                     if f["prod_incidents_90d"] else None,
                     None if f["owner_responded"] else f"{f['owner']} has not responded on the ticket yet"]
            notes = [n for n in notes if n]
            if notes:
                body += ["", "Triage notes:", bullets(notes, rng)]
            return "\n".join(body)
        # ci_health_chat
        who = f["requester"].split()[0].lower()
        cop_name = _name_unlike(rng, f["requester"])
        rep_name = _name_unlike(rng, f["requester"], cop_name)
        cop, owner_rep = cop_name.split()[0].lower(), rep_name.split()[0].lower()
        msgs = [(who, f"can we quarantine `{f['test']}`? {reason}"),
                (cop, f"flake tracker: {rate} over the last {f['window_runs']} runs, {f['retry_passes']} of "
                      f"{f['failures']} failures passed on retry, flaky for ~{f['flaky_since_days']} days"),
                (cop, f"latest failure: {err}")]
        if onset:
            msgs.append((cop, onset))
        if dup:
            msgs.append((owner_rep if f["owner_responded"] else cop, dup))
        if f["prod_incidents_90d"]:
            msgs.append((cop, f"note {f['prod_incidents_90d']} prod incident(s) touching this area in the last 90 days"))
        if f["owner_responded"]:
            msgs.append((owner_rep, pick(rng, ["we can look at it next sprint", "please don't pull it, it's the only "
                                               "thing guarding that path", "fine by us if it comes back in two weeks",
                                               "we think it's the runners, not us"])))
        else:
            msgs.append((cop, f"no answer from {f['owner']} yet"))
        msgs.append((cop, f"quarantine list is at {f['quarantined_now']} tests, oldest {f['oldest_quarantine_days']}d, "
                          f"policy says {f['policy_max_days']}d max"))
        if f["blocked_prs_7d"] >= 14:
            msgs.append((_name_unlike(rng, f["requester"], cop_name, rep_name).split()[0].lower(),
                         f"it has blocked {f['blocked_prs_7d']} PRs this week"))
        t = f["opened"]
        out = []
        for w, m in msgs:
            t += timedelta(minutes=rng.randint(1, 20))
            out.append(f"{t.strftime('%H:%M')} <{w}> {noisy(m, rng, 0.06)}")
        return f"#ci-health ({f['repo']})  [channel topic: {self._policy(f)}]\n" + "\n".join(out)

    def questions(self):
        return {
            "quarantine_call": choice("How should the build engineer answer this quarantine request?", {
                "quarantine_with_deadline": "Take the test out of the merge gate now, with a tracked fix ticket and "
                                            "an expiry date: it is environmental flakiness that retries do not absorb.",
                "retry_in_gate": "Keep it gating, but let CI retry it once automatically before failing the build: "
                                 "almost every failure passes on an immediate retry.",
                "owner_fixes_first": "Keep it gating as it is and have the owning team fix it within days, because "
                                     "the quarantine list is already at its cap.",
                "investigate_as_bug": "Treat the failures as a possible product defect and investigate before "
                                      "touching the gate.",
                "delete_test": "Delete the test, because the same behaviour is already checked by other suites.",
            }),
            "coverage_at_stake": score("How much protection would the merge gate lose if this test stopped gating?", [
                "Almost none: other tests check the same behaviour.",
                "Some: the overlap is partial and a few paths go unguarded.",
                "A lot: an important behaviour would go unchecked before merge.",
                "Critical: a money, data or core user path would lose its only pre-merge check.",
            ]),
            "nondeterminism_evidence": score(
                "How convincingly do the failures look like test nondeterminism rather than a product defect?", [
                    "Not convincing: the pattern looks like a genuine bug.",
                    "Weak: mixed signals, and some of them point at the code.",
                    "Fair: mostly consistent with a flaky test, with a doubt or two.",
                    "Strong: textbook flakiness that passes on retry and lines up with no code change.",
                ]),
            "may_mask_defect": noul(
                "The intermittent failures could be hiding a real defect in the code under test.",
                yes="Many failures fail again on an immediate retry, or the flakiness began right after a change to "
                    "the covered code: a race or bug users could hit too.",
                no="Nearly all failures pass on retry and the flakiness began with no change to the covered code, so "
                   "the test or its environment is to blame."),
            "owner_signoff_required": noul(
                "The test's owning team has to agree before it leaves the merge gate.",
                yes="Under the quarantine policy shown, this request still needs the owning team's agreement, and the "
                    "requester is not on that team.",
                no="The requester is on the owning team, or the policy lets the build cop pull this test without the "
                   "owners' agreement."),
        }


WORKFLOW = Workflow(
    name="cicd_failure_triage",
    title="CI/CD failure triage",
    description="Diagnosing failed CI/CD pipelines and deciding whether a merge or a release may proceed.",
    domain="software delivery / build and release engineering",
    schemas=[PrCheckFailure(), MainBranchRed(), CanaryDeployHalt(), NightlyPerfRegression(), DependencyUpdateBreakage(),
             FlakyTestQuarantine()],
    notes="Human build engineers and release captains reading CI systems' output; no AI agents, no security "
          "alerts, no change-board approvals (see it_change_approval).",
)
