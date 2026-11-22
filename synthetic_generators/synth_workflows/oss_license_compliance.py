"""Open-source licence compliance: how a company's open-source program office (OSPO) should handle third-party
code: a new dependency, a pre-release scan, an outside contribution to its own project, an upstream
relicensing, and a request for GPL source for a device it ships.

Packages, products, companies and people are fictional. Licence names are the public SPDX identifiers; the
company policy quoted in the states (tiers, who may approve) is invented for this generator, and the cases do
not give legal advice.

Each schema draws a `regime` first. Borderline cases sit on a policy line (an LGPL library that is statically
linked only on one platform, a release eight working days out with a nine-day fix); conflicting cases pit a
confident label against the evidence ("basically MIT" on a package whose LICENSE file says AGPL), or a clean
scan against one finding in the shipped image.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    between,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    join_sentences,
    kv_lines,
    money,
    near,
    noisy,
    noul,
    person,
    pick,
    pick_w,
    prune,
    rand_date,
    regime,
    score,
    show,
    table,
)

PACKAGES = [("quillstream", "npm"), ("libvorpal", "C/C++ (vendored)"), ("tessellate-js", "npm"), ("hexbind", "PyPI"),
            ("mariner-http", "Maven"), ("fastgrid", "npm"), ("pyzephyr", "PyPI"), ("cobbleparse", "crates.io"),
            ("loamdb-client", "Go modules"), ("stratus-ui", "npm"), ("brindle", "PyPI"), ("okapi-codec", "C/C++ (vendored)")]
TIER = {"MIT": "A", "BSD-3-Clause": "A", "Apache-2.0": "A", "ISC": "A", "MPL-2.0": "B", "EPL-2.0": "B",
        "LGPL-2.1-or-later": "B/C", "LGPL-3.0-only": "B/C", "GPL-2.0-only": "C", "GPL-3.0-or-later": "C",
        "CC-BY-SA-4.0": "C", "AGPL-3.0-only": "D", "SSPL-1.0": "D", "BUSL-1.1": "D", "CC-BY-NC-4.0": "D",
        "NOASSERTION": "C/D"}
OSS_POLICY = ("Open-source policy OSP-2 (summary). Tier A, approved by default: MIT, BSD, ISC, Apache-2.0. Tier B, an "
              "engineering lead may approve with notices recorded: MPL-2.0, EPL-2.0, and LGPL libraries that are "
              "dynamically linked and unmodified. Tier C, legal review required: GPL family, LGPL when statically linked "
              "or modified, CC-BY-SA, custom or unclear licences. Tier D, not allowed in anything distributed or offered "
              "over a network: AGPL, SSPL, source-available and non-commercial licences, and code with no licence.")
DIST_NOTE = ("OSPO guidance: 'distribution' means giving a copy to anyone outside the company (customers, app "
             "stores, devices we sell, partner-run servers); running a hosted service is not distribution, except "
             "that network-copyleft licences also cover users who interact with the service over a network.")
USES = {
    "same_process_dynamic": "imported as a library, dynamically linked",
    "static_link": "statically linked into the main binary",
    "subprocess_cli": "invoked as a separate command-line program",
    "network_service": "run as a separate service called over HTTP",
    "build_only": "used only at build time; not in the shipped artefact",
    "vendored_modified": "copied into our repository and modified",
}
PRODUCTS = {
    "saas": "a hosted web application (customers use it in the browser)",
    "on_prem": "an on-premises installer shipped to enterprise customers",
    "mobile": "a mobile app distributed through app stores",
    "firmware": "firmware for a device we sell",
    "internal": "an internal tool used only by employees",
    "partner_hosted": "a service we host on partner-operated servers",
}


NATIVE = ("C/C++ (vendored)", "crates.io", "Go modules")


def _use_text(f):
    """How the component is used, in the words that fit its ecosystem (script packages are bundled, not linked)."""
    if f["ecosystem"] not in NATIVE:
        if f["usage"] == "same_process_dynamic":
            return "imported as a library in the same process"
        if f["usage"] == "static_link":
            return "bundled into the application's single build artefact"
    return USES[f["usage"]]


def _requester(rng):
    return person(rng)


# =========================================================================== schema 1: dependency intake
class DependencyIntake(Schema):
    name = "dependency_intake"
    description = "An engineer asks to add a third-party open-source package; the OSPO decides under its licence policy."
    surfaces = {"json": 0.5, "intake_ticket": 0.25, "scan_excerpt": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        pkg, eco = pick(rng, PACKAGES)
        declared = pick_w(rng, {"MIT": 4, "Apache-2.0": 3, "BSD-3-Clause": 2, "MPL-2.0": 2, "LGPL-2.1-or-later": 2,
                                "GPL-3.0-or-later": 1, "AGPL-3.0-only": 1, "BUSL-1.1": 1})
        file_lic = declared
        scanner_conf = pick_w(rng, {"high": 5, "medium": 2, "low": 1})
        use = pick(rng, list(USES))
        product = pick(rng, list(PRODUCTS))
        modified = use == "vendored_modified" or chance(rng, 0.1)
        transitive = pick_w(rng, {0: 6, 1: 2, 2: 1})
        transitive_lic = pick(rng, ["GPL-2.0-only", "LGPL-3.0-only", "CC-BY-SA-4.0", "NOASSERTION"]) if transitive else None
        last_release_months = pick(rng, [1, 3, 8, 20, 40])
        alt = pick_w(rng, {"drop_in": 2, "some_work": 3, "none_known": 2})
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                declared = file_lic = pick(rng, ["MIT", "Apache-2.0", "BSD-3-Clause"])
                scanner_conf, transitive, transitive_lic, modified = "high", 0, None, False
                use = pick(rng, ["same_process_dynamic", "build_only", "subprocess_cli"])
            else:
                declared = file_lic = pick(rng, ["AGPL-3.0-only", "BUSL-1.1"])
                product = pick(rng, ["saas", "on_prem"])
                use = pick(rng, ["same_process_dynamic", "static_link", "vendored_modified"])
        elif reg == "borderline":
            b = pick(rng, ["lgpl_static", "copyleft_internal", "mpl_modified", "transitive"])
            if b == "lgpl_static":
                declared = file_lic = "LGPL-2.1-or-later"
                use = pick(rng, ["static_link", "same_process_dynamic"])
                product = pick(rng, ["mobile", "on_prem", "firmware"])
            elif b == "copyleft_internal":
                declared = file_lic = pick(rng, ["GPL-3.0-or-later", "AGPL-3.0-only"])
                product = pick(rng, ["internal", "partner_hosted"])
            elif b == "mpl_modified":
                declared = file_lic = "MPL-2.0"
                use, modified = "vendored_modified", True
            else:
                declared = file_lic = pick(rng, ["MIT", "Apache-2.0"])
                transitive, transitive_lic = 1, pick(rng, ["GPL-2.0-only", "NOASSERTION", "CC-BY-SA-4.0"])
        else:
            variant = pick(rng, ["metadata_vs_file", "gpl_but_subprocess", "permissive_but_abandoned"])
            if variant == "metadata_vs_file":
                declared = "MIT"
                file_lic = pick(rng, ["AGPL-3.0-only", "GPL-3.0-or-later", "NOASSERTION"])
                scanner_conf = pick(rng, ["medium", "low"])
            elif variant == "gpl_but_subprocess":
                declared = file_lic = pick(rng, ["GPL-2.0-only", "GPL-3.0-or-later"])
                use, product = "subprocess_cli", pick(rng, ["on_prem", "firmware", "mobile"])
            else:
                declared = file_lic = "Apache-2.0"
                last_release_months, alt = pick(rng, [30, 48, 60]), "none_known"
        return {
            "regime": reg, "variant": variant, "package": pkg, "ecosystem": eco,
            "version": f"{rng.randint(0, 6)}.{rng.randint(0, 19)}.{rng.randint(0, 9)}",
            "declared_licence": declared, "licence_file": file_lic, "scanner_confidence": scanner_conf,
            "usage": use, "product": product, "modified": modified, "copyleft_transitive_deps": transitive,
            "transitive_licence": transitive_lic, "months_since_release": last_release_months,
            "alternative": alt, "requester": _requester(rng), "team": pick(rng, ["Payments", "Mobile", "Data Platform",
                                                                                 "Device Firmware", "Internal Tools", "Search"]),
            "ticket": f"OSPO-{rng.randint(1000, 9999)}", "opened": rand_date(rng),
            "requester_note": ("It's basically MIT, should be quick." if variant == "metadata_vs_file" and chance(rng, 0.6)
                               else pick(rng, ["Need this for the Q3 feature, it's widely used.", "Licence looks fine to me.",
                                               "Happy to swap it out if there's a problem.", None, None])),
        }

    def _licence_line(self, f):
        if f["declared_licence"] == f["licence_file"]:
            return f"{f['declared_licence']} (package metadata and LICENSE file agree)"
        lf = "no recognisable licence text" if f["licence_file"] == "NOASSERTION" else f["licence_file"]
        return f"metadata says {f['declared_licence']}; LICENSE file in the repository: {lf}"

    def _transitive(self, f):
        if not f["copyleft_transitive_deps"]:
            return None
        lic = "unidentified licence" if f["transitive_licence"] == "NOASSERTION" else f["transitive_licence"]
        return f"{f['copyleft_transitive_deps']} transitive dependenc{'y' if f['copyleft_transitive_deps'] == 1 else 'ies'} under {lic}"

    def render(self, f, rng, surface):
        alt = {"drop_in": "a drop-in alternative under MIT exists", "some_work": "alternatives exist but need integration work",
               "none_known": "no alternative identified"}[f["alternative"]]
        if surface == "json":
            return dumps(prune({
                "request": {"ticket": f["ticket"], "opened": f["opened"].isoformat(), "requested_by": f["requester"],
                            "team": f["team"], "note": f["requester_note"]},
                "component": {"name": f["package"], "version": f["version"], "ecosystem": f["ecosystem"],
                              "licence": self._licence_line(f), "scanner_confidence": f["scanner_confidence"],
                              "transitive_findings": self._transitive(f),
                              "last_release_months_ago": f["months_since_release"]},
                "intended_use": {"how": _use_text(f), "modified": f["modified"], "product": PRODUCTS[f["product"]],
                                 "alternatives": alt},
                "policy": OSS_POLICY, "guidance": DIST_NOTE,
            }))
        if surface == "intake_ticket":
            rows = [("Ticket", f["ticket"]), ("Opened", fmt_date(f["opened"], rng)), ("Requester", f"{f['requester']} ({f['team']})"),
                    ("Package", f"{f['package']} {f['version']} [{f['ecosystem']}]"), ("Licence", self._licence_line(f)),
                    ("How it will be used", _use_text(f)), ("Modified", "yes" if f["modified"] else "no"),
                    ("Ships in", PRODUCTS[f["product"]]), ("Alternatives", alt)]
            note = f"\n\nRequester comment: \"{noisy(f['requester_note'], rng, 0.15)}\"" if f["requester_note"] else ""
            return ("NEW DEPENDENCY REQUEST\n" + kv_lines(rows, rng) + note + f"\n\n{OSS_POLICY}\n{DIST_NOTE}")
        lines = [f"$ ospo-scan --package {f['package']}@{f['version']}",
                 f"component      : {f['package']} ({f['ecosystem']})",
                 f"declared       : {f['declared_licence']}",
                 f"detected (file): {f['licence_file']}   confidence={f['scanner_confidence']}",
                 f"policy tier    : {TIER.get(f['licence_file'], '?')}",
                 f"transitive     : {self._transitive(f) or 'no copyleft or unknown licences'}",
                 f"last release   : {f['months_since_release']} months ago"]
        ctx = [f"{f['requester']} ({f['team']}) wants it in {PRODUCTS[f['product']]}, {_use_text(f)}"
               + (", with local modifications." if f["modified"] and f["usage"] != "vendored_modified" else "."),
               alt[0].upper() + alt[1:] + "."]
        return "\n".join(lines) + "\n\n" + noisy(join_sentences(ctx), rng, 0.1) + f"\n\n{OSS_POLICY}\n{DIST_NOTE}"

    def questions(self):
        return {
            "intake_decision": choice("What should the open-source program office decide on this dependency request?", {
                "approve": "Approve with the standard notice file: the governing licence (the LICENSE file, where it "
                           "disagrees with the metadata) is Tier A and no transitive dependency is GPL, CC-BY-SA or "
                           "unidentified.",
                "approve_with_conditions": "Approve with conditions such as dynamic linking only: the governing "
                                           "licence is Tier B (MPL-2.0, EPL-2.0, or LGPL dynamically linked and "
                                           "unmodified) and no transitive dependency raises it.",
                "escalate_to_legal": "Send to legal counsel: Tier C (GPL family, LGPL bundled, statically linked or "
                                     "modified, CC-BY-SA, unclear licence), a GPL, CC-BY-SA or unidentified transitive "
                                     "dependency, or a Tier D licence used only internally or at build time.",
                "reject_find_alternative": "Refuse it and ask for another component: a Tier D licence in a product "
                                           "that is distributed or offered over a network.",
            }),
            "licence_uncertainty": score("How uncertain is the licence that actually governs this component?", [
                "Settled: one clear, consistent licence.",
                "Minor doubt: small inconsistencies that do not change the tier.",
                "Real doubt: sources disagree, or a dependency's licence is unknown.",
                "Unknown: no reliable licence can be identified.",
            ]),
            "obligation_burden": score("How heavy are the obligations the team would take on by using it as planned?", [
                "Negligible: keep the licence text.",
                "Light: attribution and notice files in the product.",
                "Substantial: offering source for the component or our changes to it, and keeping build records.",
                "Unacceptable: disclosing our own product's source, or terms the policy forbids.",
            ]),
            "distribution_occurs": noul(
                "The product, as described, distributes this component to people outside the company.",
                yes="Copies reach customers, devices, app stores or servers the company does not run.",
                no="The component stays on company-run systems or is used only at build time."),
            "copyleft_triggered": noul(
                "Using the component as planned would trigger its licence's source-sharing obligations.",
                yes="The licence is copyleft and the planned use distributes it, links it in a covered way, or serves it over a network under a network clause.",
                no="The licence is permissive, or the planned use falls outside what its copyleft terms cover."),
        }


# =========================================================================== schema 2: release scan gate
FINDING_KINDS = {
    "gpl_in_image": "a GPL-2.0-only library",
    "agpl_service": "an AGPL-3.0-only component",
    "unknown_licence": "a vendored file with no licence and an unrelated copyright header",
    "missing_notices": "third-party notices missing for {n} permissive component{s}",
    "nc_asset": "an icon set under CC-BY-NC-4.0",
}
SHORT_PRODUCTS = {"saas": "hosted web app", "on_prem": "on-prem installer", "mobile": "mobile app",
                  "firmware": "device firmware"}
RELEASE_POLICY = ("Release licence gate RG-1: a build may ship only when no copyleft or source-available component in the "
                  "shipped artefact is out of compliance and every bundled third-party component appears in the notices "
                  "file. Findings confined to tests, tooling or docs do not block. The general counsel may grant a "
                  "waiver of up to 30 days when a fix is committed for the next release.")
LOCATIONS = {"shipped": "in the shipped artefact", "tests": "only in test fixtures, not shipped",
             "build_tool": "in build tooling that is not shipped", "docs": "in documentation examples"}


class ReleaseScanGate(Schema):
    name = "release_scan_gate"
    description = "A pre-release software-composition scan with open licence findings, days before general availability."
    surfaces = {"json": 0.5, "scan_report": 0.25, "release_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        product = pick(rng, ["saas", "on_prem", "mobile", "firmware"])
        top = pick(rng, list(FINDING_KINDS))
        loc = pick_w(rng, {"shipped": 5, "tests": 2, "build_tool": 2, "docs": 1})
        match = pick_w(rng, {"full_file": 4, "snippet": 3, "licence_text_only": 2})
        notices_missing = rng.randint(0, 40)
        others = rng.randint(0, 6)
        days_to_ga = pick(rng, [2, 5, 9, 15, 30, 60])
        fix_days = pick(rng, [1, 3, 6, 10, 20])
        customer_clause = pick_w(rng, {None: 3, "sbom_required": 2, "no_copyleft": 1})
        waiver_history = pick_w(rng, {0: 5, 1: 2, 3: 1})
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                top, loc, notices_missing, others = "missing_notices", "shipped", rng.randint(1, 8), 0
                days_to_ga, fix_days = pick(rng, [15, 30, 60]), 1
            else:
                top, loc, match = pick(rng, ["gpl_in_image", "agpl_service"]), "shipped", "full_file"
                customer_clause = "no_copyleft"
                product = "on_prem" if top == "gpl_in_image" else "saas"
        elif reg == "borderline":
            fix_days = near(rng, days_to_ga, max(2, days_to_ga * 0.3), lo=1, integer=True)
            match = pick(rng, ["snippet", "licence_text_only"])
        else:
            variant = pick(rng, ["serious_but_test_only", "tiny_snippet_shipped", "clean_but_contract"])
            if variant == "serious_but_test_only":
                top, loc = pick(rng, ["gpl_in_image", "agpl_service"]), pick(rng, ["tests", "build_tool"])
            elif variant == "tiny_snippet_shipped":
                top, loc, match = "unknown_licence", "shipped", "snippet"
            else:
                top, loc, customer_clause = "gpl_in_image", "shipped", "no_copyleft"
                match, days_to_ga = "licence_text_only", pick(rng, [2, 5])
        if top == "agpl_service" and product in ("firmware", "mobile"):
            product = "saas"
        if top == "missing_notices":
            loc = "shipped"
        release = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "product": product, "top_finding": top, "location": loc,
            "match_type": match, "notices_missing": notices_missing if top != "missing_notices" else max(1, notices_missing),
            "other_findings": others, "days_to_ga": days_to_ga, "fix_estimate_days": fix_days,
            "customer_clause": customer_clause, "waivers_this_year": waiver_history,
            "component": pick(rng, [p for p, e in PACKAGES if product != "firmware" or e in NATIVE]),
            "snippet_lines": rng.randint(6, 60),
            "release": f"{pick(rng, ['Atlas', 'Beacon', 'Cinder', 'Delta', 'Ember'])} {rng.randint(2, 9)}.{rng.randint(0, 9)}",
            "ga_date": release, "release_manager": person(rng), "ospo_lead": person(rng),
        }

    def _finding(self, f):
        return FINDING_KINDS[f["top_finding"]].format(n=f["notices_missing"], s="" if f["notices_missing"] == 1 else "s")

    def _match(self, f):
        if f["top_finding"] == "missing_notices":
            return "notice-file check"
        if f["top_finding"] == "nc_asset":
            return "asset licence metadata"
        return {"full_file": "whole-file match", "snippet": f"{f['snippet_lines']}-line snippet match",
                "licence_text_only": "licence text found in a header; code match not confirmed"}[f["match_type"]]

    def render(self, f, rng, surface):
        clause = {None: None, "sbom_required": "largest customer contract requires an accurate SBOM at delivery",
                  "no_copyleft": "a key customer contract warrants no copyleft code in delivered software"}[f["customer_clause"]]
        if surface == "json":
            return dumps(prune({
                "release": {"name": f["release"], "product": SHORT_PRODUCTS[f["product"]], "ga_date": f["ga_date"].isoformat(),
                            "days_to_ga": f["days_to_ga"], "release_manager": f["release_manager"]},
                "top_finding": {"issue": self._finding(f), "component": f["component"], "where": LOCATIONS[f["location"]],
                                "match": self._match(f), "estimated_fix_days": f["fix_estimate_days"]},
                "other_findings": {"open": f["other_findings"],
                                   "missing_notices": f["notices_missing"] if f["top_finding"] != "missing_notices" else None},
                "context": {"customer_terms": clause, "licence_waivers_granted_this_year": f["waivers_this_year"]},
                "policy": RELEASE_POLICY,
            }))
        if surface == "scan_report":
            rows = [["1 (top)", self._finding(f), f["component"], LOCATIONS[f["location"]], self._match(f)]]
            if f["top_finding"] != "missing_notices" and f["notices_missing"]:
                rows.append(["2", f"notices missing for {f['notices_missing']} permissive components", "various", "shipped", "-"])
            if f["other_findings"]:
                rows.append([str(len(rows) + 1), f"{f['other_findings']} low-severity findings", "various", "-", "-"])
            return (f"SCA SCAN -- {f['release']} ({SHORT_PRODUCTS[f['product']]})\nGA {fmt_date(f['ga_date'], rng)}, "
                    f"{f['days_to_ga']} days out. Estimated fix for top finding: {f['fix_estimate_days']} days.\n\n"
                    + table(["#", "finding", "component", "where", "match"], rows, rng)
                    + (f"\n\nContract note: {clause}." if clause else "") + f"\n\n{RELEASE_POLICY}")
        msgs = [(f["release_manager"], f"scan for {f['release']} came back with {self._finding(f)} ({f['component']}), "
                                      f"{LOCATIONS[f['location']]}"),
                (f["ospo_lead"], f"match type: {self._match(f)}"),
                (f["release_manager"], f"eng says a fix is about {f['fix_estimate_days']} day{'' if f['fix_estimate_days'] == 1 else 's'};"
                                      f" GA is in {f['days_to_ga']}"),
                (f["release_manager"], pick(rng, ["can we ship and fix in a patch?", "any chance of a waiver?",
                                                  "I'd rather not slip the date"]))]
        if clause:
            msgs.append((f["ospo_lead"], f"note {clause}"))
        if f["waivers_this_year"]:
            msgs.append((f["ospo_lead"], f"we've granted {f['waivers_this_year']} waiver(s) this year already"))
        t = rand_date(rng)
        out = [f"[{9 + i}:{rng.randint(10, 59)}] {who.split()[0].lower()}: {noisy(m, rng, 0.1)}" for i, (who, m) in enumerate(msgs)]
        return f"#release-{f['release'].split()[0].lower()} ({fmt_date(t, rng)})\n" + "\n".join(out) + f"\n\n{RELEASE_POLICY}"

    def questions(self):
        return {
            "release_gate_call": choice("What should the release gate decide?", {
                "hold_for_remediation": "Hold the release until the finding is fixed.",
                "ship": "Ship on schedule; nothing found needs to block it.",
                "ship_after_notice_fixes": "Ship once the notices and attribution files are completed.",
                "ship_under_legal_waiver": "Ship on a time-limited legal waiver with a committed fix in the next release.",
            }),
            "remediation_path": choice("What remediation should the team pursue for the top finding?", {
                "comply_by_offering_source": "Keep the component and meet its terms, for example by offering the corresponding source.",
                "fix_notices_only": "Only attribution and notice files need fixing.",
                "no_action_not_shipped": "Nothing; the finding is a misidentification or sits in code that is not shipped.",
                "obtain_other_licence": "Obtain the component under different terms, such as a commercial licence or the author's permission.",
                "replace_component": "Remove or replace the component in the shipped build.",
            }),
            "breach_if_shipped_today": noul(
                "At least one finding would put the company in breach of a licence if the build shipped today.",
                yes="A shipped component's terms are not being met, or its use is forbidden outright.",
                no="The findings are unshipped, misidentified, or only need notices that can be added in time."),
            "ga_date_pressure": score("How hard does the release date constrain the remediation options?", [
                "Loosely: the fix estimate fits with 5 or more days to spare before GA.",
                "Somewhat: the fix fits, but with fewer than 5 days to spare.",
                "Severely: the fix estimate runs past GA, which is more than 5 days away.",
                "Totally: the fix estimate runs past GA and GA is 5 days or less away, so only a waiver or a slip "
                "remains.",
            ]),
            "exposure_if_shipped": score("If the build shipped as it stands, how exposed would the company be?", [
                "Negligible: nothing a rights holder could object to.",
                "Minor: an attribution lapse that is easily cured.",
                "Moderate: a licence obligation unmet, but curable by publishing source or notices.",
                "Significant: a licence breach that forces a product change, or a contract warranty problem.",
                "Serious: forced disclosure of proprietary code, or breach of a customer contract.",
            ]),
        }


# =========================================================================== schema 3: inbound contribution
class InboundContribution(Schema):
    name = "inbound_contribution"
    description = "An outside pull request to one of the company's open-source projects, with questions about where the code came from."
    surfaces = {"json": 0.5, "pull_request": 0.25, "maintainer_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        project_lic = pick_w(rng, {"Apache-2.0": 4, "MIT": 3, "MPL-2.0": 1})
        contributor = pick_w(rng, {"individual": 4, "employee_of_other_company": 3, "new_anonymous_account": 2,
                                   "our_contractor": 1})
        signoff = pick_w(rng, {"dco_signed": 4, "cla_signed": 2, "none": 2})
        provenance = pick_w(rng, {"clean": 4, "adapted_gpl": 1, "qa_site_snippet": 2, "foreign_header": 1,
                                  "sdk_sample_match": 1})
        lines = pick(rng, [12, 40, 180, 600, 2400])
        employer_domain = contributor == "employee_of_other_company" and chance(rng, 0.7)
        employer_policy = pick_w(rng, {"allows": 2, "requires_approval": 2, None: 3}) if contributor == "employee_of_other_company" else None
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                contributor, signoff, provenance, lines = "individual", pick(rng, ["dco_signed", "cla_signed"]), "clean", pick(rng, [12, 40, 180])
            else:
                contributor, signoff, provenance, lines = "new_anonymous_account", "none", pick(rng, ["adapted_gpl", "sdk_sample_match"]), pick(rng, [600, 2400])
        elif reg == "borderline":
            provenance = pick(rng, ["qa_site_snippet", "foreign_header", "clean"])
            lines = pick(rng, [40, 180, 600])
            if contributor == "employee_of_other_company":
                employer_policy = pick(rng, ["requires_approval", None])
        else:
            variant = pick(rng, ["great_code_no_signoff", "signed_but_copied", "employer_says_no"])
            if variant == "great_code_no_signoff":
                signoff, provenance, lines = "none", "clean", pick(rng, [600, 2400])
            elif variant == "signed_but_copied":
                signoff, provenance = "dco_signed", pick(rng, ["adapted_gpl", "sdk_sample_match", "qa_site_snippet"])
            else:
                contributor, employer_domain, employer_policy, signoff = "employee_of_other_company", True, "requires_approval", "dco_signed"
        return {
            "regime": reg, "variant": variant, "project_licence": project_lic, "contributor_type": contributor,
            "signoff": signoff, "provenance_signal": provenance, "lines_added": lines,
            "uses_employer_email": employer_domain, "employer_policy": employer_policy,
            "project": pick(rng, ["kestrel-sdk", "tidepool", "gantry-cli", "lumen-charts", "shoal"]),
            "pr": rng.randint(300, 4800), "author_handle": pick(rng, ["dev", "code", "gh", "oss"]) + str(rng.randint(10, 9999)),
            "account_age_days": rng.randint(2, 30) if contributor == "new_anonymous_account" else rng.randint(200, 4000),
            "maintainer": person(rng), "feature": pick(rng, ["a streaming parser", "retry logic for the HTTP client",
                                                              "a Windows build fix", "a new chart type", "SIMD fast paths"]),
            "similarity_pct": rng.randint(35, 95) if provenance in ("adapted_gpl", "sdk_sample_match") else None,
            "opened": rand_date(rng),
        }

    def _prov_text(self, f):
        return {"clean": None,
                "adapted_gpl": f"code scanner: {f['similarity_pct']}% similarity to a file in a GPL-3.0 project",
                "qa_site_snippet": "a comment links to a Q&A-site answer (content there is CC-BY-SA)",
                "foreign_header": "one file carries a different company's copyright header",
                "sdk_sample_match": f"code scanner: {f['similarity_pct']}% match to sample code from a proprietary vendor SDK"}[f["provenance_signal"]]

    def _who(self, f):
        return {"individual": "an individual contributor with a long public history",
                "employee_of_other_company": "an engineer at another company" + (" using their work email" if f["uses_employer_email"] else ""),
                "new_anonymous_account": f"an account created {f['account_age_days']} days ago with no other activity",
                "our_contractor": "a contractor currently engaged by us"}[f["contributor_type"]]

    def render(self, f, rng, surface):
        sign = {"dco_signed": "DCO sign-off on every commit", "cla_signed": "contributor licence agreement on file",
                "none": "no sign-off and no CLA"}[f["signoff"]]
        emp = {None: None, "allows": "their employer's published policy allows open-source contributions",
               "requires_approval": "their employer's policy requires prior approval for outside contributions"}.get(f["employer_policy"])
        prov = self._prov_text(f)
        if surface == "json":
            return dumps(prune({
                "pull_request": {"project": f["project"], "number": f["pr"], "opened": f["opened"].isoformat(),
                                 "adds": f"{f['feature']} ({f['lines_added']} lines)", "author": f["author_handle"],
                                 "author_profile": self._who(f), "sign_off": sign},
                "provenance": {"signals": prov or "no scanner or reviewer flags", "employer": emp},
                "project": {"licence": f["project_licence"], "contribution_rule": "inbound = outbound; DCO or CLA required"},
            }))
        if surface == "pull_request":
            desc = noisy(f"This PR adds {f['feature']}. " + pick(rng, ["Tested locally on Linux and macOS.",
                                                                     "Happy to adjust style.", "Been using this in production for months."]), rng, 0.15)
            checks = [f"licence/cla: {'pass' if f['signoff'] != 'none' else 'fail'} ({sign})",
                      f"provenance-scan: {prov or 'no matches'}"]
            return (f"PR #{f['pr']} on {f['project']} ({f['project_licence']}) by @{f['author_handle']}\n"
                    f"+{f['lines_added']} lines\n\n{desc}\n\nChecks:\n" + bullets(checks, rng)
                    + f"\n\nAbout the author: {self._who(f)}" + (f"; {emp}." if emp else "."))
        note = [f"PR #{f['pr']} to {f['project']} ({f['project_licence']}) adds {f['feature']}, {f['lines_added']} lines.",
                f"Author is {self._who(f)}.", f"Paperwork: {sign}.", (prov[0].upper() + prov[1:] + "." if prov else "Scanner is quiet."),
                (emp[0].upper() + emp[1:] + "." if emp else ""),
                pick(rng, ["Code quality is good.", "Would really like this feature in.", "Review comments were addressed quickly.", ""])]
        return f"MAINTAINER NOTE -- {f['maintainer']}\n\n" + noisy(join_sentences(note), rng, 0.1)

    def questions(self):
        return {
            "contribution_decision": choice("What should the maintainers do with this pull request?", {
                "decline": "Close it; the provenance problems cannot reasonably be fixed.",
                "escalate_to_legal": "Hold it and ask legal counsel before going further.",
                "merge": "Merge it after normal code review.",
                "request_rewrite_of_parts": "Ask the author to remove or rewrite the parts whose origin is in doubt.",
                "request_signoff_first": "Ask for the missing sign-off or agreement before anything else.",
            }),
            "provenance_confidence": score("How confident can the maintainers be that the author had the right to contribute this code?", [
                "Low: strong signs it came from somewhere else.",
                "Guarded: an unresolved signal about origin or ownership.",
                "Reasonable: minor questions that paperwork would settle.",
                "High: original work with proper sign-off.",
            ]),
            "third_party_terms_compatible": noul(
                "Any third-party code in the contribution comes under terms compatible with the project's licence.",
                yes="There is no third-party code, or it is under terms the project's licence can absorb.",
                no="Some code appears to come from a copyleft, share-alike, proprietary or unknown source."),
            "ownership_challenge_risk": score("How likely is someone else, such as the author's employer, to claim rights in this code?", [
                "Remote.",
                "Possible, but unlikely to be pursued.",
                "Real: an employer or original author could plausibly object.",
                "High: a third party clearly has a claim.",
            ]),
            "copied_code_suspected": noul("Part of the contribution appears to have been copied from another source."),
        }


# =========================================================================== schema 4: upstream relicensing
COMPONENTS = [("a search engine", "Search"), ("a message queue", "MQ"), ("a key-value store", "KV"),
              ("an infrastructure-as-code tool", "Form"), ("a workflow scheduler", "Flow")]
NEW_TERMS = {
    "busl_no_hosting": "BUSL-1.1 with a clause barring any hosted offering that competes with the vendor",
    "sspl": "SSPL-1.0, which requires releasing the source of the whole service stack if the software is offered as a service",
    "custom_no_competition": "a custom source-available licence barring use 'in a competing product'",
}


class UpstreamRelicensing(Schema):
    name = "upstream_relicensing"
    description = "A core dependency switched from a permissive licence to source-available terms; decide how to respond."
    surfaces = {"json": 0.5, "decision_memo": 0.25, "arch_review_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        comp, key = pick(rng, COMPONENTS)
        terms = pick(rng, list(NEW_TERMS))
        our_use = pick_w(rng, {"internal_backend": 4, "managed_offering": 2, "on_prem_bundle": 2, "internal_tooling": 2})
        pinned_months = pick(rng, [1, 4, 9, 18])
        cves_since = pick_w(rng, {0: 4, 1: 2, 3: 1})
        eol_months = pick(rng, [None, 3, 6, 12])
        fork = pick_w(rng, {"healthy": 2, "young": 3, "none": 2})
        migration_weeks = pick(rng, [2, 6, 12, 26])
        quote = int(round(rng.uniform(20000, 400000), -3))
        budget = int(round(rng.uniform(20000, 250000), -3))
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                our_use, cves_since, fork = "internal_tooling", 0, pick(rng, ["healthy", "young"])
            else:
                our_use, terms = "managed_offering", pick(rng, ["busl_no_hosting", "sspl"])
                fork, cves_since = "none", pick(rng, [1, 3])
        elif reg == "borderline":
            our_use = pick(rng, ["internal_backend", "on_prem_bundle"])
            terms = pick(rng, ["custom_no_competition", "busl_no_hosting"])
            fork = "young"
            quote = int(round(near(rng, budget, budget * 0.2, lo=5000), -3))
        else:
            variant = pick(rng, ["cheap_licence_bad_fit", "healthy_fork_heavy_migration", "safe_use_but_cves"])
            if variant == "cheap_licence_bad_fit":
                quote, our_use = int(round(budget * rng.uniform(0.3, 0.6), -3)), "managed_offering"
            elif variant == "healthy_fork_heavy_migration":
                fork, migration_weeks = "healthy", pick(rng, [26, 40])
            else:
                our_use, cves_since, eol_months = "internal_backend", 3, pick(rng, [3, 6])
        return {
            "regime": reg, "variant": variant, "component": comp, "component_key": key, "new_terms": terms,
            "our_use": our_use, "pinned_version_age_months": pinned_months, "security_fixes_since_pin": cves_since,
            "old_line_support_ends_months": eol_months, "community_fork": fork, "migration_weeks": migration_weeks,
            "licence_quote_usd": quote, "budget_usd": budget, "announced": rand_date(rng),
            "name": pick(rng, ["Driftwood", "Ironbark", "Saltmarsh", "Upland", "Fernhill"]) + key,
            "owner": person(rng), "architect": person(rng),
        }

    def _use_text(self, f):
        return {"internal_backend": "runs inside our own SaaS backend; customers never touch it directly",
                "managed_offering": f"we sell a hosted version of {f['name']} to customers as a managed service",
                "on_prem_bundle": "bundled inside the on-premises product we ship to customers",
                "internal_tooling": "used by our internal build and ops tooling only"}[f["our_use"]]

    def _fork_text(self, f):
        return {"healthy": "a community fork under the old licence has regular releases and several companies contributing",
                "young": "a community fork was announced last month; one release so far",
                "none": "no fork has emerged"}[f["community_fork"]]

    def render(self, f, rng, surface):
        eol = (f"the last permissive release line gets security fixes for {f['old_line_support_ends_months']} more months"
               if f["old_line_support_ends_months"] else "no support commitment for the last permissive release")
        if surface == "json":
            return dumps(prune({
                "change": {"component": f"{f['name']} ({f['component']})", "announced": f["announced"].isoformat(),
                           "old_licence": "Apache-2.0", "new_licence": NEW_TERMS[f["new_terms"]]},
                "our_position": {"use": self._use_text(f), "pinned_release_age_months": f["pinned_version_age_months"],
                                 "security_fixes_released_since_pin": f["security_fixes_since_pin"], "old_line_support": eol},
                "options": {"community_fork": self._fork_text(f), "migration_estimate_weeks": f["migration_weeks"],
                            "commercial_licence_quote_usd_per_year": f["licence_quote_usd"],
                            "budget_available_usd": f["budget_usd"]},
            }))
        if surface == "decision_memo":
            rows = [("Component", f"{f['name']} -- {f['component']}"), ("Announced", fmt_date(f["announced"], rng)),
                    ("New licence", NEW_TERMS[f["new_terms"]]), ("How we use it", self._use_text(f)),
                    ("Our pinned release", f"{f['pinned_version_age_months']} months old, {f['security_fixes_since_pin']} "
                                           f"security fix(es) upstream since"),
                    ("Old line support", eol), ("Community fork", self._fork_text(f)),
                    ("Migration estimate", f"{f['migration_weeks']} engineer-weeks"),
                    ("Commercial licence", f"{money(f['licence_quote_usd'], rng)}/yr quoted; budget {money(f['budget_usd'], rng)}")]
            return f"DECISION MEMO (draft) -- prepared by {f['owner']}\n\n" + kv_lines(rows, rng)
        msgs = [(f["owner"], f"{f['name']} moved from Apache-2.0 to {NEW_TERMS[f['new_terms']]}"),
                (f["owner"], f"for context: {self._use_text(f)}"),
                (f["architect"], f"we're {f['pinned_version_age_months']} month{'' if f['pinned_version_age_months'] == 1 else 's'} behind anyway; "
                                 f"{f['security_fixes_since_pin']} security fix(es) upstream since our pin, and {eol}"),
                (f["architect"], self._fork_text(f)),
                (f["owner"], f"migration off it is ~{f['migration_weeks']} eng-weeks; vendor quoted "
                             f"{money(f['licence_quote_usd'], rng)}/yr against {money(f['budget_usd'], rng)} budget")]
        return ("Thread: #arch-review\n" + "\n".join(f"{who.split()[0]}: {noisy(m, rng, 0.1)}" for who, m in msgs))

    def questions(self):
        return {
            "relicensing_response": choice("How should the company respond to the licence change?", {
                "accept_new_terms": "Upgrade under the new licence; our use is allowed by its terms.",
                "buy_commercial_licence": "Buy a commercial licence from the vendor.",
                "migrate_to_alternative": "Plan a migration to a different component.",
                "move_to_community_fork": "Switch to the community fork that keeps the old licence.",
                "pin_last_permissive_release": "Stay on the last permissively licensed release for now.",
            }),
            "new_terms_restrict_our_use": noul(
                "The new licence terms prohibit or restrict the way the company uses the component.",
                yes="Our use falls within what the new terms bar or make conditional.",
                no="Our use sits outside the restricted uses, such as purely internal operation."),
            "pinning_security_exposure": score("How exposed would the company be if it froze on its current release?", [
                "Minimal: supported, and no security fixes are missing.",
                "Moderate: a fix or two behind, with support for a while yet.",
                "High: missing security fixes, with support ending soon.",
                "Severe: unsupported, with known vulnerabilities in a customer-facing path.",
            ]),
            "switching_cost": score("How disruptive would moving off the current licence path be?", [
                "Trivial: a drop-in fork or a cheap licence.",
                "Moderate: a few weeks of work, or a licence within budget.",
                "Heavy: months of migration, or a licence well over budget.",
                "Prohibitive: no realistic alternative in the next year.",
            ]),
            "fork_dependable": noul("The community fork is mature enough for the company to depend on."),
        }


# =========================================================================== schema 5: GPL source request for a shipped device
DEVICES = {"a home Wi-Fi router": "RX", "a smart thermostat": "TH", "a set-top box": "SB", "an industrial gateway": "IG",
           "a dashcam": "DC"}
REQUESTERS = {
    "owner": "a customer who owns the device",
    "non_owner": "a hobbyist who says they do not own the device but found the firmware online",
    "compliance_group": "a volunteer software-freedom compliance group, on behalf of an owner",
    "competitor": "an engineer whose email domain belongs to a competitor",
}


class SourceCodeRequest(Schema):
    name = "source_code_request"
    description = "A request for the GPL source of firmware in a shipped device, checked against what the company can supply."
    surfaces = {"json": 0.5, "request_ticket": 0.25, "compliance_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        device = pick(rng, list(DEVICES))
        requester = pick_w(rng, {"owner": 4, "non_owner": 2, "compliance_group": 2, "competitor": 1})
        years = round(between(rng, 0.3, 5.0), 1)
        offer = pick_w(rng, {"written_offer": 4, "source_on_website": 2, "none_found": 2})
        availability = pick_w(rng, {"exact_archived": 3, "near_tag_only": 3, "supplier_bsp_missing": 2})
        supplier_doubt = chance(rng, 0.5)
        modified = chance(rng, 0.6)
        scripts = pick_w(rng, {True: 3, False: 2})
        public = pick_w(rng, {None: 4, "forum_post": 2, "public_issue_tracker": 1, "press_enquiry": 1})
        days_open = rng.randint(1, 90)
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                requester, years, offer, availability, scripts, public = "owner", round(between(rng, 0.3, 2.0), 1), "written_offer", "exact_archived", True, None
                days_open = rng.randint(1, 10)
            else:
                requester, years, offer, availability, public = "compliance_group", round(between(rng, 0.5, 2.5), 1), "none_found", "supplier_bsp_missing", "public_issue_tracker"
                days_open = rng.randint(45, 120)
        elif reg == "borderline":
            years = near(rng, 3.0, 0.4, lo=0.5, digits=1)
            requester = pick(rng, ["non_owner", "owner"])
            availability = pick(rng, ["near_tag_only", "exact_archived"])
        else:
            variant = pick(rng, ["competitor_but_entitled", "owner_but_source_lost", "old_device_public_pressure"])
            if variant == "competitor_but_entitled":
                requester, offer, years = "competitor", "written_offer", round(between(rng, 0.5, 2.5), 1)
            elif variant == "owner_but_source_lost":
                requester, availability, scripts = "owner", "supplier_bsp_missing", False
            else:
                years, public, requester = round(between(rng, 3.5, 6.0), 1), pick(rng, ["forum_post", "press_enquiry"]), "non_owner"
        return {
            "regime": reg, "variant": variant, "device": device, "model": f"{DEVICES[device]}-{rng.randint(100, 990)}",
            "firmware": f"{rng.randint(1, 4)}.{rng.randint(0, 12)}.{rng.randint(0, 30)}", "years_since_last_shipped": years,
            "requester": requester, "offer_with_product": offer, "source_availability": availability,
            "kernel_modified": modified, "build_scripts_archived": scripts, "public_attention": public,
            "supplier_patches_uncertain": supplier_doubt and availability == "near_tag_only",
            "days_open": days_open, "received": rand_date(rng), "handler": person(rng), "requester_name": person(rng),
            "components": pick(rng, [["the OS kernel (GPL-2.0-only)", "a shell-utilities bundle (GPL-2.0-only)"],
                                     ["the OS kernel (GPL-2.0-only)", "a C library (LGPL-2.1-or-later)"],
                                     ["the OS kernel (GPL-2.0-only)", "a bootloader (GPL-2.0-or-later)",
                                      "a shell-utilities bundle (GPL-2.0-only)"]]),
        }

    def _avail(self, f):
        near = ("only a source tag near this version exists; the exact tree was not archived"
                + ("; it is unclear whether the chip supplier's later kernel patches are in it"
                   if f["supplier_patches_uncertain"] else ""))
        return {"exact_archived": "the exact build tree for this firmware version is archived",
                "near_tag_only": near,
                "supplier_bsp_missing": "the chip supplier's board-support kernel source was never received from them"}[f["source_availability"]]

    def _offer(self, f):
        return {"written_offer": "the product manual includes a written offer for source (valid three years from last shipment)",
                "source_on_website": "source was published on our website at launch, for an earlier firmware version",
                "none_found": "no written offer or source link can be found in the product materials"}[f["offer_with_product"]]

    def render(self, f, rng, surface):
        pub = {None: None, "forum_post": "the requester has posted about the delay on a user forum",
               "public_issue_tracker": "the request is tracked on a public issue page with many followers",
               "press_enquiry": "a technology journalist has asked about the request"}[f["public_attention"]]
        if surface == "json":
            return dumps(prune({
                "request": {"received": f["received"].isoformat(), "from": REQUESTERS[f["requester"]], "days_open": f["days_open"],
                            "device": f"{f['device']} ({f['model']})", "firmware": f["firmware"], "components_named": f["components"],
                            "public_attention": pub},
                "product_history": {"years_since_last_shipped": f["years_since_last_shipped"], "source_offer": self._offer(f)},
                "what_we_have": {"source": self._avail(f), "kernel_modified_by_us": f["kernel_modified"],
                                 "build_and_install_scripts_archived": f["build_scripts_archived"]},
            }))
        if surface == "request_ticket":
            rows = [("Received", fmt_date(f["received"], rng)), ("Requester", f"{f['requester_name']} -- {REQUESTERS[f['requester']]}"),
                    ("Device", f"{f['device']}, model {f['model']}, firmware {f['firmware']}"),
                    ("Components requested", "; ".join(f["components"])),
                    ("Last shipped", f"{f['years_since_last_shipped']} years ago"), ("Offer in product", self._offer(f)),
                    ("Source status", self._avail(f)), ("Kernel modified", "yes" if f["kernel_modified"] else "no"),
                    ("Build scripts archived", "yes" if f["build_scripts_archived"] else "no"),
                    ("Open for", f"{f['days_open']} days"), ("Public attention", pub)]
            return "SOURCE REQUEST -- compliance queue\n" + kv_lines(rows, rng)
        body = [f"Hi {f['handler'].split()[0]},", "",
                noisy(join_sentences([
                    f"We have a request from {REQUESTERS[f['requester']]} for the source of {', '.join(f['components'])} "
                    f"in the {f['model']} ({f['device']}), firmware {f['firmware']}.",
                    f"The model last shipped {f['years_since_last_shipped']} years ago; {self._offer(f)}.",
                    f"On our side, {self._avail(f)}" + (", and we modified the kernel." if f["kernel_modified"] else "."),
                    ("Build scripts are archived." if f["build_scripts_archived"] else "Build scripts were not archived."),
                    f"The request has been open {f['days_open']} days.", (pub[0].upper() + pub[1:] + "." if pub else "")]), rng, 0.1),
                "", "Thanks,", person(rng)]
        return f"Subject: GPL source request -- {f['model']}\n\n" + "\n".join(body)

    def questions(self):
        return {
            "source_request_response": choice("How should the company answer this source request?", {
                "decline_not_entitled": "Decline politely; the request falls outside what the licences oblige the company to supply.",
                "fulfil_promptly": "Send the complete corresponding source for the named components now.",
                "rebuild_then_fulfil": "Reconstruct the exact build first, then send complete source, telling the requester when.",
                "send_available_and_escalate": "Send what exists now and escalate the missing parts to legal and the supplier.",
            }),
            "requester_entitled": noul(
                "The requester is entitled under the licences to receive the source they are asking for.",
                yes="The offer or licence terms extend to this person and the time limit has not run out.",
                no="The requester falls outside the offer's reach, or the obligation has lapsed."),
            "compliance_gap": score("How far is the company from being able to supply complete corresponding source?", [
                "No gap: the exact source and scripts are archived.",
                "Small: some reconstruction or packaging is needed.",
                "Large: the exact source or the build scripts must be recreated.",
                "Critical: essential parts are missing and depend on a third party.",
            ]),
            "public_exposure": score("How much reputational or enforcement exposure does this request carry?", [
                "Little: a private, routine request.",
                "Some: a delay could draw attention.",
                "Considerable: the matter is already public or overdue.",
                "High: public, long overdue, and with no offer or source on record.",
            ]),
            "supplier_input_needed": noul("The company cannot complete the request without material from its chip or board supplier."),
        }


WORKFLOW = Workflow(
    name="oss_license_compliance",
    title="Open-source licence compliance",
    description="How a company's open-source program office should handle third-party code: dependency intake, release "
                "scans, outside contributions, upstream relicensing and source requests for shipped devices.",
    domain="software legal / open-source program office",
    schemas=[DependencyIntake(), ReleaseScanGate(), InboundContribution(), UpstreamRelicensing(), SourceCodeRequest()],
    notes="Packages, products and people are fictional; licence identifiers are public SPDX names and the policy tiers "
          "are invented. No security-alert or vulnerability triage questions (security fixes appear only as context "
          "for a licensing decision).",
)
