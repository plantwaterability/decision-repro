"""App store submission review: should a mobile app build be approved, sent back for changes, or rejected?

Each case is one submission in a fictional app marketplace's review queue: an app whose privacy disclosures
may not match what its SDKs collect, a purchase flow that may route around the store's billing rules, an app
submitted to the kids category, an app that may be a thin website wrapper or a template clone, or an update
that adds a sensitive permission and asks for expedited review. The state carries what a reviewer sees:
submission metadata, automated binary and network scans, the reviewer's device notes, the developer's notes and
the relevant guideline text.

Everything is fictional: the store, its guideline numbers ("G5.1" and so on), the SDK vendors, the apps and the
developers. The guidelines paraphrase common app-marketplace policy themes in original wording.

Structure follows it_change_approval.py: each schema draws a `regime` first and lets it steer the factors.
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
    ident,
    join_sentences,
    kv_lines,
    maybe,
    near,
    noisy,
    noul,
    org,
    person,
    pick,
    pick_w,
    prune,
    rand_date,
    rand_ts,
    regime,
    score,
    show,
    table,
)

APP_A = ["Pantry", "Trail", "Coin", "Lumen", "Tide", "Pocket", "Nimbus", "Sprout", "Pixel", "Quill", "Echo",
         "Maple", "Orbit", "Kettle", "Harbor", "Fable", "Cinder", "Moss", "Juno", "Otter"]
APP_B = ["Pal", "Meter", "Nest", "Go", "Hub", "Buddy", "Box", "Kit", "Loop", "Wise", "Mate", "ly", "Spark", "Path"]
STUDIO = ["Labs", "Studios", "Apps", "Software", "Digital", "Interactive"]


def _app_name(rng) -> str:
    return f"{pick(rng, APP_A)}{pick(rng, APP_B)}"


def _developer(rng) -> str:
    return f"{org(rng).split()[0]} {pick(rng, STUDIO)}"


def _bundle(dev: str, app: str) -> str:
    return f"com.{dev.split()[0].lower()}.{app.lower()}"


def _and_list(items) -> str:
    items = list(items)
    if not items:
        return "nothing"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


GUIDE = {
    "G2.1": "Submissions must be complete: no crashes, placeholder content or features the reviewer cannot reach; "
            "provide working demo credentials for anything behind a login.",
    "G2.4": "An app must offer more than a repackaged website: useful native functionality or a clearly better "
            "experience than the browser.",
    "G3.1": "Digital content, features and subscriptions used in the app must be sold through store billing. "
            "Exceptions: physical goods and services, real-time one-to-one services between people, "
            "organisation-purchased licences, and reader apps (one approved account-management link).",
    "G3.2": "Before purchase, show the price, billing period, renewal terms and any trial-to-paid conversion "
            "clearly; provide a way to restore purchases.",
    "G4.3": "Do not publish many near-identical apps; template-based apps must be submitted by, or on behalf of, "
            "the business they represent and must differ in substance.",
    "G5.1": "The privacy disclosure must list all data collected by the app and by every SDK in it.",
    "G5.2": "Tracking users across other companies' apps or websites requires the platform consent prompt, and "
            "no tracking identifier may be accessed before the user agrees.",
    "G5.3": "Request a permission only when a feature needs it, with prompt text that says specifically why.",
    "G5.4": "Apps that let users create accounts must let them delete their account from within the app.",
    "G1.3": "Kids-category apps: no behavioural advertising, only third-party SDKs approved for children's "
            "apps, and a parental gate before purchases, outside links or settings.",
}


# =========================================================================== schema 1
APP_KINDS = {
    "fitness tracker": ["email", "health and fitness data", "precise location"],
    "recipe planner": ["email", "purchase history"],
    "budgeting app": ["email", "financial account data"],
    "photo editor": ["photos", "email"],
    "parking payments app": ["precise location", "payment info", "email"],
    "language-learning app": ["email", "voice recordings"],
    "local news reader": ["email", "reading history"],
    "meditation timer": ["email", "health and fitness data"],
}
SDKS = {
    "AdVantage Mobile Ads": ("ads", ["device advertising ID", "coarse location", "product interaction"], True),
    "Metricly Analytics": ("analytics", ["device ID", "product interaction"], False),
    "CrashNest": ("crash reporting", ["crash diagnostics"], False),
    "Attributo": ("install attribution", ["device advertising ID", "install source"], True),
    "SocialSign": ("social login", ["name", "email"], False),
    "GeoPulse Insights": ("location analytics", ["precise location"], True),
}
SENSITIVE = ["health and fitness data", "precise location", "financial account data", "voice recordings",
             "payment info"]


class PrivacyDisclosureAudit(Schema):
    name = "privacy_disclosure_audit"
    description = "A new app build whose privacy disclosures, consent flow and account controls are checked against its scans."
    surfaces = {"json": 0.5, "scan_report": 0.25, "reviewer_worksheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, list(APP_KINDS))
        own = list(APP_KINDS[kind])
        sdks = rng.sample(list(SDKS), pick(rng, [1, 2, 2, 3]))
        omit_own = own[-1] if (len(own) > 1 and chance(rng, 0.3)) else None
        declared_sdk = pick_w(rng, {"all": 3, "some": 3, "none": 2})
        consent = pick_w(rng, {"prompt_before_init": 3, "sdk_init_before_prompt": 2, "no_prompt": 1,
                               "tracking_disabled_in_config": 2})
        accounts = chance(rng, 0.7)
        deletion = pick_w(rng, {"in_app": 3, "email_request_only": 2, "not_offered": 1}) if accounts else None
        policy = pick_w(rng, {"app_specific": 4, "generic_template": 2, "broken_link": 1})
        prior = pick(rng, [0, 0, 1, 2])
        if reg == "clear":
            if chance(rng, 0.5):
                sdks = rng.sample(["Metricly Analytics", "CrashNest", "SocialSign"], pick(rng, [1, 2]))
                omit_own, declared_sdk, consent, policy = None, "all", "prompt_before_init", "app_specific"
                deletion = "in_app" if accounts else None
            else:
                sdks = ["AdVantage Mobile Ads", "Attributo", "GeoPulse Insights"]
                omit_own, declared_sdk, consent, policy = own[-1], "none", "no_prompt", "broken_link"
                accounts, deletion = True, "not_offered"
        elif reg == "borderline":
            sdks = rng.sample(["Metricly Analytics", "CrashNest", "SocialSign", "Attributo"], 2)
            declared_sdk = "some"
            consent = pick(rng, ["prompt_before_init", "tracking_disabled_in_config"])
            policy = pick(rng, ["app_specific", "generic_template"])
        else:
            if chance(rng, 0.5):
                # a flawless label, but the network capture shows an advertising ID leaving before the prompt
                sdks = ["AdVantage Mobile Ads"] + rng.sample(["CrashNest", "Metricly Analytics"], 1)
                omit_own, declared_sdk, consent, policy = None, "all", "sdk_init_before_prompt", "app_specific"
            else:
                # alarming SDK list, but tracking is switched off in config, the capture agrees, the label is full
                sdks = ["AdVantage Mobile Ads", "Attributo", "Metricly Analytics"]
                omit_own, declared_sdk, consent = None, "all", "tracking_disabled_in_config"
        sdk_data = []
        for s_ in sdks:
            for d in SDKS[s_][1]:
                if d not in sdk_data:
                    sdk_data.append(d)
        if declared_sdk == "all":
            declared_from_sdk = list(sdk_data)
        elif declared_sdk == "some":
            declared_from_sdk = sdk_data[: max(0, len(sdk_data) - pick(rng, [1, 2]))]
        else:
            declared_from_sdk = []
        declared = [d for d in own if d != omit_own] + [d for d in declared_from_sdk if d not in own]
        tracks = any(SDKS[s_][2] for s_ in sdks)
        if not tracks and consent in ("sdk_init_before_prompt", "no_prompt"):
            consent = "no_prompt"            # nothing tracks, so there is nothing to prompt for
        dev_note = pick_w(rng, {"none": 3, "sdk_config": 2, "label_later": 1})
        if dev_note == "sdk_config" and not tracks:
            dev_note = "none"
        if dev_note == "label_later" and all(d in declared for d in own + sdk_data):
            dev_note = "none"
        app, dev = _app_name(rng), _developer(rng)
        return {"regime": reg, "app": app, "developer": dev, "bundle": _bundle(dev, app), "app_kind": kind,
                "own_collection": own, "sdks": sdks, "sdk_collection": sdk_data, "declared": declared,
                "tracking_sdk_present": tracks, "consent_flow": consent, "has_accounts": accounts,
                "account_deletion": deletion, "privacy_policy": policy, "prior_rejections": prior,
                "version": f"{pick(rng, [1, 2, 3, 4])}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
                "submission": ident(rng, "SUB", 7, "-"), "reviewer": person(rng),
                "submitted": rand_ts(rng), "dev_note": dev_note}

    def _network(self, f):
        c = f["consent_flow"]
        if not f["tracking_sdk_present"]:
            return ["no requests carrying an advertising identifier observed"]
        if c == "sdk_init_before_prompt":
            return ["advertising ID sent to an ad/attribution endpoint 0.8 s after launch",
                    "consent prompt displayed 6 s after launch"]
        if c == "no_prompt":
            return ["advertising ID sent to ad/attribution endpoints at launch", "no consent prompt shown in session"]
        if c == "prompt_before_init":
            return ["consent prompt shown at first launch", "advertising ID requests only after the user allowed it"]
        return ["ad SDK initialised with personalised ads off", "no advertising ID observed in 20 min session"]

    DEL = {"in_app": "delete-account option under Settings > Account", "email_request_only":
           "deletion only by emailing support", "not_offered": "no account deletion found"}
    POL = {"app_specific": "loads; describes this app", "generic_template": "loads; generic template that "
           "does not name the app or its SDKs", "broken_link": "returns HTTP 404"}
    LEAD = ("Review-team guidance: when several privacy issues apply, open the message with the first that applies "
            "in this order -- tracking before consent (G5.2), no in-app account deletion (G5.4), privacy policy "
            "link broken or generic, undisclosed collection (G5.1).")
    NOTE = {"none": None,
            "sdk_config": "Developer note: 'The ads SDK is configured for contextual ads only.'",
            "label_later": "Developer note: 'We will update the privacy details after launch.'"}

    def render(self, f, rng, surface):
        net = self._network(f)
        note = self.NOTE[f["dev_note"]]
        if surface == "json":
            return dumps(prune({
                "submission": {"id": f["submission"], "app": f["app"], "bundle_id": f["bundle"],
                               "developer": f["developer"], "version": f["version"], "category": f["app_kind"],
                               "previous_rejections": f["prior_rejections"], "developer_note": note},
                "declared_privacy": {"data_collected": f["declared"]},
                "binary_scan": {"sdks": [f"{s_} ({SDKS[s_][0]})" for s_ in f["sdks"]],
                                "data_the_sdks_collect": f["sdk_collection"],
                                "data_the_app_itself_collects": f["own_collection"]},
                "network_capture": net,
                "account_controls": {"accounts": f["has_accounts"],
                                     "deletion": self.DEL[f["account_deletion"]] if f["account_deletion"] else None},
                "privacy_policy_url": self.POL[f["privacy_policy"]],
                "guidelines": {k: GUIDE[k] for k in ("G5.1", "G5.2", "G5.4")},
                "message_guidance": self.LEAD,
            }))
        if surface == "scan_report":
            rows = [[s_, SDKS[s_][0], ", ".join(SDKS[s_][1])] for s_ in f["sdks"]]
            return "\n".join([
                f"AUTOMATED SCAN -- {f['app']} {f['version']} ({f['bundle']}), {f['submission']}",
                f"Uploaded {fmt_ts(f['submitted'], rng)} by {f['developer']}", "",
                "Third-party SDKs detected:", table(["SDK", "type", "collects"], rows, rng), "",
                "Declared in privacy disclosure: " + _and_list(f["declared"]),
                "App's own collection (from API usage): " + _and_list(f["own_collection"]), "",
                "Network capture (first session):", bullets(net, rng), "",
                kv_lines([("Accounts", "yes" if f["has_accounts"] else "no"),
                          ("Account deletion", self.DEL[f["account_deletion"]] if f["account_deletion"] else None),
                          ("Privacy policy URL", self.POL[f["privacy_policy"]]),
                          ("Previous rejections", f["prior_rejections"])], rng)]
                + ([note] if note else []) + ["", self.LEAD])
        obs = [f"Category {f['app_kind']}. SDKs: {', '.join(f['sdks'])}."]
        undeclared = [d for d in f["own_collection"] + f["sdk_collection"] if d not in f["declared"]]
        obs.append("Everything collected appears in the disclosure." if not undeclared else
                   "Not in the disclosure: " + ", ".join(dict.fromkeys(undeclared)) + ".")
        obs += [n[0].upper() + n[1:] + "." for n in net]
        if f["has_accounts"]:
            obs.append("Account deletion: " + self.DEL[f["account_deletion"]] + ".")
        obs.append("Privacy policy link " + self.POL[f["privacy_policy"]] + ".")
        return (f"REVIEW WORKSHEET {f['submission']} -- {f['app']} v{f['version']}\nReviewer: {f['reviewer']}\n"
                f"Prior rejections: {f['prior_rejections']}\n\n" + bullets(obs, rng)
                + (f"\n\n{note}" if note else "") + "\n\nG5.1 / G5.2 / G5.4 apply. " + self.LEAD)

    def questions(self):
        return {
            "review_verdict": choice("What should the reviewer do with this build?", {
                "approve": "Approve the build as submitted.",
                "approve_with_notice": "Approve this version and tell the developer about a gap to fix in the "
                                       "next update.",
                "correct_disclosures_only": "Hold approval until the developer corrects the privacy disclosure, "
                                            "which needs no new build.",
                "reject_and_refer_account": "Reject and refer the developer account for a compliance review.",
                "require_new_build": "Reject this build; the code or SDK set-up has to change before "
                                     "resubmission.",
            }),
            "lead_issue": choice("Which issue should lead the reviewer's message to the developer?", {
                "account_deletion": "Users who can create an account cannot delete it from within the app.",
                "disclosure_accuracy": "The privacy disclosure does not match what the app and its SDKs collect.",
                "no_privacy_issue": "There is no privacy problem worth raising.",
                "privacy_policy_link": "The privacy policy link is broken or does not describe this app.",
                "tracking_consent": "The app tracks users before they agree through the platform prompt.",
            }),
            "disclosure_gap": score("How far does the privacy disclosure fall short of what is actually "
                                    "collected?", [
                "No gap: everything collected is declared.",
                "Small: a low-sensitivity item, such as diagnostics, is missing.",
                "Significant: identifiers or usage data shared with third parties are missing.",
                "Severe: sensitive data or tracking data are left out of the disclosure.",
                "Near-total: the disclosure omits most of what is collected, tracking and sensitive data "
                "included.",
            ]),
            "user_data_harm": score("How much could users be harmed by the collection that is undisclosed or "
                                    "not consented to?", [
                "Negligible: nothing sensitive or cross-company is involved.",
                "Limited: routine usage data that stays with the developer.",
                "Material: identifiers or behaviour shared with advertising or analytics companies.",
                "Serious: sensitive data such as health, finances or precise location reaching third parties.",
            ]),
            "tracks_before_consent": noul(
                "The app tracks users across other companies' apps or sites before they have agreed to it.",
                yes="A tracking identifier is read or sent before the platform prompt, or no prompt is shown "
                    "while tracking happens.",
                no="Nothing tracks, tracking is switched off, or it starts only after the user agrees."),
        }


# =========================================================================== schema 2
OFFERS = {
    "video streaming": ("premium streaming subscription", "reader"),
    "e-book reader": ("e-book purchases", "reader"),
    "live tutoring": ("1:1 live tutoring sessions", "person_to_person"),
    "recorded course platform": ("recorded video courses", "none"),
    "meal-kit delivery": ("weekly meal-kit boxes", "physical"),
    "puzzle game": ("coin packs and an ad-free upgrade", "none"),
    "team task manager": ("business seats paid for by employers", "enterprise"),
    "dating app": ("premium membership", "none"),
    "personal training": ("live 1:1 video sessions with a trainer", "person_to_person"),
    "workout library": ("on-demand workout subscription", "none"),
}
EXEMPT_TEXT = {"reader": "reader app", "person_to_person": "real-time one-to-one service",
               "physical": "physical goods / services", "enterprise": "organisation-purchased licences",
               "none": "none claimed"}
PATHS = {"store_billing_only": "store billing only",
         "external_link_button": "a 'Subscribe on our website' button that opens the browser",
         "in_app_web_checkout": "an in-app web checkout page taking card details",
         "text_mention_only": "a line of text saying plans are available on the website (no link)",
         "store_billing_plus_web_price": "store billing, plus a banner saying the web price is 30% cheaper"}


class DigitalPurchaseRules(Schema):
    name = "digital_purchase_rules"
    description = "An app's purchase flow checked against the store's billing rule, its exemptions and subscription disclosures."
    surfaces = {"json": 0.5, "resolution_thread": 0.25, "reviewer_checklist": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        app_type = pick(rng, list(OFFERS))
        sells, true_basis = OFFERS[app_type]
        claimed = true_basis if chance(rng, 0.6) else pick(rng, ["reader", "person_to_person", "enterprise", "none"])
        path = pick_w(rng, {"store_billing_only": 3, "external_link_button": 2, "in_app_web_checkout": 2,
                            "text_mention_only": 1, "store_billing_plus_web_price": 1})
        mixed = None
        if true_basis == "person_to_person" and chance(rng, 0.4):
            mixed = "recorded replays and group classes sold through the same checkout"
        entitlement = chance(rng, 0.5) if true_basis == "reader" else False
        trial = pick_w(rng, {"clear": 3, "small_print": 2, "not_shown": 1, None: 2})
        price_shown = chance(rng, 0.8)
        restore = chance(rng, 0.75)
        mismatch = chance(rng, 0.15)
        complaints = pick(rng, [0, 0, 2, 7, 19])
        precedent = chance(rng, 0.25)
        if reg == "clear":
            if chance(rng, 0.5):
                app_type = pick(rng, ["meal-kit delivery", "team task manager", "live tutoring"])
                sells, true_basis = OFFERS[app_type]
                claimed, mixed, path = true_basis, None, pick(rng, ["in_app_web_checkout", "external_link_button"])
                trial, price_shown, restore, mismatch, complaints = None, True, True, False, 0
            else:
                app_type = pick(rng, ["puzzle game", "dating app", "workout library"])
                sells, true_basis = OFFERS[app_type]
                claimed, path, trial, price_shown, mismatch, complaints = "none", "in_app_web_checkout", \
                    "not_shown", False, True, 19
        elif reg == "borderline":
            app_type = pick(rng, ["video streaming", "e-book reader", "live tutoring", "personal training",
                                  "team task manager"])
            sells, true_basis = OFFERS[app_type]
            claimed = true_basis
            path = pick(rng, ["external_link_button", "text_mention_only", "store_billing_plus_web_price"])
            if true_basis in ("live tutoring", "person_to_person"):
                mixed = mixed or (None if chance(rng, 0.5) else "recorded replays sold through the same checkout")
            trial = pick(rng, ["small_print", "clear"])
        else:
            if chance(rng, 0.5):
                # the store approved this same flow before, but it steers digital sales to the web
                app_type = pick(rng, ["recorded course platform", "workout library", "dating app"])
                sells, true_basis = OFFERS[app_type]
                claimed, path, precedent = pick(rng, ["person_to_person", "reader"]), "external_link_button", True
                complaints = 0
            else:
                # a properly exempt physical-goods app whose subscription screen hides the trial conversion
                app_type = "meal-kit delivery"
                sells, true_basis = OFFERS[app_type]
                claimed, path, trial, restore, complaints = "physical", "in_app_web_checkout", "small_print", \
                    False, pick(rng, [7, 19])
        if true_basis != "reader":
            entitlement = False
        if true_basis != "person_to_person":
            mixed = None
        app, dev = _app_name(rng), _developer(rng)
        return {"regime": reg, "app": app, "developer": dev, "bundle": _bundle(dev, app), "app_type": app_type,
                "sells": sells, "true_exemption": true_basis, "claimed_exemption": claimed, "purchase_path": path,
                "also_sells": mixed, "reader_link_entitlement": entitlement, "trial_disclosure": trial,
                "price_before_purchase": price_shown, "restore_button": restore, "price_mismatch": mismatch,
                "charge_complaints_30d": complaints, "same_flow_previously_approved": precedent,
                "monthly_price": pick(rng, [3.99, 7.99, 9.99, 14.99, 29.99]),
                "submission": ident(rng, "SUB", 7, "-"), "reviewer": person(rng), "date": rand_date(rng)}

    TRIAL = {"clear": "7-day free trial, then the price, stated above the button",
             "small_print": "trial converts to paid after 3 days; stated in grey small print below the button",
             "not_shown": "trial converts to paid; conversion not mentioned before purchase", None: None}

    def _price_line(self, f):
        base_ = f"${f['monthly_price']}/month"
        if f["price_mismatch"]:
            return f"screen shows {base_}; the payment sheet bills ${f['monthly_price']}/week"
        return base_ if f["price_before_purchase"] else "not on the screen, only on the payment sheet"

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "submission": {"id": f["submission"], "app": f["app"], "developer": f["developer"],
                               "bundle_id": f["bundle"], "app_type": f["app_type"]},
                "monetisation": {"sells": f["sells"], "also_sells": f["also_sells"],
                                 "purchase_path": PATHS[f["purchase_path"]],
                                 "exemption_claimed": EXEMPT_TEXT[f["claimed_exemption"]],
                                 "reader_link_entitlement_on_file": f["reader_link_entitlement"]
                                 if f["true_exemption"] == "reader" else None},
                "subscription_screen": {"price": self._price_line(f), "trial": self.TRIAL[f["trial_disclosure"]],
                                        "restore_purchases_button": f["restore_button"]},
                "signals": {"unexpected_charge_complaints_30d": f["charge_complaints_30d"],
                            "same_flow_approved_in_earlier_version": f["same_flow_previously_approved"] or None},
                "guidelines": {"G3.1": GUIDE["G3.1"], "G3.2": GUIDE["G3.2"]},
            }))
        if surface == "resolution_thread":
            t = rand_ts(rng, f["date"], hours=(8, 18))
            msgs = [("Developer", f"{f['app']} sells {f['sells']}. Payments go through "
                                  f"{PATHS[f['purchase_path']]}."
                     + (f" We also offer {f['also_sells']}." if f["also_sells"] else "")
                     + (f" We rely on the {EXEMPT_TEXT[f['claimed_exemption']]} exemption."
                        if f["claimed_exemption"] != "none" else "")
                     + (" This flow was approved in our last version." if f["same_flow_previously_approved"] else "")),
                    ("App Review", f"Thanks. Reviewing the subscription screen. Price: {self._price_line(f)}; "
                                   f"trial: {show(self.TRIAL[f['trial_disclosure']], rng)}; restore button "
                                   f"{'present' if f['restore_button'] else 'not found'}.")]
            if f["true_exemption"] == "reader":
                msgs.append(("App Review", "Reader-link entitlement on file: "
                             + ("yes." if f["reader_link_entitlement"] else "no.")))
            if f["charge_complaints_30d"]:
                msgs.append(("App Review", f"We also see {f['charge_complaints_30d']} store reviews in 30 days "
                                           "about unexpected charges."))
            out = []
            for who, text in msgs:
                t += timedelta(hours=rng.randint(1, 20))
                out.append(f"{who} -- {fmt_ts(t, rng)}\n{noisy(text, rng, 0.1)}")
            return f"Resolution Center -- {f['submission']} ({f['app']}, {f['app_type']})\n\n" + "\n\n".join(out)
        rows = [("Submission", f["submission"]), ("App", f"{f['app']} ({f['app_type']})"),
                ("Sells", f["sells"] + (f"; also {f['also_sells']}" if f["also_sells"] else "")),
                ("Purchase path", PATHS[f["purchase_path"]]),
                ("Exemption claimed", EXEMPT_TEXT[f["claimed_exemption"]]),
                ("Reader-link entitlement", ("yes" if f["reader_link_entitlement"] else "no")
                 if f["true_exemption"] == "reader" else None),
                ("Price before purchase", self._price_line(f)), ("Trial terms", self.TRIAL[f["trial_disclosure"]]),
                ("Restore purchases", "yes" if f["restore_button"] else "no"),
                ("Charge complaints (30 d)", f["charge_complaints_30d"]),
                ("Same flow approved before", "yes" if f["same_flow_previously_approved"] else None)]
        return (f"PAYMENTS CHECKLIST -- reviewer {f['reviewer']}, {fmt_date(f['date'], rng)}\n"
                + kv_lines(rows, rng) + "\n\nG3.1: " + GUIDE["G3.1"])

    def questions(self):
        return {
            "purchase_flow_verdict": choice("What should the reviewer decide about the app's purchase flow?", {
                "approve_as_submitted": "Approve; the purchase flow complies as it stands.",
                "fix_subscription_disclosures": "Return the build until price, renewal and trial terms are shown "
                                                "clearly before purchase.",
                "move_sales_to_store_billing": "Return the build until digital items are sold through store "
                                               "billing and outside payment links are removed.",
                "refer_to_payments_policy": "Send it to the payments-policy team, because whether an exemption "
                                            "applies is genuinely unclear.",
            }),
            "exemption_covers_sales": noul(
                "An exemption from the store-billing rule covers everything this app sells outside store billing.",
                yes="Everything sold outside store billing is physical, a real-time one-to-one service, bought by "
                    "organisations, or an entitled reader-app link.",
                no="At least some digital content or features are sold outside store billing without an "
                   "applicable exemption."),
            "steering_level": score("How strongly does the app push users towards paying outside the store?", [
                "Not at all: store billing is the only route.",
                "Faintly: a passing mention with no link.",
                "Clearly: a prominent link, button or price comparison.",
                "Fully: payment is taken in the app outside store billing.",
            ]),
            "terms_clarity": score("How clearly does the app tell users what they will pay, and when, before they "
                                   "commit?", [
                "Misleading: the charge differs from what is shown or is hidden.",
                "Unclear: key terms are buried in small print or missing.",
                "Mostly clear: one detail is weak, such as a missing restore option.",
                "Fully clear: price, period, renewal and trial terms are plain.",
            ]),
            "unexpected_charges_likely": noul("Users are likely to be charged in ways they did not expect."),
        }


# =========================================================================== schema 3
KID_APPS = ["alphabet tracing game", "drawing and colouring app", "bedtime story reader", "early maths practice",
            "animal sounds quiz", "music-making toy"]
GATES = {"none": "no gate", "hold_button": "press and hold a button for 3 seconds",
         "simple_sum": "answer '4 + 3'", "adult_task": "type the year written in words, e.g. 'two thousand ten'",
         "platform_gate": "the platform's parental-control check"}
KID_SDKS = {"TinyMetrics (approved for kids apps)": True, "Metricly Analytics": False, "CrashNest": True,
            "AdVantage Mobile Ads": False, "KidSafe Contextual Ads (approved)": True}


class KidsCategorySubmission(Schema):
    name = "kids_category_submission"
    description = "An app submitted to the kids category, checked for parental gates, third-party SDKs and age fit."
    surfaces = {"json": 0.5, "reviewer_device_notes": 0.25, "submission_form": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, KID_APPS)
        band = pick(rng, ["5 and under", "6-8", "9-11"])
        sdks = rng.sample(list(KID_SDKS), pick(rng, [1, 2, 2, 3]))
        gate = pick_w(rng, {"none": 1, "hold_button": 2, "simple_sum": 2, "adult_task": 3, "platform_gate": 2})
        gated = rng.sample(["purchases", "outside links", "settings", "rate-us prompt"], pick(rng, [2, 3, 4]))
        ads = pick_w(rng, {"none": 4, "contextual": 2, "behavioural": 1})
        data = rng.sample(["first name", "voice recordings", "drawings uploaded to cloud", "device ID"],
                          pick(rng, [0, 1, 2]))
        text_heavy = chance(rng, 0.3)
        scary = chance(rng, 0.15)
        iap = chance(rng, 0.5)
        if reg == "clear":
            if chance(rng, 0.5):
                sdks, gate, gated, ads, data = ["TinyMetrics (approved for kids apps)"], "adult_task", \
                    ["purchases", "outside links", "settings", "rate-us prompt"], "none", []
                text_heavy, scary = False, False
            else:
                sdks, gate, gated, ads = ["AdVantage Mobile Ads", "Metricly Analytics"], "hold_button", \
                    ["settings"], "behavioural"
                data = ["device ID", "voice recordings"]
        elif reg == "borderline":
            gate = pick(rng, ["simple_sum", "hold_button", "adult_task"])
            sdks = rng.sample(["TinyMetrics (approved for kids apps)", "CrashNest", "Metricly Analytics",
                               "KidSafe Contextual Ads (approved)"], 2)
            ads = "contextual" if "KidSafe Contextual Ads (approved)" in sdks else "none"
            gated = rng.sample(["purchases", "outside links", "settings", "rate-us prompt"], 3)
        else:
            if chance(rng, 0.5):
                # a strong gate everywhere, but one unapproved analytics SDK receives a device ID
                gate, gated, ads = "platform_gate", ["purchases", "outside links", "settings", "rate-us prompt"], "none"
                sdks, data = ["Metricly Analytics", "CrashNest"], ["device ID"]
            else:
                # clean SDKs, but the gate is a sum a six-year-old can do and the content skews old
                sdks, ads, gate = ["TinyMetrics (approved for kids apps)"], "none", "simple_sum"
                band, text_heavy = "5 and under", True
        if ads == "behavioural" and "AdVantage Mobile Ads" not in sdks:
            sdks = sdks + ["AdVantage Mobile Ads"]
        if ads == "contextual" and not any("Ads" in s_ for s_ in sdks):
            sdks = sdks + ["KidSafe Contextual Ads (approved)"]
        if ads == "none":
            sdks = [s_ for s_ in sdks if "Ads" not in s_] or ["CrashNest"]
        app, dev = _app_name(rng), _developer(rng)
        return {"regime": reg, "app": app, "developer": dev, "bundle": _bundle(dev, app), "app_kind": kind,
                "age_band": band, "sdks": sdks, "gate": gate, "gated_items": gated, "ads": ads,
                "data_collected": data, "text_heavy": text_heavy, "frightening_scenes": scary,
                "in_app_purchases": iap, "submission": ident(rng, "SUB", 7, "-"), "reviewer": person(rng),
                "date": rand_date(rng), "device": pick(rng, ["tablet, OS 18", "phone, OS 17", "tablet, OS 17"])}

    def _ungated(self, f):
        items = ["purchases", "outside links", "settings", "rate-us prompt"]
        if not f["in_app_purchases"]:
            items.remove("purchases")
        return [x for x in items if x not in f["gated_items"]]

    def _content(self, f):
        c = []
        if f["text_heavy"]:
            c.append("menus and instructions are mostly written text")
        if f["frightening_scenes"]:
            c.append("two stories include dark, frightening scenes")
        return c or ["content suits young children"]

    def render(self, f, rng, surface):
        ung = self._ungated(f)
        sdk_rows = [f"{s_}" for s_ in f["sdks"]]
        if surface == "json":
            return dumps(prune({
                "submission": {"id": f["submission"], "app": f["app"], "developer": f["developer"],
                               "bundle_id": f["bundle"], "listing": f"kids category, ages {f['age_band']}",
                               "app_type": f["app_kind"], "in_app_purchases": f["in_app_purchases"]},
                "sdks_detected": sdk_rows, "advertising": f["ads"],
                "data_sent_off_device": f["data_collected"] or None,
                "parental_gate": {"mechanism": GATES[f["gate"]], "protects": [x for x in f["gated_items"]
                                                                              if x != "purchases" or f["in_app_purchases"]],
                                  "not_protected": ung or None},
                "content_review": self._content(f),
                "guideline": {"G1.3": GUIDE["G1.3"]},
            }))
        if surface == "reviewer_device_notes":
            s = [f"Installed {f['app']} on {f['device']}; listed for ages {f['age_band']} ({f['app_kind']})."]
            s.append(f"Parental gate: {GATES[f['gate']]}.")
            s.append("Gate appears before " + _and_list([x for x in f["gated_items"]
                                                         if x != "purchases" or f["in_app_purchases"]]) + "."
                     if f["gate"] != "none" else "No gate anywhere.")
            if ung and f["gate"] != "none":
                s.append("Reached " + _and_list(ung) + " without any gate.")
            s.append("SDKs from scan: " + ", ".join(f["sdks"]) + ".")
            s.append({"none": "No ads seen.", "contextual": "Saw contextual ads between levels.",
                      "behavioural": "Ads look personalised; ad requests carry the device ID."}[f["ads"]])
            if f["data_collected"]:
                s.append("Sent off device: " + ", ".join(f["data_collected"]) + ".")
            s.append("Content: " + "; ".join(self._content(f)) + ".")
            return f"Device notes -- {f['reviewer']}, {fmt_date(f['date'], rng)}, {f['submission']}\n\n" + bullets(
                [noisy(x, rng, 0.1) for x in s], rng)
        rows = [("App", f["app"]), ("Developer", f["developer"]), ("Category", "Kids"),
                ("Age band", f["age_band"]), ("Type", f["app_kind"]),
                ("In-app purchases", "yes" if f["in_app_purchases"] else "no"),
                ("Third-party SDKs (declared)", ", ".join(f["sdks"])), ("Advertising", f["ads"]),
                ("Data collected", ", ".join(f["data_collected"]) or "none declared"),
                ("Parental gate", GATES[f["gate"]])]
        return ("KIDS CATEGORY SUBMISSION FORM\n" + kv_lines(rows, rng) + "\n\nReviewer additions:\n"
                + bullets(["Ungated: " + (_and_list(ung) if ung else "nothing found")] + self._content(f), rng))

    def questions(self):
        return {
            "kids_listing_decision": choice("What should happen to this kids-category submission?", {
                "approve_in_kids_category": "Approve it for the kids category at the age band requested.",
                "move_out_of_kids_category": "Approve it only outside the kids category, with an ordinary age "
                                             "rating.",
                "reject_submission": "Reject it; the app is not acceptable for children in its current form.",
                "return_for_fixes": "Return it for specific fixes, such as a stronger gate or removing an SDK, then "
                                    "re-review for the kids category.",
            }),
            "child_privacy_exposure": score("How exposed is children's data in this app?", [
                "Not exposed: nothing about the child leaves the device except approved diagnostics.",
                "Slightly: approved SDKs only, with minimal data.",
                "Considerably: identifiers or content reach an SDK not approved for children.",
                "Severely: children's identifiers feed personalised advertising or data brokers.",
            ]),
            "age_band_fit": score("How well do the content and design suit the age band the app is listed for?", [
                "Poorly: children in this band could not use it, or it is unsuitable for them.",
                "Weakly: parts would confuse or upset children of this age.",
                "Reasonably: a few elements are pitched slightly old.",
                "Closely: made for exactly this age group.",
            ]),
            "gate_blocks_children": noul(
                "Every purchase, outside link and settings screen sits behind a gate a young child could not pass.",
                yes="A task needing adult knowledge, or the platform's own check, guards every such screen.",
                no="Some screen is ungated, or the gate is a hold or a sum a child could manage."),
            "unapproved_third_party": noul(
                "Data about children goes to a third party not approved for children's apps."),
        }


# =========================================================================== schema 4
BUSINESSES = ["dental clinic booking", "community sports club", "real-estate agency listings", "local newspaper",
              "conference event guide", "gym class booking", "restaurant ordering", "boutique hotel concierge"]


class MinimumFunctionalityReview(Schema):
    name = "minimum_functionality_review"
    description = "A submission that may be a thin website wrapper, a template clone, incomplete or impossible to review."
    surfaces = {"json": 0.5, "review_log": 0.25, "reviewer_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        biz = pick(rng, BUSINESSES)
        web_share = pick(rng, [20, 45, 70, 90, 100])
        native = rng.sample(["push notifications", "offline saved items", "camera check-in", "calendar export",
                             "wallet pass", "home-screen widget"], pick(rng, [0, 1, 2, 3]))
        siblings = pick(rng, [0, 0, 3, 12, 40])
        sibling_same_ui = siblings > 0 and chance(rng, 0.6)
        submitter = pick_w(rng, {"business_account": 3, "agency_on_behalf": 2, "agency_unaffiliated": 1})
        crashes = pick(rng, [0, 0, 0, 1, 3])
        devices = pick(rng, [2, 3, 4])
        login = chance(rng, 0.6)
        demo = pick_w(rng, {"works": 3, "fails": 1, "not_provided": 1}) if login else None
        placeholder = chance(rng, 0.15)
        screenshots = pick_w(rng, {"match": 4, "show_missing_features": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                web_share, native, siblings, crashes, placeholder, screenshots = 20, ["push notifications",
                    "offline saved items", "wallet pass"], 0, 0, False, "match"
                demo = "works" if login else None
                submitter = "business_account"
            else:
                web_share, native, siblings, sibling_same_ui, submitter = 100, [], 40, True, "agency_unaffiliated"
                crashes, placeholder = 1, True
        elif reg == "borderline":
            web_share = pick(rng, [70, 90])
            native = rng.sample(["push notifications", "offline saved items", "calendar export"], pick(rng, [1, 2]))
            siblings = pick(rng, [0, 3, 12])
            sibling_same_ui = siblings > 0 and chance(rng, 0.5)
            crashes = pick(rng, [0, 1])
        else:
            if chance(rng, 0.5):
                # a rich native app that the reviewer could not get into
                web_share, native, siblings = 20, ["push notifications", "camera check-in", "home-screen widget"], 0
                login, demo = True, pick(rng, ["fails", "not_provided"])
            else:
                # a template agency's app, built for the business and submitted on its behalf, mostly web views
                siblings, sibling_same_ui, submitter, web_share = 40, True, "agency_on_behalf", 90
                native, crashes, placeholder = ["push notifications"], 0, False
        if submitter == "business_account":
            siblings = min(siblings, 3)          # a business account rarely publishes more than a handful of apps
        if siblings == 0:
            sibling_same_ui = False
        app, dev = _app_name(rng), (org(rng) if submitter == "business_account" else _developer(rng))
        return {"regime": reg, "app": app, "developer": dev, "bundle": _bundle(dev, app), "business_type": biz,
                "web_view_share_pct": web_share, "native_features": native, "developer_other_apps": siblings,
                "other_apps_share_ui": sibling_same_ui, "submitter": submitter,
                "crashes": crashes, "devices_tested": devices, "login_required": login, "demo_account": demo,
                "placeholder_content": placeholder, "screenshots": screenshots,
                "submission": ident(rng, "SUB", 7, "-"), "reviewer": person(rng), "start": rand_ts(rng),
                "resubmission": pick(rng, [1, 1, 2, 3])}

    def _facts(self, f):
        x = [f"{f['web_view_share_pct']}% of screens are web views of the business's site",
             "native features: " + (", ".join(f["native_features"]) if f["native_features"] else "none found")]
        if f["developer_other_apps"]:
            x.append(f"developer account has {f['developer_other_apps']} other live apps"
                     + (", same layout with different logos and colours" if f["other_apps_share_ui"] else
                        ", all different in design and purpose"))
        x.append({"business_account": "submitted by the business's own developer account",
                  "agency_on_behalf": "submitted by an app agency on the business's behalf (signed authorisation "
                                      "attached)",
                  "agency_unaffiliated": "submitted by an app agency account; nothing links it to the business"}[
            f["submitter"]])
        if f["crashes"]:
            x.append(f"crashed {f['crashes']} time(s) across {f['devices_tested']} test devices")
        if f["login_required"]:
            x.append({"works": "demo login works", "fails": "demo login rejected ('invalid credentials')",
                      "not_provided": "login required; no demo account in the review notes"}[f["demo_account"]])
        if f["placeholder_content"]:
            x.append("placeholder text ('Lorem ipsum') on the About screen")
        if f["screenshots"] == "show_missing_features":
            x.append("store screenshots show a loyalty card feature not present in the build")
        return x

    def render(self, f, rng, surface):
        facts = self._facts(f)
        if surface == "json":
            return dumps(prune({
                "submission": {"id": f["submission"], "app": f["app"], "bundle_id": f["bundle"],
                               "developer_account": f["developer"], "business": f["business_type"],
                               "attempt": f["resubmission"]},
                "build_analysis": {"web_view_screens_pct": f["web_view_share_pct"],
                                   "native_features": f["native_features"] or ["none"],
                                   "placeholder_content_found": f["placeholder_content"] or None},
                "developer_portfolio": {"other_live_apps": f["developer_other_apps"],
                                        "same_layout_as_other_apps": f["other_apps_share_ui"] or None,
                                        "submitted_by": {"business_account": "the business itself",
                                                         "agency_on_behalf": "agency, with the business's signed "
                                                                             "authorisation",
                                                         "agency_unaffiliated": "agency, no authorisation on file"}[
                                            f["submitter"]]},
                "testing": {"devices": f["devices_tested"], "crashes": f["crashes"],
                            "demo_account": f["demo_account"]},
                "listing": {"screenshots_match_build": f["screenshots"] == "match"},
                "guidelines": {k: GUIDE[k] for k in ("G2.1", "G2.4", "G4.3")},
            }))
        if surface == "review_log":
            t = f["start"]
            lines = [f"{fmt_ts(t, rng, 'log')}  review started: {f['submission']} {f['app']} (attempt "
                     f"{f['resubmission']})"]
            for x in facts:
                t += timedelta(minutes=rng.randint(2, 14))
                lines.append(f"{fmt_ts(t, rng, 'log')}  {x}")
            return "\n".join(lines) + f"\n\nApp: {f['business_type']}. Account: {f['developer']}."
        return (f"Reviewer notes -- {f['reviewer']}\n{f['app']} ({f['business_type']}), account {f['developer']}, "
                f"attempt {f['resubmission']}\n\n"
                + noisy(join_sentences(x[0].upper() + x[1:] + "." for x in facts), rng, 0.12)
                + "\n\nRelevant: G2.1 completeness, G2.4 minimum functionality, G4.3 template apps.")

    def questions(self):
        return {
            "submission_decision": choice("What should the reviewer do with this submission?", {
                "approve": "Approve it for the store.",
                "reject_app_concept": "Reject it because the app itself is not acceptable in the store, not "
                                      "because of fixable bugs.",
                "request_demo_access": "Put the review on hold and ask the developer for working access.",
                "return_for_fixes": "Send it back for specific fixes; the app can pass once they are made.",
            }),
            "main_problem": choice("What is the main problem with this submission, if any?", {
                "broken_or_unfinished": "It crashes or contains unfinished, placeholder parts.",
                "duplicate_of_other_apps": "It is one of many near-identical apps.",
                "no_blocking_problem": "Nothing that should stop approval.",
                "no_reviewer_access": "The reviewer could not reach the features that need checking.",
                "repackaged_website": "It adds little beyond the business's website.",
            }),
            "value_beyond_website": score("How much does the app offer beyond the business's website?", [
                "Nothing: it is the website in a frame.",
                "Little: one minor native touch.",
                "Some: a few genuinely useful native features.",
                "Substantial: a clearly better experience than the browser.",
            ]),
            "review_coverage": score("How much of the app could the reviewer actually exercise in this review?", [
                "Almost none: blocked by crashes or login.",
                "Part: key flows could not be reached.",
                "Most: a minor area could not be checked.",
                "All: every feature was reachable and tested.",
            ]),
            "template_clone": noul(
                "The app is substantially the same as other apps the developer account has already published.",
                yes="Other live apps from the same account share its layout and features, with only branding "
                    "changed.",
                no="The account has no other apps, or they differ in design and purpose."),
        }


# =========================================================================== schema 5
PERMISSIONS = {
    "background_location": ("location access 'Always'", "high"),
    "contacts": ("full contacts access", "high"),
    "microphone": ("microphone access", "medium"),
    "health_data": ("read access to health records (steps, heart rate)", "high"),
    "photo_library": ("full photo library access", "medium"),
    "bluetooth": ("Bluetooth access", "low"),
}
UPDATE_APPS = {"running tracker": ["background_location", "health_data", "bluetooth"],
               "parking app": ["background_location", "bluetooth"],
               "note-taking app": ["microphone", "photo_library", "contacts"],
               "smart-home controller": ["bluetooth", "background_location", "microphone"],
               "field-sales CRM": ["contacts", "background_location"],
               "photo backup app": ["photo_library", "background_location"]}
EXPEDITE = {"crash_fix": "a crash on launch affecting users on the latest OS version",
            "legal_deadline": "a legal requirement taking effect on a fixed date",
            "marketing_launch": "a marketing campaign launching this week",
            "partner_event": "a partner's live event where the feature will be demoed",
            None: None}


class PermissionUpdateExpedite(Schema):
    name = "permission_update_expedite"
    description = "An update to a live app that adds a sensitive permission, sometimes with an expedited-review request."
    surfaces = {"json": 0.5, "expedite_request": 0.25, "review_card": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, list(UPDATE_APPS))
        perm = pick(rng, UPDATE_APPS[kind])
        purpose_text = pick_w(rng, {"specific": 3, "generic": 3, "missing": 1})
        feature = pick_w(rng, {"visible": 4, "behind_flag": 2, "not_found": 1})
        data_flow = pick_w(rng, {"on_device": 3, "developer_servers": 3, "shared_with_partner": 1})
        installs = pick(rng, [8000, 60000, 400000, 2500000])
        strikes = pick(rng, [0, 0, 0, 1, 2])
        expedite = pick_w(rng, {None: 2, "crash_fix": 2, "legal_deadline": 1, "marketing_launch": 2,
                                "partner_event": 1})
        crash_share = maybe(rng, near(rng, 4, 3.5, lo=0.2, digits=1), 0.3) if expedite == "crash_fix" else None
        days = pick(rng, [1, 2, 4, 9]) if expedite else None
        if reg == "clear":
            if chance(rng, 0.5):
                purpose_text, feature, data_flow, strikes = "specific", "visible", "on_device", 0
                perm = pick(rng, [p for p in UPDATE_APPS[kind] if PERMISSIONS[p][1] != "high"] or UPDATE_APPS[kind])
                expedite, days, crash_share = "crash_fix", 1, 12.0
            else:
                perm = pick(rng, [p for p in UPDATE_APPS[kind] if PERMISSIONS[p][1] == "high"] or UPDATE_APPS[kind])
                purpose_text, feature, data_flow, strikes = "generic", "not_found", "shared_with_partner", 2
                expedite, days, crash_share = "marketing_launch", 4, None
        elif reg == "borderline":
            purpose_text = pick(rng, ["generic", "specific"])
            feature = pick(rng, ["behind_flag", "visible"])
            data_flow = pick(rng, ["developer_servers", "on_device"])
            if expedite == "crash_fix":
                crash_share = near(rng, 1.5, 1.0, lo=0.2, digits=1)
        else:
            if chance(rng, 0.5):
                # a genuine, urgent crash fix bundled with an unexplained high-sensitivity permission
                expedite, days, crash_share = "crash_fix", 1, near(rng, 9, 4, lo=3, digits=1)
                perm = pick(rng, [p for p in UPDATE_APPS[kind] if PERMISSIONS[p][1] == "high"] or UPDATE_APPS[kind])
                purpose_text, feature = "generic", pick(rng, ["behind_flag", "not_found"])
            else:
                # a well-explained, visible, on-device feature, pushed hard for a marketing date
                purpose_text, feature, data_flow = "specific", "visible", "on_device"
                expedite, days, strikes = "marketing_launch", 1, pick(rng, [1, 2])
        if expedite is None:
            days, crash_share = None, None
        app, dev = _app_name(rng), _developer(rng)
        return {"regime": reg, "app": app, "developer": dev, "bundle": _bundle(dev, app), "app_kind": kind,
                "permission": perm, "sensitivity": PERMISSIONS[perm][1], "purpose_text": purpose_text,
                "feature_visibility": feature, "data_flow": data_flow, "installs": installs,
                "policy_strikes_12m": strikes, "expedite_reason": expedite, "crash_affected_pct": crash_share,
                "deadline_days": days, "version": f"{rng.randint(2, 7)}.{rng.randint(0, 12)}.0",
                "submission": ident(rng, "SUB", 7, "-"), "reviewer": person(rng), "requester": person(rng),
                "date": rand_date(rng), "queue_days": pick(rng, [1, 2, 3])}

    def _prompt(self, f, rng):
        if f["purpose_text"] == "missing":
            return "(no purpose text; the system default is shown)"
        if f["purpose_text"] == "generic":
            return pick(rng, ["'We use this to improve your experience.'", "'Required for app functionality.'",
                              "'Allow access for better features.'"])
        return {"background_location": "'Records your route while a run is in progress, even with the screen off.'",
                "contacts": "'Lets you pick colleagues from your contacts when sharing a note.'",
                "microphone": "'Used only while you hold the record button to add a voice memo.'",
                "health_data": "'Reads heart rate to show effort zones during your workout.'",
                "photo_library": "'Backs up the photos you choose to your account.'",
                "bluetooth": "'Connects to your parking sensor or heart-rate strap.'"}[f["permission"]]

    FEAT = {"visible": "feature using it is visible and works in review",
            "behind_flag": "feature using it is behind a server flag; not reachable during review",
            "not_found": "reviewer found no feature that uses it"}
    FLOW = {"on_device": "processed on the device only", "developer_servers": "uploaded to the developer's servers",
            "shared_with_partner": "uploaded and shared with an analytics partner"}

    def render(self, f, rng, surface):
        perm_text = PERMISSIONS[f["permission"]][0]
        exp = EXPEDITE[f["expedite_reason"]]
        crash = (f"{f['crash_affected_pct']}% of sessions crash on the current version"
                 if f["crash_affected_pct"] is not None else None)
        if surface == "json":
            return dumps(prune({
                "update": {"submission": f["submission"], "app": f["app"], "bundle_id": f["bundle"],
                           "version": f["version"], "app_type": f["app_kind"], "active_installs": f["installs"]},
                "new_permission": {"access": perm_text, "prompt_text": self._prompt(f, rng),
                                   "feature": self.FEAT[f["feature_visibility"]], "data": self.FLOW[f["data_flow"]]},
                "developer": {"name": f["developer"], "policy_strikes_12m": f["policy_strikes_12m"]},
                "expedite_request": ({"reason": exp, "deadline_in_days": f["deadline_days"], "evidence": crash}
                                     if exp else None),
                "queue": {"normal_wait_days": f["queue_days"]},
                "guideline": {"G5.3": GUIDE["G5.3"]},
            }))
        if surface == "expedite_request":
            if exp:
                body = [f"Subject: Expedited review request -- {f['app']} {f['version']} ({f['submission']})", "",
                        noisy(join_sentences([f"We are asking for an expedited review because of {exp}.",
                                              crash + "." if crash else "",
                                              f"We need it live within {f['deadline_days']} day(s).",
                                              f"This version also adds {perm_text} ({self._prompt(f, rng)})."]),
                              rng, 0.12), "", f"-- {f['requester']}, {f['developer']}"]
            else:
                body = [f"Subject: Submission notes -- {f['app']} {f['version']} ({f['submission']})", "",
                        f"This version adds {perm_text} with the prompt {self._prompt(f, rng)}.", "",
                        f"-- {f['requester']}, {f['developer']}"]
            tail = kv_lines([("Reviewer check", self.FEAT[f["feature_visibility"]]),
                             ("Data handling (scan)", self.FLOW[f["data_flow"]]),
                             ("Active installs", f"{f['installs']:,}"),
                             ("Policy strikes (12 months)", f["policy_strikes_12m"]),
                             ("Normal queue", f"{f['queue_days']} day(s)")], rng)
            return "\n".join(body) + "\n\n[review tool]\n" + tail
        rows = [["App", f"{f['app']} {f['version']} ({f['app_kind']})"], ["New access", perm_text],
                ["Prompt text", self._prompt(f, rng)], ["Feature", self.FEAT[f["feature_visibility"]]],
                ["Data", self.FLOW[f["data_flow"]]], ["Installs", f"{f['installs']:,}"],
                ["Strikes (12 mo)", str(f["policy_strikes_12m"])],
                ["Expedite", f"{exp} (needs it in {f['deadline_days']} d)" if exp else "not requested"],
                ["Crash data", show(crash, rng) if f["expedite_reason"] == "crash_fix" else "n/a"]]
        return f"REVIEW CARD {f['submission']} -- {f['developer']}\n" + table(["Field", "Value"], rows, rng)

    def questions(self):
        return {
            "update_verdict": choice("What should the reviewer do with this update?", {
                "approve_update": "Approve the update as submitted.",
                "fix_prompt_text": "Return it until the permission prompt explains specifically why the access "
                                   "is needed.",
                "reject_and_flag_account": "Reject it and flag the developer account for a policy review.",
                "remove_or_justify_permission": "Return it until the permission is removed or tied to a feature "
                                                "the reviewer can see working.",
            }),
            "expedite_priority": score("How strongly does the developer's situation justify reviewing this ahead "
                                       "of the normal queue?", [
                "Not at all: no request, or nothing is lost by waiting the usual days.",
                "Slightly: a business date, but no harm to existing users.",
                "Considerably: a fixed legal or partner date, or a defect hurting some users.",
                "Compellingly: many users are hitting a serious defect right now.",
            ]),
            "privacy_intrusiveness": score("How intrusive is the new access for users, given what it exposes and "
                                           "where the data goes?", [
                "Minimal: low-sensitivity access, kept on the device.",
                "Moderate: sensitive access used on the device, or mild access uploaded.",
                "High: sensitive access uploaded to the developer.",
                "Severe: sensitive data leaving the developer for third parties.",
            ]),
            "permission_justified": noul(
                "The new permission is justified by a feature users can see and use.",
                yes="A working, visible feature needs this exact access, and the prompt says so.",
                no="No reachable feature needs it, or the feature needs less access than requested."),
            "prompt_text_specific": noul("The permission prompt tells users specifically what the access is used "
                                         "for."),
        }


WORKFLOW = Workflow(
    name="app_store_submission_review",
    title="App store submission review",
    description="Whether a mobile app build in a fictional marketplace's review queue should be approved, sent "
                "back for changes or rejected, judged against the store's written guidelines.",
    domain="platform policy / app marketplace review",
    schemas=[PrivacyDisclosureAudit(), DigitalPurchaseRules(), KidsCategorySubmission(),
             MinimumFunctionalityReview(), PermissionUpdateExpedite()],
    notes="Pre-publication review of app builds against written store guidelines. Fictional store, guideline "
          "numbers, SDK vendors, apps and developers. No user-content moderation appeals, no support threads, "
          "no security-alert triage, no AI-agent runs.",
)
