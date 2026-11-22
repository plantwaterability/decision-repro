"""Content moderation appeals: a trust-and-safety queue ruling on user appeals against moderation actions.

Policy adjudication on a fictional platform. Should the action be upheld, reversed or modified? Which rule
applies, and how much harm could the content do? Does context such as satire, news reporting or
self-reference change the reading? Does the penalty fit the account's record, and does the case belong with a
specialist team? Objectionable content always appears as a reviewer's PARAPHRASE and is never reproduced.
The appellant's statement is evidence in the file, not a support conversation: there are no reply-drafting,
churn, security-alert or account-compromise questions.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting') and lets it steer the factors, as
in the reference module. Borderline cases sit near a classifier threshold, a strike window or a policy limit.
Conflicting cases pair a sympathetic account with damaging evidence, or the reverse.
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
    first_name,
    fmt_date,
    hexid,
    join_sentences,
    kv_lines,
    lognormal_amount,
    maybe,
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
    table,
)

# =========================================================================== shared helpers
PLATFORMS = ["Quillfeed", "Brambleweave", "Sparrowtalk", "Lanternboard", "Murmurly", "Hearthloop"]
_HANDLE_A = ["mossy", "velvet", "rusty", "lunar", "maple", "copper", "tinker", "quiet", "salty", "amber", "cinder",
             "brisk", "hollow", "saffron", "drifting", "pebble"]
_HANDLE_B = ["kettle", "otter", "fern", "harbor", "lantern", "sparrow", "pixel", "biscuit", "comet", "badger",
             "thistle", "gull", "pickle", "marlin", "cobble", "wren"]
_SHOP_B = ["Finds", "Trading Post", "Home & Garden", "Outdoors", "Wellness", "Curios", "Market Stall", "Emporium"]
_SHOP_A = ["Alder", "Basalt", "Driftwood", "Fernhill", "Juniper", "Lakeshore", "Oakridge", "Saltmarsh", "Yarrow",
           "Bluestem", "Elmstead", "Copperline"]


def _handle(rng):
    a, b = rng.choice(_HANDLE_A), rng.choice(_HANDLE_B)
    style = rng.randrange(4)
    if style == 0:
        return f"@{a}_{b}"
    if style == 1:
        return f"@{a}{b}{rng.randint(2, 99)}"
    if style == 2:
        return f"@{b}.{a}"
    return f"@{first_name(rng).lower()}_{b}"


def _count(n, rng):
    """A follower or view count as a moderation tool prints it."""
    if n >= 10000 and chance(rng, 0.4):
        return f"{n / 1000:.1f}k"
    return f"{n:,}"


def _audience(rng, median, sigma=1.2):
    return max(3, int(lognormal_amount(rng, median, sigma, 0)))


def _ds(rng):
    """One date style per rendered artefact, so dates inside a case agree with each other."""
    return rng.choice(["iso", "iso", "iso", "us", "long", "eu"])


def _appeal_id(rng):
    return pick(rng, [f"APL-{rng.randint(1000000, 9999999)}", f"AP{rng.randint(25, 27)}-{rng.randint(100000, 999999)}"])


# =========================================================================== schema 1
HARASS_RULES = {
    "targeted_harassment": ("3.1", "Bullying and harassment",
                            "Do not insult, demean or repeatedly target a specific person. Criticism of a public "
                            "figure's public work is allowed; attacks on their looks or private life are not."),
    "private_information": ("3.3", "Private information",
                            "Do not share details that would help others find or identify a private person "
                            "offline, such as the street they live on, their workplace or their vehicle."),
    "veiled_threat": ("3.4", "Threats and intimidation",
                      "Do not suggest that harm could come to someone, including 'warnings' or jokes that a "
                      "reasonable reader would find intimidating."),
    "brigading": ("3.6", "Brigading",
                  "Do not point your audience at a specific person or account so that they swarm, message or "
                  "mass-report them."),
}
HARASS_LADDER = ("Penalties rise with active harassment strikes: first strike, a warning; second, a 3-day reply "
                 "limit; third or later, a 7-day suspension. Strikes stay active for 90 days.")
HARASS_PENALTY = {1: "strike 1 (warning)", 2: "strike 2 + 3-day reply limit", 3: "strike 3 + 7-day suspension"}

# behaviour -> reviewer paraphrases (text, who it is aimed at), latent harm 0-3, the rule it most naturally
# engages, the explanations an author would plausibly offer, and the formats it turns up in
HARASS_BEHAVIOURS = {
    "heated_insult": {
        "say": [("calls another user a 'clueless clown' in an argument about a football transfer", "other_user"),
                ("tells another commenter under a cooking video that they are 'too dim to boil an egg'",
                 "other_user")],
        "harm": 1, "rule": "targeted_harassment", "claims": ["provoked", "joke_between_friends"],
        "formats": ["reply", "comment"]},
    "appearance_mockery": {
        "say": [("mocks how a named coworker dresses, with three laughing emojis", "private_person"),
                ("ridicules the way a named member of the author's running club runs", "private_person")],
        "harm": 2, "rule": "targeted_harassment", "claims": ["joke_between_friends", "satire"],
        "formats": ["post", "comment", "story"]},
    "neighbour_accusation": {
        "say": [("calls a named neighbour a thief and mentions the street he lives on", "private_person"),
                ("accuses a named local handyman of stealing tools and shows a photo of his van with the number "
                 "plate readable", "private_person")],
        "harm": 2, "rule": "private_information", "claims": ["public_warning"],
        "formats": ["post", "group post"]},
    "repeated_tagging": {
        "say": [("is the seventh in one evening to tag the same user and insist she 'owes everyone an apology'",
                 "other_user"),
                ("is the latest of several, across three threads, that follow one user and call her a liar",
                 "other_user")],
        "harm": 2, "rule": "targeted_harassment", "claims": ["seeking_answer", "provoked"],
        "formats": ["reply"]},
    "pile_on_call": {
        "say": [("shares a critic's handle and tells followers to 'go and let them know what we think'",
                 "other_user"),
                ("shows a screenshot of a small account's negative review with the line 'you know what to do'",
                 "other_user")],
        "harm": 2, "rule": "brigading", "claims": ["fair_comment", "public_warning"],
        "formats": ["post", "story"]},
    "veiled_warning": {
        "say": [("tells a town councillor she 'should be careful walking to her car these days'", "public_figure"),
                ("tells a cafe owner that 'people round here know where you park'", "private_person")],
        "harm": 3, "rule": "veiled_threat", "claims": ["figure_of_speech", "satire"],
        "formats": ["comment", "reply"]},
    "public_figure_jab": {
        "say": [("uses a meme to call a well-known TV chef a fraud and a hack", "public_figure"),
                ("calls a regional football manager 'the most useless man in the county'", "public_figure")],
        "harm": 0, "rule": "targeted_harassment", "claims": ["fair_comment", "satire"],
        "formats": ["post", "comment"]},
    "self_deprecation": {
        "say": [("has the author calling themselves a 'useless idiot' while tagging a second account", "self"),
                ("captions a photo of the author 'a walking disaster' and tags another account", "self")],
        "harm": 0, "rule": "targeted_harassment", "claims": ["self_reference"],
        "formats": ["post", "story"]},
    "ex_partner_callout": {
        "say": [("names an ex-partner, calls him a liar and a cheat, and attaches his photo", "private_person"),
                ("names an ex-partner and tells followers to 'watch out for her' on dating apps", "private_person")],
        "harm": 2, "rule": "targeted_harassment", "claims": ["public_warning", "provoked"],
        "formats": ["post", "story"]},
}

# claimed context -> what the appellant says, and what the thread shows (supports / undercuts / unclear)
HARASS_CLAIMS = {
    "joke_between_friends": {
        "say": ["We've been friends for years and talk to each other like this all the time. Nobody was upset.",
                "This is banter between mates. Ask them yourself, they thought it was funny."],
        "ev": {"supports": "The person the post is about replied with a laughing emoji and a jab of their own.",
               "undercuts": "The report came from the person the post is about, who has since blocked the author.",
               "unclear": "The person the post is about has not reacted to it or to the removal.",
               "partly": "The person the post is about follows the author and likes some of the author's other posts, but did not react to this one."}},
    "satire": {
        "say": ["It was obviously satire. Nobody who follows my page takes it literally.",
                "It's a parody account, the whole point is exaggeration."],
        "ev": {"supports": "The account bio reads 'parody - nothing here is meant seriously' and older posts "
                           "follow the same style.",
               "undercuts": "The account has no parody label and mostly posts straight local news.",
               "unclear": "The account bio is empty and there are only a handful of older posts.",
               "partly": "Several older posts on the account are plainly jokes, though the bio carries no parody label."}},
    "provoked": {
        "say": ["They insulted me first. I was answering what they said to me.",
                "Look at what was said to me before I replied. I only answered in kind."],
        "ev": {"supports": "The post being replied to contains an insult aimed at the author.",
               "undercuts": "The post being replied to is a polite question with nothing insulting in it.",
               "unclear": "The post being replied to has been deleted, so what came before is not visible.",
               "partly": "Earlier in the thread the other user was curt with the author, though not openly insulting."}},
    "public_warning": {
        "say": ["I was warning people. Others deserve to know what this person has been doing.",
                "People have a right to know. I only described what happened to me."],
        "ev": {"supports": "Other users replied describing similar experiences with the same person, and the "
                           "author mentions a police reference number.",
               "undercuts": "No one else has raised the matter, and the author shared the post into three local "
                            "groups.",
               "unclear": "Replies are split between people backing the author and people calling it a private "
                          "dispute.",
               "partly": "One other user replied with a similar complaint about the same person, without details."}},
    "seeking_answer": {
        "say": ["I just wanted a straight answer to a fair question and she kept ignoring me.",
                "I was asking a legitimate question. That is not harassment."],
        "ev": {"supports": "The other user had publicly promised an answer and later thanked people for pressing "
                           "her on it.",
               "undercuts": "The other user asked the author twice to stop tagging her before she reported it.",
               "unclear": "The other user's side of the exchange is not visible to the reviewer.",
               "partly": "The other user had said she would 'look into it' and then went quiet."}},
    "fair_comment": {
        "say": ["Criticising someone's public work is fair game. I said nothing about their private life.",
                "Opinions about someone's public work are allowed, surely? People can make up their own minds."],
        "ev": {"supports": "The post sticks to the person's public work, and nothing in the replies goes beyond "
                           "disagreement.",
               "undercuts": "Replies under the post include insults about the person's family, and the author "
                            "liked several of them.",
               "unclear": "The post has few replies and nothing else shows how it was meant.",
               "partly": "Most replies stay on the person's public work, though one or two drift into personal remarks."}},
    "figure_of_speech": {
        "say": ["That car park is badly lit and people have been mugged there. I was telling them to be careful.",
                "It was a warning, not a threat. I would never hurt anyone."],
        "ev": {"supports": "Local news posts that week reported several robberies near the car park in question.",
               "undercuts": "The comment came after a long exchange in which the author accused the person of "
                            "ruining the neighbourhood.",
               "unclear": "There is no other context; the comment was posted on its own.",
               "partly": "Another resident in the same thread mentioned feeling unsafe in that car park at night."}},
    "self_reference": {
        "say": ["I was talking about MYSELF. The account I tagged is my own.",
                "It's a picture of me. You can't harass yourself."],
        "ev": {"supports": "The tagged account lists the author's main handle in its bio and posts the same photos.",
               "undercuts": "The tagged account belongs to a different person, who commented 'not funny' under "
                            "the post.",
               "unclear": "The tagged account is private, so the reviewer cannot see who runs it.",
               "partly": "The tagged account posts some of the same photos as the author, but its bio names no one."}},
}
REPORTERS = {"classifier": "automated harassment classifier", "person_targeted": "report by the person the post is about",
             "bystander": "report by another user", "trusted_flagger": "report by a trusted-flagger organisation"}


class HarassmentRemovalAppeal(Schema):
    name = "harassment_removal_appeal"
    description = "A post removed under the harassment rules and appealed by its author: uphold, soften, restore or refer."
    surfaces = {"json": 0.45, "appeal_form": 0.3, "queue_card": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        if reg == "clear":
            violating = chance(rng, 0.55)
            beh = pick(rng, ["neighbour_accusation", "pile_on_call", "veiled_warning", "ex_partner_callout"]
                       if violating else ["self_deprecation", "public_figure_jab"])
            support = "undercuts" if violating else "supports"
            prior = pick(rng, [1, 2]) if violating else 0
        elif reg == "borderline":
            beh = pick(rng, ["heated_insult", "appearance_mockery", "repeated_tagging", "public_figure_jab",
                             "veiled_warning", "pile_on_call", "ex_partner_callout", "neighbour_accusation"])
            support = pick_w(rng, {"unclear": 3, "supports": 1, "undercuts": 1})
            prior = pick(rng, [0, 0, 1, 1, 2])
        else:
            # the thread evidence cuts against the surface reading of the post
            beh = pick(rng, list(HARASS_BEHAVIOURS))
            support = ("supports" if HARASS_BEHAVIOURS[beh]["harm"] >= 2
                       else pick(rng, ["undercuts", "undercuts", "unclear"]))
            prior = pick(rng, [0, 0, 1, 2])
        b = HARASS_BEHAVIOURS[beh]
        variant = rng.randrange(len(b["say"]))
        target = b["say"][variant][1]
        claim = pick(rng, b["claims"])
        fmt = pick(rng, b["formats"])
        if claim == "provoked":
            fmt = pick(rng, ["reply", "comment"])
        # who flagged it has to agree with what the thread shows
        if target == "self":
            reporter = "classifier"
        elif claim in ("joke_between_friends", "seeking_answer") and support == "undercuts":
            reporter = "person_targeted"
        else:
            reporter = pick_w(rng, {"classifier": 3, "person_targeted": 2, "bystander": 2, "trusted_flagger": 1})
            if reporter == "person_targeted" and claim == "joke_between_friends" and support == "supports":
                reporter = "bystander"
        threshold = pick(rng, [0.8, 0.85, 0.9])
        if reporter == "classifier":
            if reg == "clear" and support == "undercuts":
                clf = between(rng, max(threshold + 0.04, 0.93), 0.99, digits=3)
            elif reg in ("clear", "borderline"):
                clf = near(rng, threshold + 0.02, 0.025, lo=threshold + 0.003, hi=0.99, digits=3)
            else:
                clf = between(rng, threshold + 0.005, 0.99, digits=3)
        else:
            clf = maybe(rng, between(rng, 0.2, 0.95), 0.5)   # logged at report time; may sit below the line
        parent_deleted = fmt in ("reply", "comment") and (
            (claim == "provoked" and support == "unclear") or (claim != "provoked" and chance(rng, 0.15)))
        followers = _audience(rng, 700, 1.3)
        if reg == "conflicting" and chance(rng, 0.45):
            followers, prior = rng.randint(40000, 450000), 0      # first-time violator with a large audience
        other = pick(rng, [0, 0, 0, 1, 2])
        # sometimes the ladder was run on ALL active strikes instead of harassment strikes only
        misapplied = other > 0 and chance(rng, 0.35)
        step = min(3, prior + 1 + (other if misapplied else 0))
        posted = rand_date(rng)
        acted = posted + timedelta(days=pick(rng, [0, 0, 1, 2, 4]))
        return {
            "regime": reg, "behaviour": beh, "variant": variant, "target": target, "claimed_context": claim,
            "statement_variant": rng.randrange(2), "context_support": support, "format": fmt,
            "reporter": reporter, "classifier_score": clf, "threshold": threshold,
            "first_decision": ("automated" if reporter == "classifier" and chance(rng, 0.7) else "frontline reviewer"),
            "parent_deleted": parent_deleted, "followers": followers, "account_age_days": rng.randint(25, 3600),
            "prior_strikes": prior, "other_strikes": other, "penalty_misapplied": misapplied, "penalty_step": step,
            "rule_cited": b["rule"] if chance(rng, 0.7) else "targeted_harassment",
            "policy_shown": chance(rng, 0.7), "tone": pick_w(rng, {"calm": 3, "angry": 2, "apologetic": 1}),
            "attachment": support == "supports" and chance(rng, 0.5),
            "platform": pick(rng, PLATFORMS), "handle": _handle(rng), "appeal_id": _appeal_id(rng),
            "content_id": f"{fmt[0]}_{hexid(rng, 10)}", "posted": posted, "action_date": acted,
            "appeal_date": acted + timedelta(days=pick(rng, [0, 1, 1, 2, 3, 6, 11])),
            # what the thread shows, as the reviewer sees it: some supporting threads only point partly that way
            "thread_shows": "partly" if support == "supports" and chance(rng, 0.45) else support,
        }

    def _statement(self, f, rng):
        parts = [HARASS_CLAIMS[f["claimed_context"]]["say"][f["statement_variant"]]]
        if f["tone"] == "angry":
            parts.append(pick(rng, ["This is ridiculous.", "Your filter is broken and I want a human to look at this.",
                                    "Other accounts say far worse and nothing happens to them."]))
        elif f["tone"] == "apologetic":
            parts.append(pick(rng, ["If it came across badly I'm happy to delete it myself, but I don't think it "
                                    "deserves a strike.",
                                    "I can see how it looked out of context. I'd rather edit it than take a strike."]))
        return noisy(join_sentences(parts), rng, 0.2)

    def _flag_text(self, f):
        if f["reporter"] == "classifier":
            return f"{REPORTERS['classifier']}, score {f['classifier_score']} (auto-action at {f['threshold']})"
        tail = f"; classifier score at report time {f['classifier_score']}" if f["classifier_score"] is not None else ""
        return REPORTERS[f["reporter"]] + tail

    def render(self, f, rng, surface):
        b = HARASS_BEHAVIOURS[f["behaviour"]]
        summary = f"A {f['format']} that {b['say'][f['variant']][0]}."
        evidence = HARASS_CLAIMS[f["claimed_context"]]["ev"][f["thread_shows"]]
        stmt = self._statement(f, rng)
        sec, title, text = HARASS_RULES[f["rule_cited"]]
        penalty = HARASS_PENALTY[f["penalty_step"]]
        if surface == "json":
            return dumps(prune({
                "appeal": {"id": f["appeal_id"], "filed": f["appeal_date"].isoformat(), "statement": stmt,
                           "attachment": "screenshot of the thread" if f["attachment"] else None},
                "content": {"id": f["content_id"], "format": f["format"], "posted": f["posted"].isoformat(),
                            "reviewer_summary": summary,
                            "parent_post": "deleted by its author" if f["parent_deleted"] else None},
                "enforcement": {"rule": f"{sec} {title}", "flagged_by": REPORTERS[f["reporter"]],
                                "classifier_score": f["classifier_score"],
                                "auto_action_threshold": f["threshold"] if f["reporter"] == "classifier" else None,
                                "decided_by": f["first_decision"], "penalty": penalty,
                                "actioned": f["action_date"].isoformat()},
                "account": {"handle": f["handle"], "followers": f["followers"], "age_days": f["account_age_days"],
                            "active_strikes_before": {"harassment": f["prior_strikes"],
                                                      "other_rules": f["other_strikes"]}},
                "thread_context": evidence,
                "policy": {"section": sec, "text": text, "penalties": HARASS_LADDER} if f["policy_shown"] else None,
            }))
        ds = _ds(rng)
        if surface == "appeal_form":
            lines = [f"{f['platform']} | Appeal a moderation decision", f"Reference: {f['appeal_id']}",
                     f"Submitted {fmt_date(f['appeal_date'], style=ds)} by {f['handle']}", "",
                     f"Decision appealed: removal of your {f['format']} from {fmt_date(f['posted'], style=ds)} "
                     f"under rule {sec} ({title}). Penalty applied: {penalty}.", "",
                     "Why do you think we got this wrong?", f"> {stmt}"]
            if f["attachment"]:
                lines += ["", "Attachments: 1 file (screenshot of the thread)"]
            lines += ["", "-- appended by appeals tooling --", kv_lines([
                ("Removed content (reviewer summary)", summary),
                ("Parent post", "deleted by its author" if f["parent_deleted"] else None),
                ("Flagged by", self._flag_text(f)),
                ("Decided by", f["first_decision"]),
                ("Account", f"{_count(f['followers'], rng)} followers, {f['account_age_days']} days old"),
                ("Active strikes before this one", f"harassment {f['prior_strikes']}, other {f['other_strikes']}"),
                ("Thread", evidence),
                (f"Rule {sec}", text if f["policy_shown"] else None),
                ("Penalty ladder", HARASS_LADDER if f["policy_shown"] else None),
            ], rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]))]
            return "\n".join(lines)
        # queue_card
        note = None
        if f["first_decision"] == "frontline reviewer":
            basis = "the classifier hit" if f["reporter"] == "classifier" else "the report"
            note = pick(rng, [f"Removed on {basis}; did not look at the author's other posts.",
                              "Read it as targeting a specific person. Not sure about the context.",
                              f"Went with removal because of {basis}. Could be read either way.", None])
        head = (f"[APPEALS | harassment & bullying | T2]  {f['appeal_id']}  "
                f"filed {fmt_date(f['appeal_date'], style=ds)}  (actioned {fmt_date(f['action_date'], style=ds)})")
        body = kv_lines([
            ("Content", f"{f['format']} {f['content_id']} by {f['handle']} "
                        f"({_count(f['followers'], rng)} followers, account {f['account_age_days']}d)"),
            ("Summary", summary),
            ("Parent", "deleted by its author" if f["parent_deleted"] else None),
            ("Rule applied", f"{sec} {title}"),
            ("Flag", self._flag_text(f)),
            ("Penalty", penalty),
            ("Prior active strikes", f"{f['prior_strikes']} harassment / {f['other_strikes']} other"),
            ("Thread", evidence),
            ("T1 note", note),
            ("Appellant", f'"{stmt}"'),
            ("Ladder", HARASS_LADDER if f["policy_shown"] else None),
        ], rng, style=pick(rng, ["aligned", "colon", "dash"]))
        return f"{head}\n{body}"

    def questions(self):
        return {
            "appeal_ruling": choice("What should the appeals reviewer do with this removal?", {
                "uphold_in_full": "Keep the post down and leave the strike and any other penalty exactly as applied.",
                "uphold_reduced_penalty": "Keep the post down but reduce the penalty, for example by withdrawing "
                                          "the strike or shortening a restriction.",
                "restore_post": "Put the post back up and withdraw every penalty that came with its removal.",
                "refer_to_specialist": "Pass the appeal to a senior harassment specialist because the right outcome "
                                       "cannot be settled at this level.",
            }),
            "governing_rule": choice("Which rule, if any, does the removed post most plausibly break?", {
                "targeted_harassment": "The rule against insulting, demeaning or repeatedly going after a specific "
                                       "person.",
                "private_information": "The rule against posting details that help others find or identify a "
                                       "private person offline.",
                "veiled_threat": "The rule against remarks suggesting harm could come to someone, even when framed "
                                 "as a warning or a joke.",
                "brigading": "The rule against pointing an audience at a specific person so that they swarm, "
                             "message or report them.",
                "no_rule_broken": "None of them; the post may be rude or critical but stays within the rules.",
            }),
            "harm_to_target": score("How much harm is the post likely to do to the person it is aimed at?", [
                "Negligible: aimed at the author's own account, or a jab at a public figure's work nobody would take "
                "personally.",
                "Minor: a passing insult in an argument, or a remark the person it is about plainly took in good part.",
                "Significant: demeaning or repeatedly going after a person, or a jab pushed to a large audience "
                "(tens of thousands of followers) that could draw a pile-on.",
                "Serious: hints that harm could come to the person, or gives details that expose them offline, or "
                "sends a large audience after them.",
            ]),
            "context_credibility": score("How believable is the appellant's account of the context around the post?", [
                "Not believable: what the thread shows contradicts it.",
                "Doubtful: the thread shows nothing either way.",
                "Plausible: the thread shows something pointing the appellant's way, short of confirming it.",
                "Well supported: the thread plainly confirms it.",
            ]),
            "penalty_proportionate": noul(
                "The penalty attached to this removal fits the post and the account's record under the escalation "
                "ladder.",
                yes="The post broke a rule and the penalty matches what the ladder sets for the account's active "
                    "harassment strikes.",
                no="The post broke no rule, or the penalty is heavier or lighter than the ladder sets for this "
                   "account."),
        }


# =========================================================================== schema 2
STRIKE_RULES = [("spam: repeated link posting", 1), ("fake engagement", 1), ("regulated goods in a post", 1),
                ("harassment", 2), ("misleading health claim", 2), ("graphic media without a warning screen", 2),
                ("private information", 3), ("threats and intimidation", 3)]
SUSPENSION_TIERS = {"7_day": (3, "7-day suspension"), "30_day": (4, "30-day suspension"),
                    "permanent": (5, "permanent suspension")}
ACCOUNT_KINDS = {"personal": "personal account", "small_business": "small-business page",
                 "group_admin": "personal account, admin of a 4,000-member community group",
                 "local_news": "volunteer-run local news page"}
ACCOUNT_USE = {"personal": "", "small_business": "This page is how my customers place their orders.",
               "group_admin": "Thousands of people rely on the group I run.",
               "local_news": "Our volunteers use this page to post road closures and council news."}
SUSP_POSITIONS = {
    "apologises": ["I know I broke the rules more than once and I'm sorry. I've deleted the old posts.",
                   "Fair enough on the earlier strikes, I got carried away. I've read the rules properly now."],
    "disputes_trigger": ["The last strike was wrong. I was quoting something in order to disagree with it, and "
                         "without that strike I would not be suspended.",
                         "I accept the older strikes, but the newest one is a mistake and it's the one that got "
                         "me suspended."],
    "rules_unclear": ["Nobody explained the first strikes to me. I didn't realise what I was posting broke the rules.",
                      "I didn't know the rules had changed. I'll follow them now that I understand them."],
    "blames_reporters": ["A group that dislikes me has been mass-reporting everything I post. This is coordinated.",
                         "These reports all came from people who follow each other and hate my page. Look at who "
                         "reported me."],
    "blames_platform": ["Your moderation is a joke. Half the accounts I follow post worse every day.",
                        "This is censorship. I've done nothing other accounts don't do all the time."],
}


def _ladder_text(window):
    return (f"3 active strikes: 7-day suspension. 4: 30-day suspension. 5 or more: permanent suspension. A strike "
            f"stays active for {window} days; strikes overturned on appeal no longer count.")


class SuspensionAppeal(Schema):
    name = "suspension_appeal"
    description = "An account suspended after repeated strikes asks to be reinstated: keep, shorten, swap, lift or refer."
    surfaces = {"json": 0.45, "strike_ledger": 0.3, "second_review_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        window = pick(rng, [90, 90, 180])
        tier = pick_w(rng, {"7_day": 4, "30_day": 3, "permanent": 2})
        need = SUSPENSION_TIERS[tier][0]
        ages = [0] + sorted(rng.sample(range(5, window - 5), need - 1))
        rules = [pick(rng, STRIKE_RULES, [3, 3, 3, 2, 2, 2, 1, 1])[0] for _ in range(need)]
        decided = [pick_w(rng, {"automated": 2, "reviewer": 3}) for _ in range(need)]
        appeals = ["none"] + [pick_w(rng, {"none": 5, "upheld": 3}) for _ in range(need - 1)]
        position = pick(rng, list(SUSP_POSITIONS))
        kind = pick_w(rng, {"personal": 4, "small_business": 3, "group_admin": 2, "local_news": 1})
        mass_report, lookalike, twist = False, False, None
        severe = [r for r, s in STRIKE_RULES if s >= 2]
        gravest = [r for r, s in STRIKE_RULES if s == 3]
        if reg == "clear":
            if chance(rng, 0.5):          # a well-founded suspension
                decided = ["reviewer"] * need
                rules[0] = pick(rng, severe)
                position = pick(rng, ["blames_platform", "rules_unclear", "blames_reporters"])
                lookalike = chance(rng, 0.4)
            else:                         # the count does not survive a check
                i = rng.randrange(1, need)
                if chance(rng, 0.5):
                    appeals[i] = "overturned"
                else:
                    ages[i] = window + rng.randint(6, 40)      # counted although it had already expired
                decided[0] = "automated"
                rules[0] = pick(rng, [r for r, s in STRIKE_RULES if s == 1])
                position = pick(rng, ["disputes_trigger", "apologises"])
        elif reg == "borderline":
            # the oldest counted strike sits right on the edge of the window
            ages[-1] = near(rng, window, 5, lo=ages[-2] + 1, integer=True)
            decided[0] = pick(rng, ["automated", "reviewer"])
            position = pick(rng, ["disputes_trigger", "apologises", "rules_unclear"])
        else:
            twist = pick(rng, ["mass_report", "apology_vs_severity", "lookalike"])
            if twist == "mass_report":
                mass_report, position = True, "blames_reporters"
                rules[1] = pick(rng, gravest)
            elif twist == "apology_vs_severity":
                position, kind = "apologises", pick(rng, ["small_business", "local_news", "group_admin"])
                rules[0] = pick(rng, gravest)
            else:
                position, lookalike = "apologises", True
        strikes = sorted(({"age": a, "rule": r, "decided_by": d, "appeal": s}
                          for a, r, d, s in zip(ages, rules, decided, appeals)), key=lambda x: x["age"])
        valid = sum(1 for s in strikes if s["age"] <= window and s["appeal"] != "overturned")
        served = rng.randint(1, 6 if tier == "7_day" else 14)
        for s in strikes:
            s["overturned_day"] = rng.randint(1, served) if s["appeal"] == "overturned" else None
        reports = rng.randint(22, 60) if mass_report else rng.randint(1, 6)
        years = rng.randint(1, 12)
        if twist == "apology_vs_severity":
            years = rng.randint(6, 12)
        return {
            "regime": reg, "twist": twist, "tier": tier, "strikes_needed": need, "window_days": window,
            "strikes": strikes, "valid_active_strikes": valid, "threshold_really_met": valid >= need,
            "position": position, "statement_variant": rng.randrange(2), "account_kind": kind,
            "account_years": years, "followers": _audience(rng, 2500, 1.4),
            "reports_on_trigger": reports,
            "reports_from_new_accounts": (int(reports * rng.uniform(0.6, 0.9)) if mass_report
                                          else pick(rng, [0, 0, 1])),
            "lookalike_during_suspension": lookalike, "prior_suspensions": pick(rng, [0, 0, 0, 1, 2]),
            "days_served": served, "suspended_on": rand_date(rng), "handle": _handle(rng),
            "platform": pick(rng, PLATFORMS), "appeal_id": _appeal_id(rng), "reviewer": person(rng),
            "show_ages": chance(rng, 0.4), "ladder_shown": chance(rng, 0.8),
        }

    def _statement(self, f, rng):
        s = [SUSP_POSITIONS[f["position"]][f["statement_variant"]], ACCOUNT_USE[f["account_kind"]]]
        return noisy(join_sentences(s), rng, 0.2)

    def _status(self, f, s, idx):
        if idx == 0:
            return "contested in this appeal" if f["position"] == "disputes_trigger" else "not appealed separately"
        if s["appeal"] == "overturned":
            day = f["suspended_on"] + timedelta(days=s["overturned_day"])
            return f"overturned on appeal {day.isoformat()}"
        return {"none": "not appealed", "upheld": "appealed, upheld"}[s["appeal"]]

    def _signals(self, f):
        out = []
        if f["reports_on_trigger"] >= 10:
            out.append(f"The triggering post drew {f['reports_on_trigger']} reports, {f['reports_from_new_accounts']} "
                       f"of them from accounts under 30 days old.")
        if f["lookalike_during_suspension"]:
            out.append("A new account with a near-identical handle began posting the same kind of content two days "
                       "into the suspension.")
        if f["prior_suspensions"]:
            out.append(f"The account has {f['prior_suspensions']} earlier suspension(s) on record.")
        return out

    def render(self, f, rng, surface):
        susp = f["suspended_on"]
        filed = susp + timedelta(days=f["days_served"])
        label = SUSPENSION_TIERS[f["tier"]][1]
        stmt = self._statement(f, rng)
        rows = [(susp - timedelta(days=s["age"]), s, i) for i, s in enumerate(f["strikes"])]
        if surface == "json":
            return dumps(prune({
                "account": {"handle": f["handle"], "type": ACCOUNT_KINDS[f["account_kind"]],
                            "created": susp.year - f["account_years"], "followers": f["followers"],
                            "earlier_suspensions": f["prior_suspensions"]},
                "suspension": {"type": label, "applied": susp.isoformat(), "active_strikes_counted": f["strikes_needed"],
                               "days_served_at_appeal": f["days_served"]},
                "strikes": [{"date": d.isoformat(), "rule": s["rule"], "decided_by": s["decided_by"],
                             "appeal": self._status(f, s, i),
                             "days_before_suspension": s["age"]} for d, s, i in rows],
                "appeal": {"id": f["appeal_id"], "filed": filed.isoformat(), "statement": stmt},
                "signals": {"reports_on_triggering_post": f["reports_on_trigger"],
                            "reports_from_accounts_under_30_days": f["reports_from_new_accounts"],
                            "lookalike_account_during_suspension": f["lookalike_during_suspension"] or None},
                "policy": ({"ladder": _ladder_text(f["window_days"])} if f["ladder_shown"]
                           else {"strike_lifetime_days": f["window_days"]}),
            }))
        ds = _ds(rng)
        if surface == "strike_ledger":
            tab = table(["#", "date", "days before suspension", "rule", "decided by", "appeal"],
                        [[len(rows) - k, fmt_date(d, style=ds), s["age"], s["rule"], s["decided_by"],
                          self._status(f, s, i)] for k, (d, s, i) in enumerate(rows)], rng)
            sig = self._signals(f)
            parts = [f"ENFORCEMENT HISTORY | {f['handle']} | {ACCOUNT_KINDS[f['account_kind']]}",
                     f"On {f['platform']} since {susp.year - f['account_years']} | "
                     f"{_count(f['followers'], rng)} followers | earlier suspensions: {f['prior_suspensions']}",
                     f"Current action: {label}, applied {fmt_date(susp, style=ds)}; appeal {f['appeal_id']} filed on "
                     f"day {f['days_served']}", "", tab, "", f'Appellant: "{stmt}"']
            if sig:
                parts += ["", "Signals:", bullets(sig, rng)]
            parts += ["", "Ladder: " + _ladder_text(f["window_days"]) if f["ladder_shown"]
                      else f"Strikes stay active for {f['window_days']} days."]
            return "\n".join(parts)
        # second_review_note
        hes = pick(rng, ["I have not yet checked whether every strike was still active on the day.",
                         "My read is that the history speaks for itself, but I'd like a second view.",
                         "The appellant's explanation sounds sincere to me; the record is mixed.",
                         "Not sure how much weight to give the reports on the last strike."
                         if f["reports_on_trigger"] >= 10 else "Nothing unusual in how the last strike was reported."])
        lines = [f"Second review requested: suspension appeal {f['appeal_id']}",
                 f"From: {f['reviewer']} (appeals, T1)",
                 "Procedure: every suspension appeal gets a second reviewer before it closes.", "",
                 f"{f['handle']} ({ACCOUNT_KINDS[f['account_kind']]}, {_count(f['followers'], rng)} followers, on "
                 f"{f['platform']} since {susp.year - f['account_years']}) got a {label} on {fmt_date(susp, style=ds)}. "
                 f"The system counted {f['strikes_needed']} active strikes:",
                 bullets([f"{fmt_date(d, style=ds)} ({s['age']} days before the suspension): {s['rule']} "
                          f"({s['decided_by']}; {self._status(f, s, i)})" for d, s, i in rows], rng, "-"), "",
                 f'Appeal filed on day {f["days_served"]}: "{stmt}"']
        extra = self._signals(f)
        extra.append("Ladder: " + _ladder_text(f["window_days"]) if f["ladder_shown"]
                     else f"Strikes stay active for {f['window_days']} days.")
        extra.append(hes)
        lines += ["", join_sentences(extra)]
        return "\n".join(lines)

    def questions(self):
        return {
            "suspension_ruling": choice("How should the suspension appeal be resolved?", {
                "keep_suspension": "Leave the suspension in place for its full term.",
                "shorten_suspension": "Keep the account suspended, but for a shorter term than the one applied.",
                "swap_for_restrictions": "Lift the suspension and instead limit specific features, such as posting "
                                         "links or sending messages, for a set period.",
                "lift_suspension": "Lift the suspension and restore the account with no further restriction.",
                "refer_to_panel": "Send the case to the enforcement review panel because it needs a decision above "
                                  "frontline level.",
            }),
            "record_gravity": score("Taken together, how serious is the account's record of violations?", [
                "Minor: only low-level strikes, such as spam, fake engagement or regulated goods in a post.",
                "Moderate: one or two strikes for harassment, misleading health claims or unwarned graphic media; the "
                "rest low-level.",
                "Serious: a strike for private information or threats, three or more of the middle kind, or two "
                "earlier suspensions.",
                "Grave: two or more strikes for private information or threats, or a look-alike account carrying on "
                "during the suspension.",
            ]),
            "ladder_threshold_met": noul(
                "Counting only strikes that were valid and still active, the account had reached the suspension "
                "tier that was applied.",
                yes="After setting aside expired or overturned strikes, enough remain to reach the tier applied.",
                no="Once expired or overturned strikes are set aside, the count falls short of the tier applied."),
            "reoffence_likelihood": score("If the account were restored now, how likely is it to break the rules again soon?", [
                "Unlikely: the record looks like isolated lapses that the appellant has acknowledged.",
                "Possible: some risk remains, though the appellant shows some awareness.",
                "Likely: the pattern is ongoing and the appeal shows little sign of change.",
                "Almost certain: the owner is still behaving the same way.",
            ]),
            "trigger_needs_fresh_look": noul(
                "The strike that triggered the suspension should be reviewed again on its own merits before the "
                "suspension is confirmed."),
        }


# =========================================================================== schema 3
GOODS_RULES = {
    "weapons": ("GP-4", "Weapons and knives",
                "Knives with a blade longer than 3 in. (7.6 cm) may be sold only by verified sellers with age-checked "
                "checkout; spring-assisted and concealed blades are prohibited. Kitchen tools and appliances are "
                "not covered."),
    "cbd": ("GP-11", "Hemp and CBD",
            "Topical CBD products may be listed with a lab certificate for the same batch showing THC at or below "
            "0.2%. Listings may not claim to treat any condition."),
    "health_claims": ("GP-13", "Health claims",
                      "Supplements and remedies may be sold, but may not claim to prevent, treat or cure a disease or "
                      "to replace prescribed medicine. General wellness wording is allowed."),
    "medical_devices": ("GP-9", "Medical devices",
                        "Contact lenses, including purely cosmetic ones, may be sold only by registered opticians and "
                        "only with a prescription check at checkout."),
    "child_safety": ("GP-15", "Children's safety products",
                     "Used child car seats may be listed if the model has no open recall, the seat is within its "
                     "expiry date and the listing says whether it has been in a collision."),
    "animal_products": ("GP-6", "Animal products",
                        "Elephant ivory is prohibited whatever its age. Bone, horn and resin carvings may be sold if "
                        "the listing states the material."),
    "explosives": ("GP-3", "Explosives and fireworks",
                   "Fireworks, sparklers, flares and any other pyrotechnic item are prohibited, including those sold "
                   "in shops for garden use."),
}
# item -> title, typical price, rule, item class, what an image model calls it, keywords that trip the filter,
# listing details by compliance (ok / fixable / bad / unknown), appeal evidence by quality, and the seller's plea
GOODS = {
    "blade_grinder": {
        "title": "Blade coffee grinder, stainless, used twice", "price": 24, "rule": "weapons", "cls": "unrestricted",
        "private": True,
        "image_label": "knife or blade", "terms": ["blade", "razor-sharp"],
        "variants": {"ok": ["Electric grinder; the small mixing blade sits sealed inside the bowl. Photos show it "
                            "beside a bag of coffee beans.",
                            "Hand-crank grinder. Description: 'razor-sharp blade gives a fine espresso grind in "
                            "seconds.'"]},
        "evidence": {"strong": "a link to the manufacturer's product page for the same model",
                     "partial": "a photo of the box with the model number cut off"},
        "plea": "It's a coffee grinder. The only blade is the one inside that grinds the beans."},
    "folding_knife": {
        "title": "Folding pocket knife, gift boxed", "price": 45, "rule": "weapons", "cls": "conditional",
        "image_label": "knife or blade", "terms": ["knife", "blade"],
        "variants": {"ok": ["Blade 2.75 in. (7 cm), non-locking. Listed under camping gear."],
                     "fixable": ["Blade 3.4 in. (8.6 cm), locking. The listing's age-check setting is off."],
                     "bad": ["Blade about 3.5 in. The listing video shows it flicking open on a spring."],
                     "unknown": ["Blade length not stated; one photo shows the knife held in a hand."]},
        "evidence": {"strong": "a photo of the open blade lying against a ruler",
                     "partial": "a spec sheet for a similar but not identical model"},
        "plea": "It's a small pocket knife for camping. Every outdoor shop sells these."},
    "cbd_balm": {
        "title": "CBD muscle balm, 500 mg, 60 ml", "price": 38, "rule": "cbd", "cls": "conditional",
        "image_label": "cannabis product", "terms": ["CBD", "hemp"],
        "variants": {"ok": ["Topical balm; batch B-2231 visible on the label. Description: 'for massage after "
                            "exercise.'"],
                     "fixable": ["Topical balm, batch B-2231 on the label. Description says it 'relieves arthritis "
                                 "pain'."],
                     "bad": ["Label lists THC at 0.6%. Description: 'full-spectrum for a stronger effect.'"],
                     "unknown": ["No batch number visible in the photos; THC content not stated."]},
        "evidence": {"strong": "a lab certificate for batch B-2231 showing THC at 0.1%",
                     "partial": "a two-year-old lab certificate that names no batch"},
        "plea": "My balm is legal and lab tested. High-street shops sell exactly the same thing."},
    "herbal_tincture": {
        "title": "Elderberry immune tincture, 50 ml", "price": 18, "rule": "health_claims", "cls": "unrestricted",
        "image_label": "medicine", "terms": ["immune", "flu"],
        "variants": {"ok": ["Description: 'a traditional syrup that supports everyday wellbeing.'"],
                     "fixable": ["Description: 'knocks out flu in 24 hours, better than anything from the chemist.'"],
                     "bad": ["Description: 'a natural alternative to your prescribed inhaler.'"],
                     "unknown": ["Description is a single line: 'message me to ask what it's good for!'"]},
        "evidence": {"strong": "a revised description with the health claims taken out",
                     "partial": "a link to a blog post about the history of elderberry"},
        "plea": "Elderberry has been used for centuries. I only wrote what my customers tell me."},
    "coloured_lenses": {
        "title": "Coloured contact lenses, 6 pairs, assorted", "price": 29, "rule": "medical_devices",
        "cls": "conditional", "image_label": "medical device", "terms": ["contact lenses"],
        "variants": {"ok": ["Listing sends buyers to a prescription upload step at checkout."],
                     "fixable": ["Title says 'no prescription needed'."],
                     "bad": ["Title says 'no prescription needed'; lenses ship from overseas in unmarked packs."],
                     "unknown": ["Cosmetic lenses with no corrective power; nothing about prescriptions either way."]},
        "verified_by": {"ok": True, "fixable": True, "bad": False, "unknown": None},
        "evidence": {"strong": "the seller's optician registration number, which matches the register",
                     "partial": "a statement that the lenses are 'only for costume parties'"},
        "plea": "These are costume lenses. They don't correct anything, so a prescription makes no sense."},
    "car_seat": {
        "title": "Toddler car seat, 2019, good condition", "price": 60, "rule": "child_safety", "cls": "conditional",
        "private": True,
        "image_label": "child safety product", "terms": ["car seat"],
        "variants": {"ok": ["Model not on the recall list; expiry 2028 on the label; listing says 'never in a "
                            "collision'."],
                     "fixable": ["Model not on the recall list; expiry 2027; the listing does not say whether it has "
                                 "been in a collision."],
                     "bad": ["The model had a buckle recall in 2021; the listing does not mention a repair."],
                     "unknown": ["Model number not visible in the photos; the seller says it is 'about five years "
                                 "old'."]},
        "evidence": {"strong": "a screenshot of the manufacturer's recall checker showing 'no open recall' for the "
                               "seat's serial number",
                     "partial": "a photo of the label with the serial number partly covered"},
        "plea": "My own kids used it and it's in great condition. I just want it to go to another family."},
    "antique_carving": {
        "title": "Antique carved figurine, circa 1900", "price": 180, "rule": "animal_products", "cls": "conditional",
        "private": True,
        "image_label": "animal product", "terms": ["ivory", "antique carving"],
        "variants": {"ok": ["Material given as ox bone; a close-up shows bone pores on the base."],
                     "bad": ["The title calls it an 'old ivory netsuke'."],
                     "unknown": ["Material given as 'ivory-coloured, maybe resin?'; the photos are blurry."]},
        "evidence": {"strong": "an appraiser's letter identifying the material as bone",
                     "partial": "a family photo showing the figurine on a shelf decades ago"},
        "plea": "This was my grandmother's. It's very old and I'm just clearing out her house."},
    "hemp_tote": {
        "title": "Hemp canvas tote bag, natural", "price": 16, "rule": "cbd", "cls": "unrestricted", "decoy": True,
        "image_label": "cannabis product", "terms": ["hemp"],
        "variants": {"ok": ["Woven hemp-fibre shopping bag, 40 x 38 cm; the photos show it holding groceries."]},
        "evidence": {"strong": "the maker's product page describing it as a hemp-fibre shopping bag",
                     "partial": "a photo of the sewn-in fabric label"},
        "plea": "It's a shopping bag made of hemp cloth. There is nothing in it to smoke or eat."},
    "ivory_shoes": {
        "title": "Ivory satin wedding shoes, size 6", "price": 42, "rule": "animal_products", "cls": "unrestricted",
        "decoy": True, "private": True, "image_label": "animal product", "terms": ["ivory"],
        "variants": {"ok": ["Satin court shoes in an ivory colour, worn once; the photos show the shoes and their box."]},
        "evidence": {"strong": "the shop's product page listing the colour as 'ivory'",
                     "partial": "a photo of the shoebox"},
        "plea": "Ivory is the colour. They are satin wedding shoes."},
    "fireworks_scarf": {
        "title": "Silk scarf, fireworks print", "price": 34, "rule": "explosives", "cls": "unrestricted", "decoy": True,
        "image_label": "pyrotechnics", "terms": ["fireworks"],
        "variants": {"ok": ["A 90 cm silk scarf printed with a night-sky fireworks pattern."]},
        "evidence": {"strong": "the designer's catalogue page for the print",
                     "partial": "a close-up photo of the fabric"},
        "plea": "It's a scarf with a fireworks pattern on it, not fireworks."},
    "lens_cases": {
        "title": "Contact lens cases, pack of 5", "price": 7, "rule": "medical_devices", "cls": "unrestricted",
        "decoy": True, "image_label": "medical device", "terms": ["contact lenses"],
        "variants": {"ok": ["Plastic screw-top lens cases; no lenses are included, as the photos show."]},
        "evidence": {"strong": "the product page stating the pack contains cases only",
                     "partial": "a photo of one empty case"},
        "plea": "These are just the little plastic cases. There are no lenses in the pack."},
    "garden_fireworks": {
        "title": "Garden fireworks, unopened", "price": 55, "rule": "explosives", "cls": "prohibited",
        "image_label": "pyrotechnics", "terms": ["fireworks", "rockets", "sparklers"],
        "variants": {"bad": ["24-rocket garden bundle in the original packaging.",
                             "Box of sparklers and table-top fountains, bought for a party and never used."]},
        "evidence": {"strong": "a receipt from a garden centre and a photo of the safety marking on the box",
                     "partial": "a link to a supermarket page selling the same product"},
        "plea": "These are the same garden fireworks every supermarket sells in November. They're completely legal."},
}


class MarketplaceListingAppeal(Schema):
    name = "marketplace_listing_appeal"
    description = "A marketplace listing removed as prohibited or regulated goods, appealed by the seller."
    surfaces = {"json": 0.45, "queue_card": 0.3, "seller_appeal": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        if reg == "clear":
            comp = pick(rng, ["ok", "bad"])
            ev = "strong" if comp == "ok" else pick(rng, ["none", "none", "partial"])
            prior = 0 if comp == "ok" else rng.randint(2, 5)
        elif reg == "borderline":
            comp = pick_w(rng, {"fixable": 3, "unknown": 2, "ok": 3, "bad": 1})
            ev = pick_w(rng, {"partial": 3, "strong": 1, "none": 1})
            prior = pick(rng, [0, 0, 1, 2])
        else:
            # well-documented appeals for items that should not be sold, thin appeals for items that are fine
            comp = pick(rng, ["ok", "ok", "bad", "fixable", "unknown"])
            ev = "strong" if comp in ("bad", "unknown") else pick(rng, ["none", "partial"])
            prior = rng.randint(3, 6) if comp == "ok" else 0
        items = [k for k, g in GOODS.items() if comp in g["variants"]]
        item = "blade_grinder" if comp == "ok" and reg == "clear" and chance(rng, 0.4) else pick(rng, items)
        if comp == "ok" and item != "blade_grinder" and chance(rng, 0.65):
            item = pick(rng, [k for k, g in GOODS.items() if g.get("decoy")])
        g = GOODS[item]
        detection = pick_w(rng, {"keyword": 3, "image": 3, "buyer_report": 2, "sweep": 1})
        if item == "blade_grinder" or g.get("decoy"):
            detection = pick(rng, ["keyword", "image"])
        img_thr = pick(rng, [0.7, 0.75, 0.8])
        img_score = None
        if detection == "image":
            img_score = (near(rng, img_thr + 0.03, 0.03, lo=img_thr + 0.005, hi=0.99, digits=3)
                         if reg == "borderline" or item == "blade_grinder" or g.get("decoy")
                         else between(rng, img_thr + 0.02, 0.99, digits=3))
        verified = g.get("verified_by", {}).get(comp, "sample")
        if verified == "sample":
            verified = pick_w(rng, {True: 2, False: 3})
        listed = rand_date(rng)
        removed = listed + timedelta(days=rng.randint(0, 20))
        return {
            "regime": reg, "item": item, "item_class": g["cls"], "compliance": comp,
            "detail_variant": rng.randrange(len(g["variants"][comp])), "evidence_quality": ev,
            "detection": detection, "matched_term": pick(rng, g["terms"]), "image_score": img_score,
            "image_threshold": img_thr, "seller_verified": verified, "prior_removals_12mo": prior,
            "seller_rating": round(between(rng, 3.4, 5.0), 1), "completed_sales": _audience(rng, 60, 1.3),
            "seller_since": listed.year - rng.randint(0, 9),
            "price": round(g["price"] * rng.uniform(0.6, 1.6), 2), "policy_shown": chance(rng, 0.75),
            "tone": pick_w(rng, {"calm": 3, "frustrated": 2}),
            "seller": (_handle(rng).lstrip("@") if g.get("private") or chance(rng, 0.4)
                       else f"{rng.choice(_SHOP_A)} {rng.choice(_SHOP_B)}"),
            "listing_id": f"LST-{hexid(rng, 8).upper()}", "appeal_id": _appeal_id(rng),
            "listed": listed, "removed": removed, "appeal_date": removed + timedelta(days=pick(rng, [0, 1, 2, 5, 9])),
            "platform": pick(rng, PLATFORMS),
        }

    def _trigger(self, f):
        g = GOODS[f["item"]]
        if f["detection"] == "keyword":
            return f"keyword filter matched '{f['matched_term']}'"
        if f["detection"] == "image":
            return f"image model: '{g['image_label']}' {f['image_score']} (threshold {f['image_threshold']})"
        if f["detection"] == "buyer_report":
            return f"buyer report, category chosen: 'prohibited item - {g['image_label']}'"
        return "proactive sweep of the category"

    def _statement(self, f, rng):
        g = GOODS[f["item"]]
        parts = [g["plea"]]
        if f["evidence_quality"] != "none":
            parts.append(f"I've attached {g['evidence'][f['evidence_quality']]}.")
        if f["tone"] == "frustrated":
            parts.append(pick(rng, ["This is the second time your system has done this to me.",
                                    "I'm losing sales every day this stays down."]))
        return noisy(join_sentences(parts), rng, 0.2)

    def render(self, f, rng, surface):
        g = GOODS[f["item"]]
        sec, rname, rtext = GOODS_RULES[g["rule"]]
        detail = g["variants"][f["compliance"]][f["detail_variant"]]
        stmt = self._statement(f, rng)
        attach = g["evidence"][f["evidence_quality"]] if f["evidence_quality"] != "none" else None
        verified = {True: "verified business", False: "not verified", None: None}[f["seller_verified"]]
        if surface == "json":
            return dumps(prune({
                "listing": {"id": f["listing_id"], "title": g["title"], "price": money(f["price"], rng),
                            "details": detail, "listed": f["listed"].isoformat()},
                "removal": {"rule": f"{sec} {rname}", "flagged_by": self._trigger(f),
                            "removed": f["removed"].isoformat()},
                "seller": {"name": f["seller"], "verification": verified, "rating": f["seller_rating"],
                           "completed_sales": f["completed_sales"], "member_since": f["seller_since"],
                           "listings_removed_12mo": f["prior_removals_12mo"]},
                "appeal": {"id": f["appeal_id"], "filed": f["appeal_date"].isoformat(), "statement": stmt,
                           "attachment": attach},
                "policy_excerpt": rtext if f["policy_shown"] else None,
            }))
        ds = _ds(rng)
        if surface == "queue_card":
            head = f"{f['platform']} Market | listing appeals | {f['appeal_id']} | removed {fmt_date(f['removed'], style=ds)}"
            body = kv_lines([
                ("Listing", f"{f['listing_id']}  \"{g['title']}\"  {money(f['price'], rng)}"),
                ("Details", detail),
                ("Removed under", f"{sec} {rname}"),
                ("Trigger", self._trigger(f)),
                ("Seller", f"{f['seller']} | {verified or 'verification status not on file'} | "
                           f"{f['seller_rating']} stars, {f['completed_sales']} sales | "
                           f"removals in last 12 months: {f['prior_removals_12mo']}"),
                ("Appeal", f'"{stmt}"'),
                ("Attachment", attach or "none"),
                ("Policy", rtext if f["policy_shown"] else None),
            ], rng, style=pick(rng, ["aligned", "colon"]))
            return f"{head}\n{body}"
        # seller_appeal
        lines = [f"Subject: Appeal - listing {f['listing_id']} removed",
                 f"From: {f['seller']} (seller since {f['seller_since']})",
                 f"Received: {fmt_date(f['appeal_date'], style=ds)} via the {f['platform']} Market seller centre", "",
                 stmt, "", "-- listing snapshot --",
                 kv_lines([("Title", g["title"]), ("Price", money(f["price"], rng)), ("Details", detail)], rng,
                          style="colon"),
                 "", "-- removal record --",
                 kv_lines([("Rule", f"{sec} {rname}"), ("Trigger", self._trigger(f)),
                           ("Seller status", verified or "not on file"),
                           ("Rating / sales", f"{f['seller_rating']} / {f['completed_sales']}"),
                           ("Removals, last 12 months", f["prior_removals_12mo"]),
                           ("Attachment", attach or "none")], rng, style="colon")]
        if f["policy_shown"]:
            lines += ["", f"{sec}: {rtext}"]
        return "\n".join(lines)

    def questions(self):
        return {
            "listing_ruling": choice("What should happen to the removed listing?", {
                "keep_removed": "Keep the listing down; the item cannot be sold on the marketplace in this form.",
                "reinstate_unchanged": "Put the listing back exactly as it was.",
                "reinstate_after_edits": "Let the listing back once the seller removes or corrects specific "
                                         "wording or photos.",
                "reinstate_with_limits": "Let it back with selling limits applied, such as age-checked checkout "
                                         "or sales restricted to verified sellers.",
                "hold_for_proof": "Keep it down until the seller supplies proof such as a lab certificate, a "
                                  "recall check or a registration.",
            }),
            "goods_classification": choice("Under the goods policy, how should the item itself be classed?", {
                "prohibited_item": "An item the marketplace does not allow to be sold at all.",
                "conditionally_allowed": "An item that may be sold only when specific conditions are met.",
                "unrestricted_item": "An ordinary item with no special selling rules.",
                "cannot_tell": "The listing lacks the details needed to tell which class the item falls into.",
            }),
            "harm_if_sold": score("How serious is the harm that could follow if the item were sold as listed?", [
                "None: an ordinary purchase.",
                "Low: a small risk, or a paperwork breach with little real-world effect.",
                "Moderate: a real chance of injury or illness, or of trade the law restricts.",
                "High: plausible serious injury, or trade the law bans outright.",
            ]),
            "appeal_evidence_strength": score("How convincingly does the seller's appeal answer the reason the listing was removed?", [
                "Not at all: no evidence, or evidence that cannot help because the item as listed breaks the policy "
                "anyway.",
                "Weakly: an assertion backed by something partial: a similar model, an undated or batch-less "
                "certificate, a cropped or old photo, a link to someone else's page.",
                "Fairly: a document that fits this exact item, while the listing itself still needs an edit to comply.",
                "Convincingly: a document that fits this exact item and settles what the removal was about.",
            ]),
            "flag_misfired": noul(
                "The automated or reported flag mistook what the item actually is.",
                yes="The flag fired on a word, image or category that misdescribes the item, as when an everyday "
                    "product shares a term with a restricted one.",
                no="The flag identified the kind of item correctly, whatever the final ruling on the listing."),
        }


# =========================================================================== schema 4
MISINFO_POLICY = {
    "health": ("HI-2", "Health misinformation",
               "Claims that contradict health authorities and could lead people to refuse treatment or harm "
               "themselves get a label and reduced reach; claims likely to cause imminent physical harm are removed. "
               "Personal experiences are allowed but may carry a context note."),
    "elections": ("CI-1", "Voting information",
                  "Posts that misstate when, where or how to vote are removed in the 30 days before a vote and "
                  "labelled outside that period. Other disputed claims about elections get a context label."),
}
MISINFO_EXEMPT = "Reporting on, questioning or rebutting a claim is not itself a violation."
# claim -> domain, what the post conveys, the fact-check desk's note (with an accuracy shift), latent accuracy
# 0 (false) .. 4 (accurate), latent harm 0-3
MISINFO_CLAIMS = {
    "zinc_cold": dict(domain="health", say="says zinc lozenges 'cut a cold in half'", acc=2, harm=0,
                      notes=[("Studies suggest zinc may shorten a cold by a day or so in some adults; 'in half' "
                              "overstates it.", 0)]),
    "booster_fever": dict(domain="health", say="describes the author's two days of fever and a sore arm after a "
                                               "booster dose and says 'nobody warns you about this'", acc=3, harm=1,
                          notes=[("Fever and a sore arm are listed side effects on the official leaflet.", 0)]),
    "sunscreen_cancer": dict(domain="health", say="claims a common sunscreen ingredient 'is proven to cause cancer' "
                                                  "and tells followers to stop wearing sunscreen", acc=0, harm=2,
                             notes=[("Regulators say the ingredient is safe at permitted levels; the lab study the "
                                     "post links used doses far above normal exposure.", 0)]),
    "fast_diabetes": dict(domain="health", say="says a three-day water fast 'reverses type 2 diabetes' so medication "
                                               "'is not needed'", acc=1, harm=3,
                          notes=[("Some diets can bring type 2 diabetes into remission under medical supervision; "
                                  "stopping medication without advice is dangerous.", 0)]),
    "outbreak_count": dict(domain="health", say="gives the number of local measles cases from the health agency's "
                                                "weekly bulletin", acc=3, harm=0,
                           notes=[("The agency revised the count upward in its next bulletin, published after the "
                                   "post.", 0)]),
    "antibiotics_flu": dict(domain="health", say="tells parents to ask for antibiotics at the first sign of flu 'to "
                                                 "be safe'", acc=0, harm=2,
                            notes=[("Antibiotics do not treat flu, which is caused by a virus; health agencies advise "
                                    "against this.", 0)]),
    "wrong_day": dict(domain="elections", say="gives the date of the regional election as one day later than the real "
                                              "date", acc=0, harm=3,
                      notes=[("The electoral commission's published date is the day before the one in the post.", 0)]),
    "two_ids": dict(domain="elections", say="says voters must bring two forms of photo ID to vote", acc=1, harm=2,
                    notes=[("The rules require one form of photo ID; the commission's site lists accepted "
                            "documents.", 0)]),
    "postal_deadline": dict(domain="elections", say="gives the postmark deadline for postal ballots from the "
                                                    "commission's first announcement", acc=2, harm=0,
                            notes=[("The commission later extended the deadline by three days; the post gives the "
                                    "earlier date.", 0)]),
    "counted_twice": dict(domain="elections", say="claims ballots in one district were 'counted twice', pointing to a "
                                                  "video of staff re-scanning a batch", acc=1, harm=1,
                          notes=[("The county says the batch was re-scanned after a paper jam and the machine log "
                                  "shows it was counted once.", 0)]),
    "station_moved": dict(domain="elections", say="says the polling station at a primary school has moved to the "
                                                  "public library", acc=2, harm=1,
                          notes=[("The council's polling-place list confirms the move for that ward.", 2),
                                 ("The council's polling-place list still shows the school; the council has not "
                                  "replied to a query.", 0)]),
    "candidate_withdrew": dict(domain="elections", say="says an independent candidate has pulled out of the race and "
                                                       "endorsed a rival", acc=0, harm=2,
                               notes=[("The candidate posted the same morning that she is still running.", 0)]),
}
FRAMINGS = {
    "asserted": "Written as a plain statement of fact, with no source.",
    "question": "Phrased as a question ('can anyone confirm this?'); the author gives no answer.",
    "personal": "Told as the author's own experience, in the first person.",
    "rebuttal": "Shows the claim in a screenshot; the author's caption calls it false and asks people not to share it.",
    "satire": "Written as a spoof headline in the style of a joke news site.",
    "news_report": "A news post about the claim going round; the headline repeats it, and the body quotes officials "
                   "or experts disputing it.",
}
FRAMING_PLEAS = {
    "asserted": ["This is true and I'm allowed to say it. Do your own research.",
                 "I stand by what I posted. Your fact-checkers are wrong about this."],
    "question": ["I was ASKING a question, not claiming anything.",
                 "I asked if anyone could confirm it. How is a question misinformation?"],
    "personal": ["This happened to me. You can't fact-check my own experience.",
                 "I only described what I went through. People deserve to hear real stories."],
    "rebuttal": ["I posted this to warn people it's FALSE. Read my caption.",
                 "The label is on a post debunking the rumour. That is backwards."],
    "satire": ["It's a joke page. Nobody believes a spoof headline.",
               "Satire. Look at the rest of the page."],
    "news_report": ["We are a news outlet reporting that this claim is going round. Our article says clearly that "
                    "it is disputed.",
                    "Our reporting is being labelled as the thing it reports on. Please read the article."],
}
AUTHOR_TYPES = {"health": ["ordinary user", "wellness coach", "community group admin", "nurse (self-described)"],
                "elections": ["ordinary user", "campaign volunteer", "community group admin", "local councillor's aide"]}
PARTNERS = ["Clearwater Fact Desk", "Verity Northwest", "Plainfacts Cooperative"]
LABEL_NAMES = {"False": "False information", "Partly false": "Partly false information",
               "Missing context": "Missing context", "Outdated": "Outdated information"}
APPELLANT_LINKS = {
    "health": {"official": "Here is the health agency's own page.", "news": "I've linked a news article that backs "
               "this up.", "extra": "There's a new study that supports it, link attached."},
    "elections": {"official": "Here is the electoral commission's page.", "news": "I've linked a local news story "
                  "about it.", "extra": "I've attached a photo of the notice I saw."},
}


class MisinformationLabelDispute(Schema):
    name = "misinformation_label_dispute"
    description = "A health or elections post carrying a misinformation label, disputed by its author."
    surfaces = {"json": 0.45, "policy_brief": 0.3, "appeal_form": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        claims = list(MISINFO_CLAIMS)
        if reg == "clear":
            if chance(rng, 0.5):   # false and endorsed, well rated, widely seen
                claim = pick(rng, ["sunscreen_cancer", "fast_diabetes", "antibiotics_flu", "wrong_day",
                                   "candidate_withdrew", "two_ids"])
                framing = "asserted"
                source = "fact_check_partner"
            else:                  # the label landed on a rebuttal or a report by claim-matching
                claim = pick(rng, claims)
                framing = pick(rng, ["rebuttal", "news_report"])
                source = "claim_match"
        elif reg == "borderline":
            claim = pick(rng, ["zinc_cold", "booster_fever", "outbreak_count", "postal_deadline", "counted_twice",
                               "station_moved", "two_ids", "fast_diabetes"])
            framing = pick_w(rng, {"question": 3, "personal": 3, "asserted": 2, "satire": 1})
            source = pick_w(rng, {"claim_match": 3, "fact_check_partner": 2, "internal_reviewer": 1})
        else:
            claim = pick(rng, claims)
            framing = pick(rng, ["satire", "news_report", "personal", "question", "asserted"])
            source = pick(rng, ["fact_check_partner", "claim_match", "internal_reviewer"])
        c = MISINFO_CLAIMS[claim]
        note_i = rng.randrange(len(c["notes"]))
        acc = min(4, c["acc"] + c["notes"][note_i][1])
        dom = c["domain"]
        if framing == "news_report":
            author = "local news outlet"
        elif framing == "satire":
            author = "satire page" if (reg != "conflicting" or chance(rng, 0.4)) else "ordinary user"
        else:
            author = pick(rng, AUTHOR_TYPES[dom])
        match_thr = pick(rng, [0.85, 0.88, 0.9])
        match = None
        if source == "claim_match":
            match = (near(rng, match_thr + 0.02, 0.025, lo=match_thr + 0.002, hi=0.99, digits=3)
                     if reg != "clear" else between(rng, match_thr + 0.01, 0.99, digits=3))
        if claim in ("outbreak_count", "postal_deadline"):
            rating = "Outdated"
        elif acc >= 3:
            rating = pick(rng, ["Missing context", "Partly false"])      # labelled although largely right
        else:
            rating = ["False", "Partly false", "Missing context"][acc]
        if reg == "conflicting" and rating in ("Missing context", "Partly false") and chance(rng, 0.4):
            rating = "False"                                              # rated harsher than the note supports
        days_to = None
        if dom == "elections" and claim != "counted_twice":
            days_to = pick(rng, [2, 4, 9, 16, 28, 45, 120])
        views = _audience(rng, 4000, 1.5)
        if reg == "conflicting" and chance(rng, 0.4):
            views = rng.randint(150000, 900000)
        posted = rand_date(rng)
        labelled = posted + timedelta(days=rng.randint(0, 3))
        return {
            "regime": reg, "claim": claim, "domain": dom, "note_variant": note_i, "claim_accuracy_latent": acc,
            "claim_harm_latent": c["harm"], "framing": framing, "author_type": author, "rating": rating,
            "label": LABEL_NAMES[rating], "reach_reduced": chance(rng, 0.8), "label_source": source,
            "partner": pick(rng, PARTNERS),
            "match_score": match, "match_threshold": match_thr, "days_to_election": days_to,
            "views": views, "shares": int(views * rng.uniform(0.005, 0.06)),
            "appellant_link": pick_w(rng, {"official": 2, "news": 2, "extra": 1, None: 3}),
            "note_shown": chance(rng, 0.75), "policy_shown": chance(rng, 0.8), "plea_variant": rng.randrange(2),
            "handle": _handle(rng), "platform": pick(rng, PLATFORMS), "appeal_id": _appeal_id(rng),
            "post_id": f"p_{hexid(rng, 10)}", "posted": posted, "labelled": labelled,
            "appeal_date": labelled + timedelta(days=pick(rng, [0, 1, 1, 2, 4])),
        }

    def _summary(self, f):
        return f"A post that {MISINFO_CLAIMS[f['claim']]['say']}."

    def _applied_by(self, f):
        if f["label_source"] == "fact_check_partner":
            return f"fact-check partner {f['partner']} (rating: {f['rating']})"
        if f["label_source"] == "claim_match":
            return (f"claim matching against an earlier {f['partner']} rating of '{f['rating']}' on a similar post, "
                    f"similarity {f['match_score']} (threshold {f['match_threshold']})")
        return "internal policy reviewer"

    def _statement(self, f, rng):
        # a joke page or a newsroom argues from framing, not from outside evidence for the claim
        link = ("" if f["framing"] in ("news_report", "satire")
                else APPELLANT_LINKS[f["domain"]].get(f["appellant_link"], ""))
        return noisy(join_sentences([FRAMING_PLEAS[f["framing"]][f["plea_variant"]], link]), rng, 0.2)

    def render(self, f, rng, surface):
        c = MISINFO_CLAIMS[f["claim"]]
        sec, pname, ptext = MISINFO_POLICY[f["domain"]]
        note = c["notes"][f["note_variant"]][0] if f["note_shown"] else None
        stmt = self._statement(f, rng)
        summary = self._summary(f)
        if surface == "json":
            return dumps(prune({
                "appeal": {"id": f["appeal_id"], "filed": f["appeal_date"].isoformat(), "statement": stmt},
                "post": {"id": f["post_id"], "author": f["handle"], "author_type": f["author_type"],
                         "posted": f["posted"].isoformat(), "summary": summary,
                         "presentation": FRAMINGS[f["framing"]], "views": f["views"], "shares": f["shares"]},
                "label": {"type": f["label"], "applied": f["labelled"].isoformat(), "applied_by": self._applied_by(f),
                          "reach_reduced": f["reach_reduced"]},
                "context": {"days_to_election": f["days_to_election"], "fact_check_note": note},
                "policy": ({"section": f"{sec} {pname}", "text": ptext + " " + MISINFO_EXEMPT}
                           if f["policy_shown"] else None),
            }))
        ds = _ds(rng)
        if surface == "policy_brief":
            parts = []
            if f["policy_shown"]:
                parts += [f"POLICY {sec} ({pname})", ptext, MISINFO_EXEMPT, ""]
            case = [f"Post {f['post_id']} by {f['handle']} ({f['author_type']}), {fmt_date(f['posted'], style=ds)}: "
                    f"{summary}", FRAMINGS[f["framing"]],
                    f"Seen {_count(f['views'], rng)} times, shared {f['shares']:,} times."]
            if f["days_to_election"] is not None:
                case.append(f"The vote is {f['days_to_election']} days away.")
            case.append(f"Label '{f['label']}' applied {fmt_date(f['labelled'], style=ds)} by {self._applied_by(f)}"
                        + ("; reach reduced." if f["reach_reduced"] else "."))
            if note:
                case.append(f"Fact-check note: {note}")
            parts += [f"CASE SUMMARY ({f['appeal_id']})", join_sentences(case), "",
                      f'Author\'s appeal: "{stmt}"']
            return "\n".join(parts)
        # appeal_form
        lines = [f"{f['platform']} | Dispute a label", f"Case {f['appeal_id']} | submitted "
                 f"{fmt_date(f['appeal_date'], style=ds)} | {f['handle']}", "",
                 f"Label on your post: {f['label']}" + (" (fewer people will see it)" if f["reach_reduced"] else ""),
                 "Tell us why this label is wrong:", f"> {stmt}", "",
                 "-- reviewer panel --", kv_lines([
                     ("Post", summary), ("Presentation", FRAMINGS[f["framing"]]),
                     ("Author type", f["author_type"]),
                     ("Reach", f"{_count(f['views'], rng)} views / {f['shares']:,} shares"),
                     ("Label source", self._applied_by(f)),
                     ("Days to the vote", f["days_to_election"]),
                     ("Fact-check note", note),
                     (f"{sec}", (ptext + " " + MISINFO_EXEMPT) if f["policy_shown"] else None),
                 ], rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]))]
        return "\n".join(lines)

    def questions(self):
        return {
            "label_ruling": choice("What should happen to the label on this post?", {
                "keep_label": "Leave the label and any reduced reach exactly as they are.",
                "soften_to_context_note": "Swap the label for a lighter context note linking to authoritative "
                                          "information, and restore normal reach.",
                "remove_label": "Take the label off and restore the post's normal reach.",
                "remove_post": "Take the post down, because a label is not enough to contain the harm it could do.",
                "send_back_for_rating": "Send the post back to the fact-checking partner for a fresh rating before "
                                        "deciding.",
            }),
            "claim_accuracy": score("Read as an ordinary viewer would read it, how accurate is what the post conveys?", [
                "False: it conveys something authoritative sources contradict.",
                "Misleading: a true fragment framed so that it suggests something false.",
                "Incomplete: partly right, but missing context that changes its meaning.",
                "Largely accurate: only minor imprecision.",
                "Accurate: consistent with current authoritative information.",
            ]),
            "harm_if_acted_on": score("If viewers acted on the post, how much harm could follow?", [
                "Negligible: little anyone would do differently.",
                "Limited: minor consequences that are easily corrected.",
                "Substantial: could change people's health choices, or whether and how they vote.",
                "Severe: could lead to physical harm or cost people their vote.",
            ]),
            "endorses_claim": noul(
                "The post puts the claim forward as its own position rather than reporting, questioning, rebutting "
                "or parodying it.",
                yes="A typical viewer would take the post as endorsing the claim.",
                no="The post plainly reports, questions, rebuts or satirises the claim instead of endorsing it."),
            "specialist_team_needed": noul(
                "This dispute should be decided by the specialist health or elections policy team rather than in "
                "the general appeals queue."),
        }


# =========================================================================== schema 5
MONET_POLICY = {
    "strong_language": "Occasional strong language is fine; frequent strong language, or any in the first 30 "
                       "seconds, limits ads.",
    "graphic": "Detailed descriptions or footage of injury or death limit ads unless presented in an educational "
               "or documentary context that does not dwell on detail.",
    "reused": "Channels must add substantial original commentary, editing or narration to other people's material "
              "to earn from it.",
    "misleading": "Titles and thumbnails must not promise something the video does not contain.",
    "sensitive": "Videos about a recent tragedy are limited for 14 days after the event unless they are purely "
                 "informational.",
}
MONET_REASON_LABEL = {"strong_language": "Strong language", "graphic": "Violent or graphic content",
                      "reused": "Reused content", "misleading": "Misleading title or thumbnail",
                      "sensitive": "Sensitive event"}
# content -> niche, reviewer description, the concern the system cites, latent real issue, latent suitability 0-3,
# how the creator can remediate, and how pervasive the format tends to be
MONET_CONTENT = {
    "true_crime": dict(niche="true crime", say="a 40-minute episode on an unsolved 1990s disappearance that reads out "
                                               "the injuries listed in the coroner's report",
                       cited="graphic", issue="graphic", suit=1, fix="trimmed the coroner-report reading"),
    "gaming_swearing": dict(niche="gaming", say="a stream replay with frequent swearing, several times within the "
                                                "first minute",
                            cited="strong_language", issue="strong_language", suit=1,
                            fix="bleeped the opening minutes"),
    "reaction_compilation": dict(niche="reaction videos", say="a compilation of other creators' clips with short "
                                                              "captions and a few seconds of the creator laughing "
                                                              "between them",
                                 cited="reused", issue="reused", suit=2, fix="recorded voice-over commentary for"),
    "flood_video": dict(niche="news commentary", say="a video about last week's flood with footage of damaged homes "
                                                     "and a thumbnail reading 'THEY KNEW'",
                        cited="sensitive", issue="sensitive", suit=1, fix="replaced the thumbnail and retitled"),
    "knife_skills": dict(niche="cooking", say="a knife-skills lesson showing how to fillet a fish",
                         cited="graphic", issue="none", suit=3, fix=None),
    "fitness_thumbnail": dict(niche="fitness", say="a 20-minute workout whose thumbnail promises 'lose 10 kg in "
                                                   "7 days'",
                              cited="misleading", issue="misleading", suit=2, fix="replaced the thumbnails on"),
    "war_documentary": dict(niche="history", say="a documentary on a 20th-century war that includes archive combat "
                                                 "footage and a brief shot of casualties",
                            cited="graphic", issue="graphic", suit=2, fix="added a warning card and cut the "
                                                                          "casualty shot from"),
    "podcast_clips": dict(niche="podcast clips", say="short clips cut from a popular podcast the creator does not "
                                                     "host, each with a one-line caption",
                          cited="reused", issue="reused", suit=2, fix="added commentary segments to"),
}
MONET_PLEAS = {
    "true_crime": ["True crime is about real cases. I read the public record, I don't sensationalise it.",
                   "Every true-crime channel covers cases like this. Mine is research-heavy and respectful."],
    "gaming_swearing": ["It's a live stream. People react in the moment, that's the whole format.",
                        "Gamers swear. My audience is adults and nobody has ever complained."],
    "reaction_compilation": ["Picking the clips and putting them together is my editing. That's original work.",
                             "Reaction channels all work like this. My captions are the commentary."],
    "flood_video": ["People needed to know what happened in the flood. I was covering the news.",
                    "The thumbnail is my opinion about the council's warnings. It's commentary."],
    "knife_skills": ["It's a cooking lesson. Nobody is in any danger, it's a fish.",
                     "Your system has confused a kitchen knife with a weapon."],
    "fitness_thumbnail": ["Every fitness channel uses thumbnails like that. The workout itself is real.",
                          "Thumbnails are meant to grab attention. The video explains the plan properly."],
    "war_documentary": ["This is a history documentary. Archive footage is how history is taught.",
                        "The footage is from public archives and the channel is educational."],
    "podcast_clips": ["Clip channels are how podcasts get discovered. The hosts benefit from it.",
                      "I choose and caption the best moments. That takes real editing work."],
}
SPREAD_RANGES = {"isolated": (0.0, 0.0), "occasional": (0.03, 0.08), "common": (0.2, 0.45), "pervasive": (0.6, 0.95)}


class MonetisationAppeal(Schema):
    name = "monetisation_appeal"
    description = "A creator appeals a decision switching off advertising on videos or on the whole channel."
    surfaces = {"json": 0.45, "creator_appeal": 0.3, "partner_escalation": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        if reg == "clear":
            if chance(rng, 0.45):
                key, spread = "knife_skills", "isolated"
                scope, warnings = pick(rng, ["channel", "videos"]), 0
            else:
                key, spread = pick(rng, ["gaming_swearing", "reaction_compilation", "podcast_clips"]), "pervasive"
                scope, warnings = "channel", rng.randint(2, 3)
            remediation = 0.0
        elif reg == "borderline":
            key = pick(rng, ["true_crime", "war_documentary", "flood_video", "fitness_thumbnail",
                             "reaction_compilation", "gaming_swearing", "podcast_clips"])
            spread = pick(rng, ["isolated", "occasional", "common"])
            scope, warnings = pick(rng, ["channel", "videos"]), pick(rng, [0, 0, 1, 2])
            remediation = pick(rng, [0.0, 0.3, 0.5, 0.8])
        else:
            key = pick(rng, list(MONET_CONTENT))
            # the penalty's reach and the problem's reach disagree
            if chance(rng, 0.5):
                spread, scope = pick(rng, ["isolated", "occasional"]), "channel"
            else:
                spread, scope = pick(rng, ["common", "pervasive"]), "videos"
            warnings = pick(rng, [0, 1, 3])
            remediation = pick(rng, [0.0, 1.0])
        c = MONET_CONTENT[key]
        if c["issue"] == "none":
            remediation = 0.0
        total = rng.randint(40, 520)
        lo, hi = SPREAD_RANGES[spread]
        flagged = rng.randint(1, 2) if spread == "isolated" else max(3, int(total * rng.uniform(lo, hi)))
        sample = pick(rng, [10, 20])
        in_sample = min(sample, flagged, max(0, round(sample * flagged / total + rng.choice([-1, 0, 0, 1]))))
        thr = 0.7
        risk = (near(rng, thr + 0.03, 0.04, lo=thr + 0.005, hi=0.99, digits=3)
                if reg == "borderline" or c["issue"] == "none" else between(rng, 0.78, 0.98, digits=3))
        cited = c["cited"]
        decided = rand_date(rng)
        if reg == "conflicting" and key == "war_documentary" and chance(rng, 0.5):
            cited = "reused"                     # the archive footage tripped the reuse detector
        return {
            "regime": reg, "content": key, "niche": c["niche"], "real_issue": c["issue"], "cited_reason": cited,
            "suitability_latent": c["suit"], "spread": spread, "scope": scope, "videos_total": total,
            "videos_flagged": flagged, "sample_checked": sample, "sample_flagged": in_sample,
            "ad_risk_score": risk, "ad_risk_threshold": thr, "prior_warnings": warnings,
            "original_share_pct": (rng.randint(4, 35) if c["issue"] == "reused" else None),
            "permission_share": (maybe(rng, pick(rng, ["2 of 9", "5 of 12", "1 of 14", "8 of 10"]), 0.5)
                                 if c["issue"] == "reused" else None),
            "remediated": int(round(flagged * remediation)), "subscribers": _audience(rng, 60000, 1.3),
            "monthly_revenue": round(lognormal_amount(rng, 1400, 0.9), -1),
            "channel": f"{rng.choice(_HANDLE_A).title()} {rng.choice(_HANDLE_B).title()} {pick(rng, ['TV', 'Studio', 'Files', 'Channel', 'Media', 'Weekly'])}",
            "manager": person(rng), "platform": pick(rng, PLATFORMS), "appeal_id": _appeal_id(rng),
            "decided": decided, "appeal_date": decided + timedelta(days=pick(rng, [1, 2, 3, 6, 10])),
            "policy_shown": chance(rng, 0.75), "tone": pick(rng, ["measured", "upset"]),
        }

    def _scope_text(self, f):
        return ("channel removed from the partner programme" if f["scope"] == "channel"
                else f"ads switched off on {f['videos_flagged']} flagged videos")

    def _share(self, f):
        return f"{f['videos_flagged']} of {f['videos_total']} videos ({round(100 * f['videos_flagged'] / f['videos_total'])}%)"

    def _statement(self, f, rng):
        c = MONET_CONTENT[f["content"]]
        s = [pick(rng, MONET_PLEAS[f["content"]])]
        if f["remediated"] and c["fix"]:
            s.append(f"Since the decision I have {c['fix']} {f['remediated']} of the {f['videos_flagged']} flagged "
                     f"videos.")
        if f["permission_share"]:
            s.append(f"I have written permission from {f['permission_share']} of the creators whose clips I use.")
        if f["tone"] == "upset":
            s.append("This is my income and nobody even watched the videos before deciding.")
        return noisy(join_sentences(s), rng, 0.2)

    def render(self, f, rng, surface):
        c = MONET_CONTENT[f["content"]]
        stmt = self._statement(f, rng)
        policy = MONET_POLICY[f["cited_reason"]] if f["policy_shown"] else None
        reason = MONET_REASON_LABEL[f["cited_reason"]]
        if surface == "json":
            return dumps(prune({
                "channel": {"name": f["channel"], "niche": f["niche"], "subscribers": f["subscribers"],
                            "videos": f["videos_total"], "earlier_monetisation_warnings": f["prior_warnings"],
                            "monthly_ad_revenue_before": money(f["monthly_revenue"], rng)},
                "decision": {"date": f["decided"].isoformat(), "reason": reason, "scope": self._scope_text(f),
                             "example_flagged": c["say"], "ad_risk_score": f["ad_risk_score"],
                             "threshold": f["ad_risk_threshold"]},
                "catalogue": {"videos_flagged": f["videos_flagged"], "flagged_share": self._share(f),
                              "manual_sample": f"{f['sample_flagged']} of {f['sample_checked']} recent uploads "
                                               f"matched on review",
                              "original_material_share_pct": f["original_share_pct"],
                              "videos_changed_since_decision": f["remediated"]},
                "appeal": {"id": f["appeal_id"], "filed": f["appeal_date"].isoformat(), "statement": stmt},
                "policy": policy,
            }))
        if surface == "creator_appeal":
            lines = [f"{f['platform']} Partner Programme | monetisation appeal {f['appeal_id']} | "
                     f"filed {fmt_date(f['appeal_date'], style=_ds(rng))}",
                     f"Channel: {f['channel']} ({f['niche']}, {_count(f['subscribers'], rng)} subscribers, "
                     f"{f['videos_total']} videos)",
                     f"Decision under appeal: {self._scope_text(f)}. Reason given: {reason}.", "",
                     "Your explanation:", f"> {stmt}", "", "-- attached by the review tool --",
                     kv_lines([("Example of flagged material", c["say"]),
                               ("Ad-risk score", f"{f['ad_risk_score']} (limit {f['ad_risk_threshold']})"),
                               ("Flagged videos", self._share(f)),
                               ("Manual sample", f"{f['sample_flagged']}/{f['sample_checked']} recent uploads matched"),
                               ("Original material share", f"{f['original_share_pct']}%"
                                if f["original_share_pct"] is not None else None),
                               ("Earlier warnings", f["prior_warnings"]),
                               ("Revenue before decision", f"{money(f['monthly_revenue'], rng)}/month"),
                               ("Guideline", policy)], rng, style=pick(rng, ["colon", "aligned", "dash"]))]
            return "\n".join(lines)
        # partner_escalation: a partner manager advocating for the creator
        praise = pick(rng, ["They're one of our fastest-growing channels in the category and have never caused "
                            "trouble before.",
                            "They've been a great partner and I'd hate to lose them over one decision.",
                            "Honestly I think the classifier got this one wrong."])
        facts = [f"Decision: {self._scope_text(f)} ({reason}).", f"Flagged example: {c['say']}.",
                 f"Ad-risk score {f['ad_risk_score']} against a limit of {f['ad_risk_threshold']}.",
                 f"{self._share(f)} flagged; a manual check found "
                 f"{f['sample_flagged']} of the {f['sample_checked']} most recent uploads matching.",
                 f"Earlier warnings on the account: {f['prior_warnings']}."]
        if f["original_share_pct"] is not None:
            facts.append(f"Original material is about {f['original_share_pct']}% of runtime.")
        return "\n".join([
            f"Subject: Please take another look - {f['channel']} ({f['appeal_id']})",
            f"From: {f['manager']}, Creator Partnerships", "To: monetisation-appeals", "",
            noisy(praise, rng, 0.15), "", bullets(facts, rng), "",
            f'Creator\'s own words: "{stmt}"'] + ([f"", f"Relevant guideline: {policy}"] if policy else []))

    def questions(self):
        return {
            "monetisation_ruling": choice("How should the monetisation appeal be decided?", {
                "keep_ads_off": "Keep advertising switched off for everything the decision covered: the problem runs "
                                "through most of the catalogue, there were earlier warnings and nothing has been changed.",
                "restore_full_ads": "Turn full advertising back on for everything: the flag was mistaken, or the "
                                    "creator has already fixed every flagged video.",
                "restore_limited_ads": "Turn ads back on only for advertisers who accept sensitive content: graphic "
                                       "or tragedy-related material presented as news, history or true crime.",
                "narrow_to_flagged_videos": "Confine the penalty to the videos at fault and restore ads everywhere "
                                            "else: a whole channel was penalised over less than a fifth of its videos.",
                "restore_after_changes": "Restore ads once the creator edits the specific material at issue (bleeping, "
                                         "retitling, adding commentary), which is not yet done for every flagged video.",
            }),
            "primary_concern": choice("Which advertiser-policy concern best describes the flagged material?", {
                "strong_language": "Frequent swearing or crude language, especially near the start of a video.",
                "graphic_or_tragic": "Detailed description or footage of injury, death or a recent disaster.",
                "reused_material": "Other people's material presented with little original contribution.",
                "misleading_packaging": "A title or thumbnail that promises something the video does not deliver.",
                "no_real_concern": "Nothing a mainstream advertiser would object to; the flag looks mistaken.",
            }),
            "advertiser_suitability": score("How comfortable would a mainstream advertiser be appearing alongside the flagged material?", [
                "Unsuitable: most brands would refuse to appear there.",
                "Limited: only brands that opt in to sensitive content would accept it.",
                "Mostly suitable: a few cautious brands might avoid it.",
                "Fully suitable: nearly any brand would accept it.",
            ]),
            "catalogue_spread": score("How widespread is the problem across the creator's catalogue?", [
                "Isolated: one video or a single thumbnail.",
                "Occasional: a handful of videos among many.",
                "Common: a sizeable share of recent uploads.",
                "Pervasive: the channel's format depends on it.",
            ]),
            "scope_proportionate": noul(
                "The reach of the penalty, channel-wide or per video, matches how widespread the problem actually is.",
                yes="A channel-wide penalty with a fifth or more of the videos flagged, or a per-video penalty with "
                    "fewer than that, on material that really has the problem.",
                no="A channel-wide penalty over less than a fifth of the videos, a per-video penalty when a fifth or "
                   "more of the catalogue has the problem, or any penalty on material with no real problem."),
        }


# =========================================================================== schema 6
IP_PROCEDURE = {
    "copyright": "A counter-notice must carry the appellant's full legal name as a signature, a street address, a "
                 "statement that the material was removed by mistake, and consent to the courts of the appellant's "
                 "area. Once a complete counter-notice is forwarded, the material is restored after 10 working days "
                 "unless the complainant shows it has started legal action.",
    "impersonation": "Accounts may not pose as another person or business in a way that could mislead. Fan and "
                     "parody accounts must say so in their name or bio. To appeal, the holder must pass an identity "
                     "check, give a verified contact email and sign a statement that the appeal is accurate.",
}
FILING_ELEMENTS = {
    "copyright": [("signature", "full legal name", "account handle only"),
                  ("address", "street address", "PO box only"),
                  ("mistake statement", "included", None),
                  ("consent to jurisdiction", "included", None)],
    "impersonation": [("identity check", "passed", "document uploaded, check not yet run"),
                      ("contact email", "verified", "given but unverified"),
                      ("accuracy statement", "signed", None)],
}
# case -> kind, reviewer description, who complained, latent complaint strength 0-3, the appellant's defence,
# and the evidence behind that defence by quality
IP_CASES = {
    "acoustic_cover": dict(kind="copyright", say="a full acoustic cover of a current chart song, performed by the "
                                                 "uploader", claimant="a music publisher", claim=2,
                           defence="the platform's music licence covers cover songs like mine",
                           ev={"strong": "a screenshot of the platform's music-licence page listing the publisher as "
                                         "a participating rights holder",
                               "partial": "a link to a forum thread saying covers are 'always fine' here"}),
    "film_review": dict(kind="copyright", say="a 14-minute film review using six short clips, about 70 seconds in "
                                              "total, each followed by the uploader's commentary",
                        claimant="the film's distributor", claim=1, defence="my review is fair use for criticism",
                        ev={"strong": "a timestamped breakdown showing commentary over or between every clip",
                            "partial": "a general statement that 'reviews are always fair use'"}),
    "reposted_illustration": dict(kind="copyright", say="a repost of a digital illustration with the artist's "
                                                        "signature cropped out and no credit",
                                  claimant="the illustrator", claim=3,
                                  defence="I found the image on a free wallpaper site",
                                  ev={"strong": "a link to the wallpaper site's page, which offers the image free "
                                                "but gives no artist credit",
                                      "partial": "the name of the wallpaper site, with no link"}),
    "baking_music": dict(kind="copyright", say="a baking tutorial with background music the uploader says came from "
                                               "a royalty-free library",
                         claimant="a rights-management agency acting for a composer", claim=1,
                         defence="I licensed the track from a royalty-free library",
                         ev={"strong": "a licence receipt from the library naming the track and the uploader's "
                                       "account",
                             "partial": "a screenshot of the library's search page showing a track with a similar "
                                        "title"}),
    "skyline_photo": dict(kind="copyright", say="a photo of a city skyline at dusk, posted as the uploader's own "
                                                "work", claimant="a professional photographer", claim=2,
                          defence="I took this photo myself from a public viewpoint",
                          ev={"strong": "the original camera file, with a capture date two years before the "
                                        "complainant's upload",
                              "partial": "a second, slightly different photo from the same spot"}),
    "match_highlights": dict(kind="copyright", say="a four-minute highlights reel of a league match recorded from a "
                                                   "TV broadcast", claimant="the broadcaster", claim=3,
                             defence="highlights are news reporting and I added my own scoreline caption",
                             ev={"strong": "a note that the reel is four minutes of a 90-minute match and carries "
                                           "no ads",
                                 "partial": "a statement that 'everyone posts highlights'"}),
    "fan_page": dict(kind="impersonation", say="an account using a singer's stage name and publicity photos, with "
                                               "'fan page' at the end of a long bio",
                     claimant="the singer's management company", claim=2, defence="it is a clearly marked fan account",
                     ev={"strong": "the bio's edit history showing 'fan page' has been there since the account opened",
                         "partial": "a screenshot of the bio taken after the takedown"}),
    "mayor_parody": dict(kind="impersonation", say="an account posting jokes in the voice of a town's mayor, using "
                                                   "her official portrait as its profile picture",
                         claimant="the mayor's office", claim=2, defence="it is obvious parody of a public official",
                         ev={"strong": "screenshots showing 'parody' in the display name since launch, with replies "
                                       "treating the posts as jokes",
                             "partial": "a note that the jokes are 'obviously fake'"}),
    "same_name_bakery": dict(kind="impersonation", say="a page for a bakery with the same name as a better-known "
                                                       "bakery in another city",
                             claimant="the other bakery", claim=1,
                             defence="my bakery is registered under that name",
                             ev={"strong": "a business-registration certificate in that name dated eight years ago",
                                 "partial": "photos of the shop front showing the name"}),
    "former_manager_gym": dict(kind="impersonation", say="a page for a gym that closed last year, now run by its "
                                                         "former manager to promote his own classes",
                               claimant="the gym's former owner", claim=2,
                               defence="I ran the page for years and built its following",
                               ev={"strong": "old messages in which the owner asked him to run the page",
                                   "partial": "a list of posts I wrote while I worked there"}),
    "copied_creator": dict(kind="impersonation", say="an account copying a craft creator's profile photo and bio and "
                                                     "messaging her followers about a 'giveaway'",
                           claimant="the craft creator", claim=3,
                           defence="it is a backup account I run for her as a friend",
                           ev={"strong": "a screenshot of a direct message, apparently from the creator, saying "
                                         "'thanks for running this'",
                               "partial": "a statement from someone who says they are the friend"}),
}


# how many users messaged the account believing it was the original, by case
CONFUSION_RANGE = {"fan_page": (0, 6), "mayor_parody": (0, 12), "same_name_bakery": (0, 3),
                   "former_manager_gym": (2, 18), "copied_creator": (6, 40)}


class TakedownCounterNotice(Schema):
    name = "takedown_counter_notice"
    description = "A counter-notice against a copyright or impersonation takedown: restore, wait, keep down, ask or refer."
    surfaces = {"json": 0.45, "counter_notice": 0.3, "queue_card": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        filing = pick_w(rng, {"complete": 2, "one_weak": 2, "one_missing": 1})
        bulk = False
        if reg == "clear":
            if chance(rng, 0.5):
                case = pick(rng, ["reposted_illustration", "match_highlights", "copied_creator"])
                ev, filing = pick(rng, ["none", "partial"]), pick(rng, ["one_weak", "one_missing"])
                verified, prior = True, rng.randint(1, 3)
            else:
                case = pick(rng, ["film_review", "same_name_bakery", "baking_music"])
                ev, filing, verified, prior, bulk = "strong", "complete", pick(rng, [True, False]), 0, True
        elif reg == "borderline":
            case = pick(rng, ["acoustic_cover", "skyline_photo", "fan_page", "former_manager_gym", "mayor_parody",
                              "film_review"])
            ev = pick_w(rng, {"partial": 3, "strong": 1, "none": 1})
            filing = pick_w(rng, {"one_weak": 3, "complete": 2, "one_missing": 1})
            verified, prior = pick(rng, [True, False]), pick(rng, [0, 0, 1])
            bulk = chance(rng, 0.3)
        else:
            case = pick(rng, list(IP_CASES))
            twist = pick(rng, ["good_defence_bad_filing", "weak_claim_big_claimant", "strong_claim_bulk_filer"])
            if twist == "good_defence_bad_filing":
                ev, filing, verified, prior = "strong", pick(rng, ["one_weak", "one_missing"]), True, 0
            elif twist == "weak_claim_big_claimant":
                case = pick(rng, ["film_review", "same_name_bakery", "baking_music", "mayor_parody"])
                ev, verified, prior = pick(rng, ["partial", "strong"]), True, rng.randint(2, 3)
            else:
                case = pick(rng, ["reposted_illustration", "match_highlights", "acoustic_cover", "fan_page"])
                ev, verified, prior, bulk = pick(rng, ["none", "partial"]), False, 0, True
        c = IP_CASES[case]
        elements = FILING_ELEMENTS[c["kind"]]
        states = ["ok"] * len(elements)
        if filing == "one_weak":
            weakable = [i for i, e in enumerate(elements) if e[2]]
            states[pick(rng, weakable)] = "weak"
        elif filing == "one_missing":
            states[rng.randrange(len(elements))] = "missing"
        if bulk:
            notices, rejected = rng.randint(150, 900), rng.randint(22, 48)
        elif c["claimant"].startswith(("a music", "the film", "the broad", "a rights", "the singer")):
            notices, rejected = rng.randint(40, 600), rng.randint(3, 15)
        else:
            notices, rejected = rng.randint(1, 3), 0
        taken_down = rand_date(rng)
        filed = taken_down + timedelta(days=rng.randint(1, 12))
        return {
            "regime": reg, "case": case, "kind": c["kind"], "complaint_strength_latent": c["claim"],
            "bulk_filer": bulk,
            "defence_evidence": ev, "filing": filing, "element_states": states, "claimant_verified": verified,
            "claimant_notices_90d": notices, "claimant_rejected_pct": rejected, "uploader_prior_ip_strikes": prior,
            "monetised": chance(rng, 0.4), "views": _audience(rng, 3000, 1.5),
            "confused_users": (rng.randint(*CONFUSION_RANGE[case]) if c["kind"] == "impersonation" else None),
            "handle": _handle(rng), "platform": pick(rng, PLATFORMS), "appeal_id": _appeal_id(rng),
            "notice_id": f"TDN-{rng.randint(100000, 999999)}", "taken_down": taken_down, "filed": filed,
            "legal_name": person(rng), "procedure_shown": chance(rng, 0.75), "tone": pick(rng, ["formal", "plain"]),
            # what the complainant attached to the notice
            "complaint_proof": pick_w(rng, {"documented": 3, "bare": 2}),
        }

    PROOF = {"documented": "the notice attaches proof of ownership (original files, registration or a signed authority)",
             "bare": "the notice names the work or identity but attaches no proof of ownership"}

    def _claimant(self, f):
        """Bulk filers are brand-protection agencies, whoever the underlying rights holder is."""
        who = IP_CASES[f["case"]]["claimant"]
        if f["bulk_filer"] and not who.startswith(("a music", "the film", "the broad", "a rights", "the singer")):
            return f"a brand-protection agency acting for {who}"
        return who

    def _elements(self, f):
        out = []
        for (label, ok, weak), st in zip(FILING_ELEMENTS[f["kind"]], f["element_states"]):
            out.append((label, ok if st == "ok" else weak if st == "weak" else "missing"))
        return out

    def _statement(self, f, rng):
        c = IP_CASES[f["case"]]
        opener = pick(rng, ["I believe this was removed by mistake: {}.", "This takedown is wrong because {}.",
                            "I dispute the complaint. In short, {}.", "The complaint is mistaken: {}."])
        s = [opener.format(c["defence"])]
        if f["defence_evidence"] != "none":
            s.append(f"I have included {c['ev'][f['defence_evidence']]}.")
        if f["tone"] == "plain":
            s.append(pick(rng, ["Please put it back.", "I don't understand why this was taken down."]))
        return noisy(join_sentences(s), rng, 0.15)

    def _claimant_line(self, f):
        ver = "verified" if f["claimant_verified"] else "unverified"
        return (f"{self._claimant(f)} ({ver}); {f['claimant_notices_90d']} notices in 90 days, "
                f"{f['claimant_rejected_pct']}% rejected or withdrawn; {self.PROOF[f['complaint_proof']]}")

    def render(self, f, rng, surface):
        c = IP_CASES[f["case"]]
        stmt = self._statement(f, rng)
        elements = self._elements(f)
        proc = IP_PROCEDURE[f["kind"]] if f["procedure_shown"] else None
        if surface == "json":
            return dumps(prune({
                "takedown": {"id": f["notice_id"], "type": f["kind"], "date": f["taken_down"].isoformat(),
                             "material": c["say"], "views_before_removal": f["views"],
                             "monetised": f["monetised"],
                             "users_who_mistook_it_for_the_original": f["confused_users"]},
                "complainant": {"who": self._claimant(f), "verified": f["claimant_verified"],
                                "proof_attached": self.PROOF[f["complaint_proof"]],
                                "notices_last_90_days": f["claimant_notices_90d"],
                                "pct_rejected_or_withdrawn": f["claimant_rejected_pct"]},
                "counter_notice": {"id": f["appeal_id"], "filed": f["filed"].isoformat(), "by": f["handle"],
                                   "statement": stmt,
                                   "elements": {k.replace(" ", "_"): v for k, v in elements}},
                "appellant_history": {"prior_ip_strikes": f["uploader_prior_ip_strikes"]},
                "procedure": proc,
            }))
        ds = _ds(rng)
        if surface == "counter_notice":
            sig = dict(elements).get("signature")
            sign = (f["legal_name"] if sig == "full legal name" else f["handle"] if sig == "account handle only"
                    else None)
            lines = [f"{'COUNTER-NOTICE' if f['kind'] == 'copyright' else 'IMPERSONATION APPEAL'} "
                     f"{f['appeal_id']} (re: {f['notice_id']})",
                     f"Filed {fmt_date(f['filed'], style=ds)} on {f['platform']} by {f['handle']}", "",
                     f"Material removed on {fmt_date(f['taken_down'], style=ds)}: {c['say']}.", "",
                     "Statement:", stmt, ""]
            lines += ["Required elements, as received:", kv_lines(elements, rng, style=pick(rng, ["dash", "colon"]))]
            if sign:
                lines += ["", f"Signed: {sign}"]
            lines += ["", "-- reviewer context --", kv_lines([
                ("Complainant", self._claimant_line(f)),
                ("Appellant's earlier IP strikes", f["uploader_prior_ip_strikes"]),
                ("Views before removal", _count(f["views"], rng)),
                ("Monetised", "yes" if f["monetised"] else "no"),
                ("Users who mistook it for the original", f["confused_users"]),
                ("Procedure", proc)], rng, style=pick(rng, ["colon", "aligned", "dash"]))]
            return "\n".join(lines)
        # queue_card
        head = (f"[IP DISPUTES | {f['kind']}] {f['appeal_id']} | takedown {f['notice_id']} "
                f"{fmt_date(f['taken_down'], style=ds)} | counter-filed {fmt_date(f['filed'], style=ds)}")
        body = kv_lines([
            ("Material", c["say"]), ("Complainant", self._claimant_line(f)),
            ("Appellant", f"{f['handle']} | earlier IP strikes {f['uploader_prior_ip_strikes']} | "
                          f"{_count(f['views'], rng)} views | {'monetised' if f['monetised'] else 'not monetised'}"),
            ("Mistaken for original", f"{f['confused_users']} users" if f["confused_users"] is not None else None),
            ("Filing", "; ".join(f"{k}: {v}" for k, v in elements)),
            ("Appellant says", f'"{stmt}"'),
            ("Procedure", proc),
        ], rng, style=pick(rng, ["aligned", "colon"]))
        return f"{head}\n{body}"

    def questions(self):
        return {
            "counter_notice_ruling": choice("What should the reviewer do with this counter-notice?", {
                "restore_now": "Restore the material or account at once, because the complaint plainly does not hold.",
                "restore_after_wait": "Forward the filing to the complainant and restore after the standard waiting "
                                      "period unless they start formal proceedings.",
                "keep_down": "Keep the material or account down, because the filing does not overcome the complaint.",
                "request_more_information": "Go back to the appellant for missing details or proof before doing "
                                            "anything else.",
                "refer_to_legal": "Hand the dispute to the legal team because of its complexity or what is at stake.",
            }),
            "complaint_strength": score("On the evidence in the file, how strong is the complainant's claim?", [
                "Baseless: the appellant's own documents (a licence, registration or original file) show the right "
                "is theirs, or an unverified complainant attaches no proof at all.",
                "Weak: the use or resemblance is slight or plainly allowed, or the complainant attaches no proof.",
                "Arguable: the complainant shows a real right, and the appellant has a reasonable but unproven answer.",
                "Strong: the material or identity is plainly the complainant's, backed by proof, and was used "
                "without leave.",
            ]),
            "defence_strength": score("How well does the appellant's defence hold up?", [
                "Fails: even if true it would not give the right to use the material or identity (free to download "
                "is not a licence, a highlights reel is not news, a friend's say-so is not permission).",
                "Thin: a defence that could work, backed only by the appellant's word.",
                "Reasonable: a defence that could work, backed by indirect or incomplete evidence.",
                "Compelling: a defence that could work, backed by documents that match this exact material.",
            ]),
            "filing_complete": noul(
                "The appellant's filing contains every element the platform's procedure requires for this kind of "
                "dispute.",
                yes="Each required element is present in a usable form.",
                no="At least one required element is missing or unusable, such as a signature given only as a handle."),
            "complaint_abuse_suspected": noul(
                "The complaint looks more like misuse of the takedown process than a genuine rights claim.",
                yes="The complainant files takedowns in bulk with a fifth or more rejected or withdrawn, or is unverified "
                    "and attaches no proof for a claim the file shows is slight.",
                no="The complainant files few notices or has few rejected, whatever the merits of this particular claim."),
        }


WORKFLOW = Workflow(
    name="content_moderation_appeals",
    title="Content moderation appeals",
    description="Trust-and-safety adjudication of user appeals against moderation actions on a fictional platform.",
    domain="Trust and safety / platform policy enforcement",
    schemas=[HarassmentRemovalAppeal(), SuspensionAppeal(), MarketplaceListingAppeal(), MisinformationLabelDispute(),
             MonetisationAppeal(), TakedownCounterNotice()],
    notes="Policy adjudication from the reviewer's side of the queue. Objectionable content is paraphrased, never "
          "reproduced; the appellant's statement is evidence, not a support thread. No reply-drafting, churn, "
          "security-alert or account-compromise questions.",
)
