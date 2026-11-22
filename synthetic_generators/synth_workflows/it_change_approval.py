"""IT change approval: should a planned change to production systems go ahead, and on what terms?

The REFERENCE module: copy its structure when adding a workflow (see README.md). Each schema samples a
`regime` first ('clear' / 'borderline' / 'conflicting') and lets it steer the factors, so most cases carry
thresholds that are nearly met, signals that disagree, or a key fact that is missing.
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
    money,
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

# (service, tier 1 = most critical, owning team, rough daily users)
SERVICES = [
    ("payments-gateway", 1, "Payments Platform", 180000), ("order-api", 1, "Commerce Core", 240000),
    ("sso-idp", 1, "Identity", 310000), ("search-indexer", 2, "Discovery", 90000),
    ("mobile-push", 2, "Mobile", 150000), ("warehouse-etl", 2, "Data Platform", 400),
    ("pricing-engine", 2, "Commerce Core", 60000), ("hr-portal", 3, "People Systems", 3200),
    ("reporting-dashboards", 3, "BI", 900), ("internal-wiki", 4, "Workplace IT", 2500),
    ("build-cache", 4, "Developer Productivity", 700), ("label-printing", 3, "Fulfilment Systems", 350),
]
CHANGE_KINDS = {
    "config_flag": "toggle a configuration flag",
    "app_release": "deploy a new application release",
    "infra_upgrade": "upgrade the underlying cluster nodes",
    "network": "change load-balancer routing rules",
    "db_parameter": "tune database connection-pool parameters",
    "os_patch": "apply the monthly OS patch bundle",
    "cert_rotation": "rotate the TLS certificate",
    "dependency_bump": "upgrade a shared client library",
}


def _service(rng, tier_weights=(3, 3, 2, 2)):
    tier = pick(rng, [1, 2, 3, 4], tier_weights)
    return pick(rng, [s for s in SERVICES if s[1] == tier])


# =========================================================================== schema 1
class NormalChangeCab(Schema):
    name = "normal_change_cab"
    description = "A normal change request on the change advisory board's agenda: approve, condition, defer or reject."
    surfaces = {"json": 0.5, "itsm_form": 0.25, "cab_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        svc, tier, team, users = _service(rng)
        kind = pick(rng, list(CHANGE_KINDS))
        window_min = pick(rng, [60, 90, 120, 180, 240])
        duration = near(rng, window_min * 0.5, window_min * 0.3, lo=10, integer=True)
        if reg == "clear":
            good = chance(rng, 0.5)
            tests = "staging_full" if good else pick(rng, ["unit_only", "none"])
            rollback = "automated_tested" if good else pick(rng, ["vague", "none"])
            rollback_min = near(rng, 15, 10, lo=2, integer=True) if good else None
            prior_fail = 0 if good else pick(rng, [2, 3, 4])
            peer, owner = ("approved", "approved") if good else ("not_requested", pick(rng, ["pending", "missing"]))
            in_window = good
            peak_days = None if good else pick(rng, [1, 2, 3])
            self_risk = "low" if good else pick(rng, ["medium", "high"])
        else:
            tests = pick_w(rng, {"staging_full": 3, "staging_partial": 4, "unit_only": 2, "none": 1})
            rollback = pick_w(rng, {"automated_tested": 2, "documented_untested": 4, "vague": 2, "none": 1})
            # borderline: the backout nearly fills the time the change leaves in the window
            slack = window_min - duration
            rollback_min = near(rng, slack, max(5, slack * 0.35), lo=3, integer=True)
            prior_fail = pick(rng, [0, 1, 1, 2, 3])
            peer = pick_w(rng, {"approved": 5, "pending": 3, "not_requested": 1})
            owner = pick_w(rng, {"approved": 4, "pending": 4, "missing": 1})
            in_window = chance(rng, 0.6)
            peak_days = pick(rng, [None, None, 2, 5, 9])
            self_risk = pick(rng, ["low", "low", "medium", "high"])
            if reg == "conflicting":
                # the requester's own view and the evidence disagree, and one fact goes missing
                if tier <= 2:
                    self_risk = "low"
                    prior_fail = pick(rng, [2, 3, 4])
                else:
                    self_risk = "high"
                    tests = "staging_full"
                if chance(rng, 0.6):
                    rollback = None
                    rollback_min = None
        if rollback == "none":
            rollback_min = None
        start = rand_ts(rng, hours=(1, 5) if in_window else (9, 17))
        return {
            "regime": reg, "service": svc, "tier": tier, "team": team, "daily_users": users, "change_kind": kind,
            "change_id": ident(rng, "CHG", 7, ""), "requester": person(rng),
            "implementer_experience": pick(rng, ["senior", "regular", "regular", "new_to_team", "contractor"]),
            "start": start, "window_min": window_min, "planned_duration_min": duration,
            "in_maintenance_window": in_window, "peak_event_in_days": peak_days,
            "test_evidence": maybe(rng, tests, 0.1), "rollback": rollback, "rollback_estimate_min": rollback_min,
            "monitoring_plan": pick_w(rng, {"dashboards_and_alerts": 4, "manual_checks": 3, "none_stated": 2}),
            "self_assessed_risk": self_risk, "failed_changes_90d": prior_fail,
            "concurrent_changes_same_window": pick(rng, [0, 0, 0, 1, 2]),
            "peer_review": peer, "service_owner_approval": owner,
        }

    def _notes(self, f, rng):
        s = [f"Need to {CHANGE_KINDS[f['change_kind']]} on {f['service']}."]
        if f["test_evidence"] == "staging_partial":
            s.append(pick(rng, ["Smoke tests passed in staging; the full regression suite did not finish before submission.",
                                "Validated the main path in staging, edge cases not covered yet."]))
        elif f["test_evidence"] == "unit_only":
            s.append("Covered by unit tests; no staging run because staging is shared this week.")
        elif f["test_evidence"] == "none":
            s.append(pick(rng, ["Small change, did not seem worth a staging run.", "No test run yet."]))
        if f["rollback"] == "documented_untested":
            s.append("Backout: redeploy previous version per runbook (not rehearsed).")
        elif f["rollback"] == "vague":
            s.append(pick(rng, ["If anything goes wrong we will revert.", "Backout: revert as needed."]))
        if f["failed_changes_90d"] >= 2:
            s.append(pick(rng, [f"Note the last {f['failed_changes_90d']} changes here needed hotfixes afterwards.",
                                "Previous attempts on this service had issues, this one is simpler."]))
        if f["peak_event_in_days"]:
            s.append(f"Marketing campaign starts in {f['peak_event_in_days']} day(s), so we want this in before then.")
        if f["concurrent_changes_same_window"]:
            s.append(f"{f['concurrent_changes_same_window']} other change(s) are booked in the same window.")
        return noisy(join_sentences(s), rng)

    def render(self, f, rng, surface):
        notes = self._notes(f, rng)
        start = f["start"]
        if surface == "json":
            return dumps(prune({
                "change": {"id": f["change_id"], "service": f["service"], "service_tier": f["tier"],
                           "type": f["change_kind"], "requested_by": f["requester"],
                           "implementer_experience": f["implementer_experience"],
                           "self_assessed_risk": f["self_assessed_risk"], "requester_notes": notes},
                "schedule": {"start_utc": fmt_ts(start, style="iso"), "window_minutes": f["window_min"],
                             "planned_duration_minutes": f["planned_duration_min"],
                             "inside_maintenance_window": f["in_maintenance_window"],
                             "days_to_peak_event": f["peak_event_in_days"],
                             "other_changes_same_window": f["concurrent_changes_same_window"]},
                "plan": {"test_evidence": f["test_evidence"], "backout": f["rollback"],
                         "backout_estimate_minutes": f["rollback_estimate_min"], "monitoring": f["monitoring_plan"]},
                "approvals": {"peer_review": f["peer_review"], "service_owner": f["service_owner_approval"]},
                "history": {"failed_changes_90d": f["failed_changes_90d"], "service_daily_users": f["daily_users"]},
            }))
        if surface == "itsm_form":
            body = kv_lines([
                ("Change", f["change_id"]), ("Configuration item", f"{f['service']} (tier {f['tier']})"),
                ("Category", f["change_kind"].replace("_", " ")), ("Requested by", f["requester"]),
                ("Planned start", fmt_ts(start, rng)), ("Window", f"{f['window_min']} min"),
                ("Planned duration", f"{f['planned_duration_min']} min"),
                ("Maintenance window", "yes" if f["in_maintenance_window"] else "no (business hours)"),
                ("Risk (requester)", f["self_assessed_risk"]),
                ("Test evidence", show(f["test_evidence"], rng, lambda v: v.replace("_", " "))),
                ("Backout plan", show(f["rollback"], rng, lambda v: v.replace("_", " "))),
                ("Backout time", show(f["rollback_estimate_min"], rng, lambda v: f"~{v} min")),
                ("Monitoring", f["monitoring_plan"].replace("_", " ")),
                ("Peer review", f["peer_review"].replace("_", " ")),
                ("Service owner approval", f["service_owner_approval"]),
                ("Failed changes on this service (90d)", f["failed_changes_90d"]),
            ], rng, skip_none=False)
            return f"CHANGE REQUEST -- normal change\n{body}\n\nJustification / notes:\n{notes}"
        # cab_email
        chair = person(rng)
        lines = [
            f"Subject: CAB {fmt_date(start.date(), rng)} -- {f['change_id']} {f['service']}",
            f"From: {f['requester']}", f"To: cab-{f['team'].lower().replace(' ', '-')}@corp.example", "",
            f"Hi {chair.split()[0]},", "",
            f"Putting {f['change_id']} on this week's CAB list. {notes}", "",
            bullets([
                f"Service: {f['service']} (tier {f['tier']}, ~{f['daily_users']:,} users/day)",
                f"When: {fmt_ts(start, rng)}, {f['planned_duration_min']} of {f['window_min']} min window"
                + ("" if f["in_maintenance_window"] else ", outside the maintenance window"),
                f"Testing: {show(f['test_evidence'], rng, lambda v: v.replace('_', ' '))}",
                f"Rollback: {show(f['rollback'], rng, lambda v: v.replace('_', ' '))}"
                + (f", about {f['rollback_estimate_min']} min" if f["rollback_estimate_min"] else ""),
                f"Approvals: peer review {f['peer_review'].replace('_', ' ')}, owner {f['service_owner_approval']}",
                f"My risk rating: {f['self_assessed_risk']}",
            ], rng), "", "Thanks,", f["requester"].split()[0],
        ]
        return "\n".join(lines)

    def questions(self):
        return {
            "cab_decision": choice("What should the change advisory board decide about this request?", {
                "approve": "Approve the change for the proposed window as submitted.",
                "approve_with_conditions": "Approve it only once stated conditions are met, such as more testing, "
                                           "another window or an extra approver.",
                "defer": "Take no decision now and put it back on a later board's agenda.",
                "reject": "Refuse it in this form; the plan has to be reworked before it can come back.",
            }),
            "failure_likelihood": score("How likely is this change to fail or need backing out?", [
                "Remote: a routine change with a clean record behind it.",
                "Unlikely: some chance of trouble, but it is well covered.",
                "Possible: there are open questions a reviewer would want answered first.",
                "Likely: several warning signs in the plan or the service's history.",
                "Near certain: as written, the plan will probably not go cleanly.",
            ]),
            "impact_if_failed": score("If the change went wrong, how much would the business notice?", [
                "Barely: an internal tool that few people depend on.",
                "Contained: one team or a secondary feature is degraded for a while.",
                "Broad: many users or an important capability is affected.",
                "Severe: revenue, a contractual service or safety is disrupted.",
            ]),
            "backout_fits_window": noul(
                "The backout plan would restore service within the approved window if the change failed.",
                yes="The backout steps are concrete and would fit in the time the change leaves in the window.",
                no="The backout is missing, vague or untested, or it would overrun the window."),
            "needs_further_signoff": noul("This change needs sign-off beyond the approvals already recorded."),
        }


# =========================================================================== schema 2
class EmergencyChangeRetro(Schema):
    name = "emergency_change_retro"
    description = "An emergency change already made during an incident, submitted for retrospective approval."
    surfaces = {"json": 0.5, "incident_timeline": 0.25, "review_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        svc, tier, team, users = _service(rng, (4, 3, 2, 1))
        sev = pick_w(rng, {"SEV1": 2, "SEV2": 4, "SEV3": 3, None: 1})
        linked = sev is not None and chance(rng, 0.85)
        approver = pick_w(rng, {"on_call_manager": 4, "incident_commander": 3, "nobody": 2, "director_verbal": 1})
        doc = near(rng, 0.6, 0.35, lo=0.05, hi=1.0)
        verified = pick_w(rng, {"yes": 4, "partial": 2, "no": 1, None: 1})
        outcome = pick_w(rng, {"resolved": 5, "partially_resolved": 3, "worsened_briefly": 1})
        repeat = pick(rng, [0, 0, 1, 2, 3, 5])
        next_window_h = pick(rng, [2, 6, 10, 20, 40])
        if reg == "clear":
            if chance(rng, 0.5):
                sev, linked, approver, doc, verified, outcome = "SEV1", True, "incident_commander", 0.9, "yes", "resolved"
            else:
                sev, linked, approver, doc, verified = "SEV3", False, "nobody", 0.2, "no"
        elif reg == "conflicting":
            # a good outcome achieved the wrong way, or a proper process that made things worse
            if chance(rng, 0.5):
                outcome, approver, doc = "resolved", "nobody", near(rng, 0.3, 0.2, lo=0.05, hi=1.0)
            else:
                outcome, approver, verified = "worsened_briefly", "incident_commander", "partial"
            if chance(rng, 0.5):
                linked = False
        started = rand_ts(rng)
        return {"regime": reg, "service": svc, "tier": tier, "team": team, "daily_users": users,
                "incident_severity": sev, "incident_linked": linked,
                "incident_id": ident(rng, "INC", 6, "") if linked else None,
                "change_id": ident(rng, "ECHG", 6, ""), "engineer": person(rng),
                "verbal_approver": approver, "doc_completeness": doc, "post_change_verification": verified,
                "outcome": outcome, "team_emergency_changes_qtr": repeat,
                "hours_to_next_window": next_window_h, "started": started,
                "minutes_to_apply": pick(rng, [4, 9, 15, 25, 40]),
                "action": pick(rng, ["raised the connection-pool limit", "rolled back the release from Tuesday",
                                     "disabled the recommendation widget flag", "failed traffic over to the secondary region",
                                     "restarted the stuck consumer group with a larger heap",
                                     "manually edited the routing table on the edge proxies"])}

    def _fields(self, f):
        d = f["doc_completeness"]
        return {"what_changed": f["action"],
                "why": "restore service during the incident" if d > 0.25 else None,
                "backout_steps": "revert the setting / redeploy previous build" if d > 0.45 else None,
                "verification": f["post_change_verification"] if d > 0.35 else None,
                "approver_recorded": f["verbal_approver"] if d > 0.3 else None}

    def render(self, f, rng, surface):
        fl = self._fields(f)
        t0 = f["started"]
        if surface == "json":
            return dumps(prune({
                "change": {"id": f["change_id"], "service": f["service"], "service_tier": f["tier"],
                           "implemented_by": f["engineer"], "implemented_at": fmt_ts(t0, style="iso"),
                           "minutes_to_apply": f["minutes_to_apply"], "record": prune(fl)},
                "incident": ({"id": f["incident_id"], "severity": f["incident_severity"]}
                             if f["incident_linked"] else {"linked": False}),
                "aftermath": {"outcome": f["outcome"], "post_change_verification": f["post_change_verification"]},
                "context": {"hours_to_next_scheduled_window": f["hours_to_next_window"],
                            "team_emergency_changes_this_quarter": f["team_emergency_changes_qtr"],
                            "service_daily_users": f["daily_users"]},
            }))
        if surface == "incident_timeline":
            ts = [t0 - timedelta(minutes=rng.randint(20, 90)), t0, t0 + timedelta(minutes=f["minutes_to_apply"]),
                  t0 + timedelta(minutes=f["minutes_to_apply"] + rng.randint(5, 40))]
            ev = [f"alerting fired on {f['service']}" + (f" ({f['incident_severity']} declared)" if f["incident_linked"] else ""),
                  f"{f['engineer']} {f['action']}" + ("" if f["verbal_approver"] == "nobody"
                                                     else f" (ok'd by {f['verbal_approver'].replace('_', ' ')})"),
                  "change applied",
                  {"resolved": "error rates back to baseline", "partially_resolved": "errors reduced but not gone",
                   "worsened_briefly": "brief spike in 5xx after the change, then recovery"}[f["outcome"]]]
            log = "\n".join(f"{fmt_ts(t, style='log')}  {e}" for t, e in zip(ts, ev))
            tail = kv_lines([("Retro record", f["change_id"]), ("Incident", f["incident_id"]),
                             ("Verification", f["post_change_verification"]),
                             ("Emergency changes by team this quarter", f["team_emergency_changes_qtr"]),
                             ("Next scheduled window was in", f"{f['hours_to_next_window']} h")], rng, skip_none=False)
            return f"Timeline ({f['service']}, tier {f['tier']}):\n{log}\n\n{tail}"
        # review_memo
        para = [f"Retrospective approval requested for {f['change_id']} on {f['service']} (tier {f['tier']}).",
                f"During {'incident ' + f['incident_id'] if f['incident_linked'] else 'an unlinked degradation'}, "
                f"{f['engineer']} {f['action']} at {fmt_ts(t0, rng)}.",
                ("No one approved it at the time." if f["verbal_approver"] == "nobody"
                 else f"Verbal go-ahead from the {f['verbal_approver'].replace('_', ' ')}.")]
        if fl["backout_steps"] is None:
            para.append("The record has no backout section.")
        if fl["verification"] is None:
            para.append("Verification is not documented.")
        para.append({"resolved": "Service recovered afterwards.",
                     "partially_resolved": "Things improved but the incident stayed open for another hour.",
                     "worsened_briefly": "There was a short spike in errors right after the change."}[f["outcome"]])
        if f["team_emergency_changes_qtr"] >= 3:
            para.append(f"This is the team's {f['team_emergency_changes_qtr']}th emergency change this quarter.")
        para.append(f"The next normal window was {f['hours_to_next_window']} hours away.")
        return "MEMO -- emergency change review\n\n" + noisy(join_sentences(para), rng, 0.1)

    def questions(self):
        return {
            "retro_verdict": choice("How should the review board close out this emergency change?", {
                "ratify": "Accept the change as justified and close the record.",
                "ratify_with_actions": "Accept it, and assign follow-ups such as completing the record or a proper test.",
                "refer_to_problem_review": "Accept the fix but open a problem review into why emergencies keep recurring.",
                "record_as_breach": "Treat it as an unauthorised change that bypassed the process and escalate it.",
            }),
            "emergency_was_warranted": noul(
                "Waiting for the normal approval path would have caused material harm.",
                yes="The live impact was serious enough that a delay of hours mattered.",
                no="The change could reasonably have waited for an expedited or normal review."),
            "record_completeness": score("How complete is the record of what was changed and why?", [
                "Skeletal: little beyond a title.",
                "Patchy: key parts such as backout or verification are missing.",
                "Adequate: a reviewer can follow what happened.",
                "Thorough: steps, timing, verification and approvals are all written down.",
            ]),
            "procedure_deviation": score("How far did the team depart from the emergency-change procedure?", [
                "Not at all; every step of the emergency path was followed.",
                "Slightly; a step was informal or late, but it happened.",
                "Significantly; an approval or a verification step was skipped.",
                "Entirely; the change went around the process altogether.",
            ]),
            "change_made_things_worse": noul(
                "The change itself caused disruption beyond the original problem."),
        }


# =========================================================================== schema 3
class DbSchemaMigration(Schema):
    name = "db_schema_migration"
    description = "A production database schema migration awaiting approval: run it, move it, phase it or redesign it."
    surfaces = {"json": 0.45, "migration_ticket": 0.3, "chat_thread": 0.25}

    OPS = {
        "add_nullable_column": ("ALTER TABLE {t} ADD COLUMN notes_v2 text NULL;", "additive"),
        "add_column_with_default": ("ALTER TABLE {t} ADD COLUMN region_code varchar(8) NOT NULL DEFAULT 'NA';", "rewrite"),
        "create_index": ("CREATE INDEX idx_{t}_created ON {t} (created_at);", "index"),
        "drop_column": ("ALTER TABLE {t} DROP COLUMN legacy_status;", "destructive"),
        "change_column_type": ("ALTER TABLE {t} ALTER COLUMN amount TYPE numeric(14,2);", "rewrite"),
        "backfill": ("UPDATE {t} SET tenant_id = o.tenant_id FROM orgs o WHERE {t}.org_id = o.id;", "bulk_write"),
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        op = pick(rng, list(self.OPS))
        table_name = pick(rng, ["orders", "ledger_entries", "sessions", "shipments", "audit_log", "user_prefs"])
        rows_m = pick(rng, [0.2, 3, 18, 60, 240, 900])
        window = pick(rng, [60, 120, 240])
        est = near(rng, window * 0.8, window * 0.5, lo=1, integer=True)
        online_tool = chance(rng, 0.5)
        tested = pick_w(rng, {"prod_size_copy": 3, "partial_copy": 3, "dev_only": 2, None: 1})
        backup_h = pick(rng, [2, 12, 30, 80, None])
        down = pick_w(rng, {"tested": 2, "written_untested": 3, "none": 2})
        compat = pick_w(rng, {"backward_compatible": 4, "needs_coordinated_deploy": 3, None: 1})
        dba = pick_w(rng, {"approved": 3, "pending": 3, "not_requested": 2})
        peak = chance(rng, 0.3)
        if reg == "clear":
            if chance(rng, 0.5):
                op, rows_m, est, tested, backup_h, down, compat, dba, peak = (
                    "add_nullable_column", 3, 5, "prod_size_copy", 2, "tested", "backward_compatible", "approved", False)
            else:
                op, rows_m, tested, backup_h, down, dba = "drop_column", 240, "dev_only", None, "none", "not_requested"
                est = int(window * 1.6)
        elif reg == "conflicting":
            # tested thoroughly but the estimate blows the window, or quick but with no way back
            if chance(rng, 0.5):
                tested, dba, est = "prod_size_copy", "approved", near(rng, window * 1.2, window * 0.2, lo=1, integer=True)
            else:
                est, down, backup_h = near(rng, window * 0.3, window * 0.15, lo=1, integer=True), "none", None
        return {"regime": reg, "engine": pick(rng, ["PostgreSQL 15", "PostgreSQL 16", "MySQL 8.0", "SQL Server 2022"]),
                "operation": op, "operation_class": self.OPS[op][1], "table": table_name, "rows_millions": rows_m,
                "window_min": window, "estimated_runtime_min": est, "online_schema_tool": online_tool,
                "tested_on": tested, "backup_verified_hours_ago": backup_h, "down_migration": down,
                "app_compatibility": compat, "dba_review": dba, "during_peak_hours": peak,
                "replica_count": pick(rng, [0, 1, 2, 3]), "author": person(rng),
                "ticket": ident(rng, "DBCHG", 5, "-"), "start": rand_ts(rng, hours=(13, 23) if peak else (1, 5))}

    def render(self, f, rng, surface):
        sql = self.OPS[f["operation"]][0].format(t=f["table"])
        if surface == "json":
            return dumps(prune({
                "migration": {"ticket": f["ticket"], "author": f["author"], "engine": f["engine"],
                              "statement": sql, "table": f["table"], "table_rows_millions": f["rows_millions"],
                              "uses_online_schema_tool": f["online_schema_tool"]},
                "schedule": {"start_utc": fmt_ts(f["start"], style="iso"), "window_minutes": f["window_min"],
                             "estimated_runtime_minutes": f["estimated_runtime_min"],
                             "overlaps_peak_traffic": f["during_peak_hours"]},
                "safety": {"rehearsed_on": f["tested_on"], "backup_verified_hours_ago": f["backup_verified_hours_ago"],
                           "down_migration": f["down_migration"], "read_replicas": f["replica_count"]},
                "compatibility": {"application": f["app_compatibility"]},
                "review": {"dba": f["dba_review"]},
            }))
        if surface == "migration_ticket":
            head = kv_lines([
                ("Ticket", f["ticket"]), ("Author", f["author"]), ("Engine", f["engine"]),
                ("Table", f"{f['table']} (~{f['rows_millions']}M rows)"),
                ("Start", fmt_ts(f["start"], rng)), ("Window", f"{f['window_min']} min"),
                ("Estimated runtime", f"{f['estimated_runtime_min']} min"),
                ("Online schema tool", "yes" if f["online_schema_tool"] else "no"),
                ("Rehearsal", show(f["tested_on"], rng, lambda v: v.replace("_", " "))),
                ("Last verified backup", show(f["backup_verified_hours_ago"], rng, lambda v: f"{v} h ago")),
                ("Down migration", f["down_migration"].replace("_", " ")),
                ("App compatibility", show(f["app_compatibility"], rng, lambda v: v.replace("_", " "))),
                ("DBA review", f["dba_review"].replace("_", " ")),
            ], rng, skip_none=False)
            return f"{head}\n\nMigration:\n    {sql}"
        # chat_thread
        dba_name, lead = person(rng), person(rng)
        msgs = [(f["author"], f"planning to run `{sql}` on {f['table']} ({f['rows_millions']}M rows), "
                              f"window {f['window_min']}m, estimate {f['estimated_runtime_min']}m"
                              + (" using the online schema tool" if f["online_schema_tool"] else ""))]
        if f["tested_on"] == "prod_size_copy":
            msgs.append((f["author"], "rehearsed on a prod-size restore, timings came from that"))
        elif f["tested_on"] == "partial_copy":
            msgs.append((f["author"], "ran it on a 10% sample, scaled the estimate up linearly"))
        elif f["tested_on"] == "dev_only":
            msgs.append((f["author"], "only tried it on dev so far"))
        if f["backup_verified_hours_ago"] is None:
            msgs.append((lead, "when was the last restore test?"))
        else:
            msgs.append((lead, f"backup verified {f['backup_verified_hours_ago']}h ago, ok"))
        if f["down_migration"] == "none":
            msgs.append((f["author"], "no down migration, it's one-way"))
        if f["app_compatibility"] == "needs_coordinated_deploy":
            msgs.append((f["author"], "app release 4.2 has to go out right after"))
        msgs.append((dba_name, {"approved": "looked at it, fine from my side", "pending": "haven't had time to look yet",
                                "not_requested": "first I'm hearing of this tbh"}[f["dba_review"]]))
        if f["during_peak_hours"]:
            msgs.append((lead, "that start time is during peak traffic, right?"))
        t = f["start"] - timedelta(days=1)
        out = []
        for who, text in msgs:
            t += timedelta(minutes=rng.randint(1, 25))
            out.append(f"[{t.strftime('%H:%M')}] {who.split()[0].lower()}: {noisy(text, rng, 0.1)}")
        return f"#db-changes ({f['engine']})\n" + "\n".join(out)

    def questions(self):
        return {
            "migration_verdict": choice("What should happen to this migration request?", {
                "run_as_planned": "Run it in the requested slot exactly as written.",
                "move_off_peak": "Keep the migration but move it to a quieter, longer slot.",
                "run_in_phases": "Split it into smaller steps, each verified before the next.",
                "redesign_first": "Send it back; the approach itself needs to change before it runs.",
            }),
            "data_loss_exposure": score("How exposed is existing data if this migration goes wrong?", [
                "None: purely additive, nothing existing is touched.",
                "Small: routine backups would recover it with little loss.",
                "Material: recovery would be slow or lose recent writes.",
                "Serious: data could be destroyed or silently corrupted.",
            ]),
            "table_blocking_risk": score("How likely is the migration to block reads or writes on the live table?", [
                "Negligible blocking.",
                "Brief stalls that users would hardly notice.",
                "Noticeable blocking for part of the run.",
                "Prolonged blocking amounting to an outage.",
            ]),
            "fits_in_window": noul(
                "The migration, including checks afterwards, fits inside the planned window.",
                yes="The runtime estimate, backed by a realistic rehearsal, leaves room for verification.",
                no="The estimate is unreliable or leaves no time to verify and back out."),
            "dba_must_review": noul("A database administrator must review this migration before it runs."),
        }


# =========================================================================== schema 4
class FreezeException(Schema):
    name = "freeze_exception"
    description = "A request to make a production change during a change freeze: grant, narrow, hold or decline."
    surfaces = {"json": 0.45, "request_email": 0.3, "exception_form": 0.25}

    JUST = {
        "revenue_blocking_bug": "a defect is blocking checkout for part of our customers",
        "regulatory_deadline": "a reporting format required by the regulator from the 1st of next month",
        "vendor_end_of_support": "the vendor ends support for the current version during the freeze",
        "cost_saving": "moving batch jobs to cheaper instances would save money every day",
        "feature_launch": "a partner launch is scheduled and marketing has committed to the date",
        "convenience": "the team would rather not carry the change over the freeze",
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        svc, tier, team, users = _service(rng)
        just = pick(rng, list(self.JUST))
        days_left = pick(rng, [2, 4, 7, 12, 20])
        size = pick_w(rng, {"one_line_config": 3, "small_code_change": 4, "multi_service_release": 2})
        undo = pick_w(rng, {"feature_flag": 3, "redeploy": 4, "irreversible": 1})
        sponsor = pick_w(rng, {"vp": 2, "director": 3, "none": 3})
        precedent = pick(rng, [0, 1, 2, 4, 7])
        tested = pick_w(rng, {"full": 3, "partial": 3, "none": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                just, size, undo, sponsor, tested = "revenue_blocking_bug", "one_line_config", "feature_flag", "vp", "full"
            else:
                just, size, undo, sponsor, days_left = "convenience", "multi_service_release", "redeploy", "none", 2
        elif reg == "conflicting":
            if chance(rng, 0.5):
                just, size, undo = "revenue_blocking_bug", "multi_service_release", "irreversible"
            else:
                just, sponsor, size, tested = "cost_saving", "vp", "one_line_config", "full"
        # the claimed cost of waiting follows the final justification; often it is simply not quantified
        base_loss = {"revenue_blocking_bug": 18000, "cost_saving": 900, "feature_launch": 4000}.get(just)
        loss = None if base_loss is None or chance(rng, 0.2) else round(base_loss * rng.uniform(0.2, 2.5), -2)
        return {"regime": reg, "service": svc, "tier": tier, "team": team, "justification": just,
                "claimed_daily_loss_usd": loss, "freeze": pick(rng, ["quarter-end close", "holiday peak", "annual audit"]),
                "freeze_days_left": days_left, "change_size": size, "undo_method": undo, "exec_sponsor": sponsor,
                "exceptions_granted_this_freeze": precedent, "testing": tested, "customer_facing": tier <= 2 or chance(rng, 0.3),
                "requester": person(rng), "request_id": ident(rng, "FRZ", 4, "-"),
                "tone": pick(rng, ["measured", "measured", "insistent"]), "date": rand_ts(rng).date()}

    def render(self, f, rng, surface):
        why = self.JUST[f["justification"]]
        loss = money(f["claimed_daily_loss_usd"], rng) + "/day" if f["claimed_daily_loss_usd"] else None
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "requested_by": f["requester"], "service": f["service"],
                            "service_tier": f["tier"], "customer_facing": f["customer_facing"],
                            "justification": why, "claimed_cost_of_waiting": loss,
                            "executive_sponsor": None if f["exec_sponsor"] == "none" else f["exec_sponsor"]},
                "change": {"size": f["change_size"], "undo_method": f["undo_method"], "testing": f["testing"]},
                "freeze": {"name": f["freeze"], "days_remaining": f["freeze_days_left"],
                           "exceptions_granted_so_far": f["exceptions_granted_this_freeze"]},
            }))
        if surface == "request_email":
            push = ("We really cannot wait on this. " if f["tone"] == "insistent" else "")
            body = [f"Subject: Freeze exception request {f['request_id']} -- {f['service']}", "",
                    "Hello change management,", "",
                    noisy(join_sentences([
                        f"I'd like an exception to the {f['freeze']} freeze for {f['service']}.",
                        f"The reason: {why}.", push,
                        f"We estimate waiting costs about {loss}." if loss else "",
                        f"The change is a {f['change_size'].replace('_', ' ')}"
                        + {"feature_flag": " behind a feature flag.", "redeploy": " and can be undone by redeploying.",
                           "irreversible": "; it cannot easily be undone once applied."}[f["undo_method"]],
                        {"full": "Fully tested in staging.", "partial": "Partly tested.",
                         "none": "Not yet tested, we will test on the day."}[f["testing"]],
                        (f"Our {f['exec_sponsor'].upper() if f['exec_sponsor'] == 'vp' else f['exec_sponsor']} supports this."
                         if f["exec_sponsor"] != "none" else ""),
                        f"The freeze has {f['freeze_days_left']} days to go."]), rng),
                    "", "Regards,", f["requester"]]
            return "\n".join(body)
        rows = [("Request", f["request_id"]), ("Date", fmt_date(f["date"], rng)), ("Service", f"{f['service']} (tier {f['tier']})"),
                ("Customer facing", "Y" if f["customer_facing"] else "N"), ("Freeze", f["freeze"]),
                ("Days of freeze left", f["freeze_days_left"]), ("Business reason", why),
                ("Cost of waiting", loss), ("Change size", f["change_size"].replace("_", " ")),
                ("Undo", f["undo_method"].replace("_", " ")), ("Testing", f["testing"]),
                ("Executive sponsor", f["exec_sponsor"]), ("Exceptions already granted this freeze", f["exceptions_granted_this_freeze"])]
        return "FREEZE EXCEPTION FORM\n" + kv_lines(rows, rng, skip_none=False)

    def questions(self):
        return {
            "exception_decision": choice("How should the freeze-exception request be answered?", {
                "grant": "Allow the change during the freeze as requested.",
                "grant_narrowed": "Allow only a reduced version, such as the flag flip without the rest of the release.",
                "hold_until_freeze_ends": "Refuse for now and schedule it for the first window after the freeze.",
                "decline": "Refuse outright; the request does not justify a change in this form at all.",
            }),
            "business_case_strength": score("How strong is the business case for breaking the freeze?", [
                "Weak: preference or convenience.",
                "Modest: a real benefit, but one that can wait.",
                "Strong: a measurable loss or a fixed external deadline.",
                "Compelling: waiting would cause serious, documented harm.",
            ]),
            "waiting_causes_harm": noul(
                "Holding the change until the freeze lifts would cause material harm.",
                yes="A concrete loss, penalty or broken commitment follows from waiting.",
                no="Waiting is inconvenient or costs little."),
            "oversight_needed": score("How much extra oversight should accompany the change if it goes ahead?", [
                "Normal on-call coverage is enough.",
                "A named approver watches the rollout.",
                "A live call with the service owner for the whole change.",
                "Executive sign-off plus a staffed call and a rehearsed backout.",
            ]),
            "decision_owner": choice("Who should make the final call on this exception?", {
                "change_manager": "The change manager alone; it is a routine exception.",
                "service_director": "The director who owns the affected service.",
                "exception_board": "The full exception board, because the case is contested.",
                "executive_office": "An executive, because of the money or risk involved.",
            }),
        }


# =========================================================================== schema 5
class IacPlanReview(Schema):
    name = "iac_plan_review"
    description = "An infrastructure-as-code plan waiting to be applied to a cloud environment."
    surfaces = {"json": 0.45, "plan_output": 0.3, "pull_request": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        env = pick_w(rng, {"production": 5, "staging": 3, "shared_services": 2})
        add, change, destroy = rng.randint(0, 12), rng.randint(0, 9), pick(rng, [0, 0, 0, 1, 2, 5])
        stateful = destroy > 0 and chance(rng, 0.45)
        replace = pick(rng, [0, 0, 1, 2, 3])
        cost = round(near(rng, 0, 2500), -1)
        policy = pick_w(rng, {"passed": 5, "warnings": 3, "failed": 1, "not_run": 1})
        plan_age_h = pick(rng, [0.5, 3, 20, 70])
        drift = chance(rng, 0.25)
        approvals = pick(rng, [0, 1, 1, 2])
        purpose = pick(rng, ["resize the worker node pool", "move logs to a cheaper storage class",
                             "add a read replica for reporting", "upgrade the managed Kubernetes version",
                             "tighten the default network rules", "enable encryption on the queue",
                             "consolidate two load balancers into one"])
        if reg == "clear":
            if chance(rng, 0.5):
                env, destroy, stateful, replace, policy, plan_age_h, drift, approvals = "staging", 0, False, 0, "passed", 0.5, False, 2
            else:
                env, destroy, stateful, policy, drift = "production", 3, True, "failed", True
        elif reg == "conflicting":
            if chance(rng, 0.5):
                destroy, stateful, policy, approvals = 1, True, "passed", 2       # clean checks, but a data store goes
            else:
                cost, purpose, destroy, stateful = round(rng.uniform(4000, 15000), -2), "resize the worker node pool", 0, False
        return {"regime": reg, "environment": env, "to_add": add, "to_change": change, "to_destroy": destroy,
                "destroys_stateful": stateful, "forced_replacements": replace, "monthly_cost_delta_usd": cost,
                "policy_checks": policy, "plan_age_hours": plan_age_h, "drift_detected": drift,
                "approvals": approvals, "purpose": purpose, "author": person(rng),
                "workspace": f"{pick(rng, ['core', 'data', 'edge', 'ml'])}-{env[:4]}",
                "pr": rng.randint(1200, 9800),
                "stateful_resource": pick(rng, ["aws_db_instance.reporting", "google_storage_bucket.exports",
                                                 "aws_ebs_volume.queue_data", "azurerm_postgresql_server.main"])}

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "plan": {"workspace": f["workspace"], "environment": f["environment"], "purpose": f["purpose"],
                         "add": f["to_add"], "change": f["to_change"], "destroy": f["to_destroy"],
                         "forced_replacements": f["forced_replacements"],
                         "destroyed_resources": [f["stateful_resource"]] if f["destroys_stateful"] else None,
                         "generated_hours_ago": f["plan_age_hours"]},
                "checks": {"policy": f["policy_checks"], "drift_detected": f["drift_detected"]},
                "cost": {"monthly_delta_usd": f["monthly_cost_delta_usd"]},
                "review": {"author": f["author"], "approvals": f["approvals"], "pull_request": f["pr"]},
            }))
        if surface == "plan_output":
            lines = [f"Workspace: {f['workspace']}   ({f['environment']})",
                     f"Plan generated {f['plan_age_hours']} hours ago by {f['author']}",
                     f"Change purpose (from the PR): {f['purpose']}", ""]
            if f["forced_replacements"]:
                lines.append(f"  # {f['forced_replacements']} resource(s) must be replaced")
            if f["destroys_stateful"]:
                lines.append(f"  - {f['stateful_resource']} will be destroyed")
            lines.append(f"Plan: {f['to_add']} to add, {f['to_change']} to change, {f['to_destroy']} to destroy.")
            if f["drift_detected"]:
                lines.append("Note: objects have changed outside of this tool since the last run.")
            lines += ["", f"Policy checks: {f['policy_checks']}",
                      f"Estimated monthly cost change: {'+' if f['monthly_cost_delta_usd'] >= 0 else ''}"
                      f"{money(f['monthly_cost_delta_usd'], rng)}"]
            return "\n".join(lines)
        desc = noisy(join_sentences([f"This PR will {f['purpose']} in {f['environment']}.",
                                     pick(rng, ["Straightforward change.", "Should be a no-op for traffic.",
                                                "Please review carefully, first time touching this module.", ""])]), rng)
        return "\n".join([
            f"## PR #{f['pr']}: {f['purpose']}", "", desc, "",
            table(["check", "result"], [["plan", f"+{f['to_add']} ~{f['to_change']} -{f['to_destroy']}"],
                                        ["replacements", f["forced_replacements"]], ["policy", f["policy_checks"]],
                                        ["drift", "yes" if f["drift_detected"] else "no"],
                                        ["cost/month", money(f["monthly_cost_delta_usd"], rng)],
                                        ["plan age", f"{f['plan_age_hours']}h"]], rng),
            "", f"Approvals: {f['approvals']}"
            + (f"\nDestroys: `{f['stateful_resource']}`" if f["destroys_stateful"] else "")])

    def questions(self):
        return {
            "apply_verdict": choice("What should happen with this plan?", {
                "apply": "Apply the plan as it stands.",
                "apply_after_fixes": "Apply once specific small problems in it are corrected.",
                "split_change": "Break it into smaller applies that can each be checked.",
                "replan_first": "Generate a fresh plan first; this one may not match the live environment.",
                "do_not_apply": "Do not apply it; the change is unsafe or unjustified in this form.",
            }),
            "destroys_data_store": noul(
                "Applying this plan would destroy or replace a resource that holds data."),
            "outage_likelihood": score("How likely is applying this plan to cause a user-visible outage?", [
                "Very unlikely.",
                "Unlikely, with a small blip at most.",
                "A real possibility.",
                "Likely, for part of the service.",
                "Almost certain, for a major service.",
            ]),
            "spend_change_justified": noul(
                "The change in monthly spend is justified by what the plan delivers.",
                yes="Spend falls or stays flat, or rises in keeping with the purpose: up to $3,000 a month for added or "
                    "resized capacity, up to $300 for an upgrade, a network-rule or an encryption change.",
                no="Spend rises although the purpose (cheaper storage, consolidation) should save money, or it rises "
                   "by more than those amounts."),
            "review_depth": score("How closely should a second engineer review this plan before it is applied?", [
                "A glance is enough.",
                "An ordinary line-by-line review.",
                "A careful review by someone who knows this stack well.",
                "A walkthrough with the owning team before anything runs.",
            ]),
        }


WORKFLOW = Workflow(
    name="it_change_approval",
    title="IT change approval",
    description="Whether a planned change to production IT systems should go ahead, and on what terms.",
    domain="IT operations / change management",
    schemas=[NormalChangeCab(), EmergencyChangeRetro(), DbSchemaMigration(), FreezeException(), IacPlanReview()],
    notes="Change requests are human-authored plans BEFORE (or, for emergencies, just after) implementation; "
          "no AI-agent runs, no security alerts, no vendor bills.",
)
