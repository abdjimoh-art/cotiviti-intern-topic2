"""
Agentic Claim-Review Assistant — Cotiviti Intern Assessment, Topic 2
(Clinical Decision Making & Pattern Recognition: agentic generative AI for TPO).

An LLM agent (Groq `llama-3.3-70b-versatile`, native tool-calling) reviews a
synthetic healthcare claim. It reasons step-by-step against coding/payment
rules by calling three tools, then returns a structured verdict:

    { decision, reasons[], confidence, recommended_action }

Design choices that mirror the written report (made visible to the demo):
  1. Auditable by construction — every tool call + observation is a recorded
     step, so the chain-of-thought IS the audit trail.
  2. Confidence as a routing threshold — only high-confidence APPROVALs
     AUTO-CLEAR; everything else ROUTES TO A HUMAN. (compute_routing)
  3. Defense in depth — amount_anomaly is an INDEPENDENT statistical check that
     runs alongside the LLM, so the verdict never rests on the model alone.
  4. RAG honesty — lookup_policy is the stand-in for a production RAG retrieval
     step (same agent loop, simpler backend). The POC does NOT do real RAG.

SYNTHETIC DATA ONLY — NO PHI. Human-in-the-loop: the AI recommends, a human decides.

Phase 2 = this CLI core. Phase 3 adds the Streamlit UI on top of review_claim().

Run (CLI):
    pip install -r requirements.txt
    export GROQ_API_KEY=...        # or put it in poc/.env
    python poc/app.py             # review all 10 claims + acceptance check
    python poc/app.py CL-007      # review a single claim
"""

import json
import os
import sys
import time

import numpy as np
from groq import Groq

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Override with GROQ_MODEL env var (e.g. the lighter fallback if 70B is rate-limited).
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")  # fallback: "llama-3.1-8b-instant"
CONFIDENCE_THRESHOLD = 0.85              # routing cutoff for AUTO-CLEAR
ANOMALY_SIGMA = 3.0                      # z-score cutoff for amount_anomaly
MAX_TURNS = 6                            # safety cap on the agent loop

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_env() -> None:
    """Minimal .env loader (avoids a python-dotenv dependency)."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _load_json(name: str):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


RULES = _load_json("rules.json")
CLAIMS = _load_json("claims.json")
BILLING_HISTORY = _load_json("billing_history.json")

# Synthetic member/eligibility data (kept inline so the POC stays at 3 data
# files). check_eligibility() reads this; everyone is active except M1008.
MEMBERS = {
    "M1001": {"active": True, "covered": True},
    "M1002": {"active": True, "covered": True},
    "M1003": {"active": True, "covered": True},
    "M1004": {"active": True, "covered": True},
    "M1005": {"active": True, "covered": True},
    "M1006": {"active": True, "covered": True},
    "M1007": {"active": True, "covered": True},
    "M1008": {"active": False, "covered": False, "note": "coverage terminated 2025-12-31"},
    "M1009": {"active": True, "covered": True},
    "M1010": {"active": True, "covered": True},
}

# --------------------------------------------------------------------------- #
# Tools (plain Python functions exposed to the LLM via native tool-calling)
# --------------------------------------------------------------------------- #
def lookup_policy(cpt: str) -> dict:
    """Return every coding/payment rule that applies to a CPT code.

    Production note: this is the STAND-IN for a RAG retrieval step. In a real
    system this would semantically retrieve policy text from a vector store;
    here it is a deterministic filter over hand-authored rules.json.
    """
    matched = [r for r in RULES if r.get("cpt") == str(cpt)]
    return {"cpt": str(cpt), "rules": matched, "rule_count": len(matched)}


def check_eligibility(member_id: str) -> dict:
    """Return the member's coverage status (active / covered)."""
    rec = MEMBERS.get(member_id)
    if rec is None:
        return {"member_id": member_id, "active": False, "covered": False,
                "note": "member not found"}
    return {"member_id": member_id, **rec}


def amount_anomaly(cpt: str, billed_amount: float) -> dict:
    """INDEPENDENT statistical check: z-score of billed_amount vs the CPT's
    historical billed amounts. Flags an anomaly at > ANOMALY_SIGMA sigma.

    This runs ALONGSIDE the LLM's rule reasoning (defense in depth) — it can
    catch outliers that pass every hard rule.
    """
    hist = BILLING_HISTORY.get(str(cpt))
    if not hist:
        return {"cpt": str(cpt), "is_anomaly": False, "z_score": 0.0,
                "mean": None, "std": None, "n": 0,
                "note": "no billing history for this CPT"}
    arr = np.array(hist, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())  # population std (ddof=0)
    z = 0.0 if std == 0 else (float(billed_amount) - mean) / std
    return {
        "cpt": str(cpt),
        "billed_amount": float(billed_amount),
        "is_anomaly": abs(z) > ANOMALY_SIGMA,
        "z_score": round(z, 2),
        "mean": round(mean, 2),
        "std": round(std, 2),
        "n": int(arr.size),
        "sigma_threshold": ANOMALY_SIGMA,
    }


# JSON schemas advertised to the model (OpenAI/Groq tool-calling format).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Look up all coding/payment rules for a CPT code. "
                           "Returns rules with fields like sex, min_age, "
                           "allowed_icd, and max_billed_amount.",
            "parameters": {
                "type": "object",
                "properties": {"cpt": {"type": "string", "description": "CPT code, e.g. '99213'"}},
                "required": ["cpt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": "Check whether a member's coverage is active and the service is covered.",
            "parameters": {
                "type": "object",
                "properties": {"member_id": {"type": "string"}},
                "required": ["member_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "amount_anomaly",
            "description": "Independent statistical check: is the billed amount a "
                           "z-score outlier (>3 sigma) versus this CPT's billing history?",
            "parameters": {
                "type": "object",
                "properties": {
                    "cpt": {"type": "string"},
                    "billed_amount": {"type": "number"},
                },
                "required": ["cpt", "billed_amount"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "lookup_policy": lookup_policy,
    "check_eligibility": check_eligibility,
    "amount_anomaly": amount_anomaly,
}

# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""\
You are a payment-integrity claim reviewer for a health plan. You review a
single synthetic medical claim and decide whether to APPROVE it or
FLAG_FOR_REVIEW. All data is synthetic; there is no PHI.

You MUST use your tools before deciding. For the claim, call EACH of these
tools exactly once:
  1. lookup_policy(cpt)                      -> the rules for this CPT
  2. check_eligibility(member_id)            -> coverage status
  3. amount_anomaly(cpt, billed_amount)      -> independent statistical check

Then evaluate EVERY rule returned by lookup_policy against the claim, ONE AT A
TIME. Do not skip any rule and do not approve on a general impression. For each
rule, determine pass or violation by its category:
  - "sex_restriction": VIOLATION unless claim.patient_sex == rule.sex.
  - "age_range": VIOLATION unless claim.patient_age >= rule.min_age
    (and <= rule.max_age when rule.max_age is present).
  - "allowed_dx": check whether the claim's icd string appears EXACTLY in
    rule.allowed_icd. If claim.icd is NOT one of those exact strings it is a
    VIOLATION — a diagnosis that is merely "related" or "plausible" but not in
    the list does NOT pass. (e.g. an asthma code J45.909 is NOT in a list of
    gastrointestinal codes, so it is a violation.)
  - "max_amount": VIOLATION unless claim.billed_amount <= rule.max_billed_amount.
Also evaluate two more signals:
  - eligibility: VIOLATION if check_eligibility is not active or not covered.
  - amount_anomaly: an INDEPENDENT statistical violation if is_anomaly == true,
    even when every hard rule passes (defense in depth).

DECISION RULE (apply mechanically):
  - decision = "FLAG_FOR_REVIEW" if ANY rule check is a violation, OR the member
    is ineligible, OR amount_anomaly fired.
  - decision = "APPROVE" only if every rule passes, the member is eligible, and
    there is no anomaly.

CONFIDENCE: a calibrated float 0..1 — this score DRIVES ROUTING, so do not
inflate it. Calibrate it from the evidence:
  - >= 0.9 for clear-cut decisions: a clean approval where every signal is
    comfortably clear (amount_anomaly |z_score| < 2), or a clear-cut violation.
  - ~0.75 when you APPROVE (no rule violated, eligible, is_anomaly == false) but
    a signal is borderline — in particular when amount_anomaly's |z_score| is
    ELEVATED (between 2 and 3, i.e. unusually high yet below the 3σ cutoff). The
    claim still passes, but the elevated amount is genuine uncertainty: add a
    reason noting it. Keep this low (~0.75) on purpose so a borderline-but-clean
    claim can be routed to a human by the confidence threshold even though
    nothing is outright flagged. A borderline approval must NOT read as 0.9.

When finished, respond with ONLY a JSON object (no markdown, no prose):
{{
  "rule_checks": [
    {{"rule_id": "R-005", "category": "allowed_dx", "status": "pass" | "violation",
      "detail": "claim.icd 'J45.909' is not in allowed_icd [...] -> violation"}}
    // one entry per rule from lookup_policy, PLUS one for eligibility
    // (rule_id "eligibility", category "eligibility") and one for the anomaly
    // check (rule_id "amount_anomaly", category "anomaly"). Never use null ids.
  ],
  "decision": "APPROVE" | "FLAG_FOR_REVIEW",
  "reasons": ["short bullet for each violation citing its rule_id/signal", ...],
  "confidence": 0.0-1.0,
  "recommended_action": "one-sentence human-in-the-loop recommendation"
}}
If decision is APPROVE, reasons may be a single bullet like "all checks passed".
Every violation reason must cite its specific rule_id (e.g. "R-005") or signal
(e.g. "amount_anomaly z=5.82").
"""


def _create_with_retry(client: Groq, **kwargs):
    """One call to Groq with a single backoff on rate-limit, for demo robustness."""
    for attempt in range(2):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # groq.RateLimitError and friends
            if attempt == 0 and "rate" in str(e).lower():
                time.sleep(3)
                continue
            raise


def _parse_decision(text: str) -> dict:
    """Extract the final JSON verdict from the model's text (tolerates fences)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return {
        "decision": "FLAG_FOR_REVIEW",
        "reasons": ["agent did not return parseable JSON; routing to human by default"],
        "confidence": 0.0,
        "recommended_action": "Manual review required (agent output unparseable).",
    }


def compute_routing(decision: str, confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """Confidence-as-a-threshold routing (the report's core recommendation).

    Only a high-confidence APPROVE is auto-cleared; everything else goes to a
    human. The cutoff is explicit and shown (and adjustable) in the UI.
    """
    if decision == "APPROVE" and confidence >= threshold:
        return "AUTO_CLEAR"
    return "ROUTE_TO_HUMAN"


def review_claim(claim: dict, client: Groq, on_step=None):
    """Run the agent loop on one claim.

    Returns (result, steps):
      result = {decision, reasons, confidence, recommended_action, routing}
      steps  = ordered list of {kind, ...} records for the UI / audit trail.
    on_step(step) is called as each step happens (used by the Streamlit UI).
    """
    steps = []

    def emit(step):
        steps.append(step)
        if on_step:
            on_step(step)

    # Don't leak the human-readable _note into the model's context.
    claim_public = {k: v for k, v in claim.items() if not k.startswith("_")}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Review this claim:\n" + json.dumps(claim_public, indent=2)},
    ]

    for _ in range(MAX_TURNS):
        resp = _create_with_retry(
            client, model=MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto", temperature=0, max_tokens=900,
        )
        msg = resp.choices[0].message

        # Re-attach the assistant turn (with any tool_calls) to the history.
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if msg.tool_calls:
            for c in msg.tool_calls:
                name = c.function.name
                try:
                    args = json.loads(c.function.arguments or "{}")
                except Exception:
                    args = {}
                fn = TOOL_DISPATCH.get(name)
                observation = fn(**args) if fn else {"error": f"unknown tool {name}"}
                emit({"kind": "tool", "tool": name, "args": args, "observation": observation})
                messages.append({
                    "role": "tool", "tool_call_id": c.id, "name": name,
                    "content": json.dumps(observation),
                })
            continue  # let the model see the observations and keep going

        # No tool calls => this is the final verdict.
        result = _parse_decision(msg.content)
        result["confidence"] = float(result.get("confidence", 0.0) or 0.0)

        # Deterministic guardrail: a clean APPROVE whose amount is elevated but
        # below the 3σ cutoff (2 ≤ |z| ≤ 3) carries genuine uncertainty, so cap
        # its confidence under the auto-clear threshold. This makes pillar ②
        # (confidence → routing) reproducible regardless of the LLM's own score.
        # Use the agent's amount_anomaly observation if it called the tool;
        # otherwise compute it directly from the claim so the cap applies every run.
        anom = next((s["observation"] for s in steps
                     if s["kind"] == "tool" and s["tool"] == "amount_anomaly"), None)
        if anom is None:
            anom = amount_anomaly(str(claim["cpt"]), claim["billed_amount"])
        if result.get("decision") == "APPROVE":
            z = abs(anom.get("z_score", 0.0))
            if 2.0 <= z <= ANOMALY_SIGMA:
                result["confidence"] = min(result["confidence"], 0.75)

        result["routing"] = compute_routing(result.get("decision", ""), result["confidence"])
        emit({"kind": "final", "result": result, "raw": msg.content})
        return result, steps

    # Hit the turn cap without a final answer -> fail safe to human.
    result = {
        "decision": "FLAG_FOR_REVIEW",
        "reasons": [f"agent did not converge within {MAX_TURNS} turns"],
        "confidence": 0.0,
        "recommended_action": "Manual review required.",
        "routing": "ROUTE_TO_HUMAN",
    }
    emit({"kind": "final", "result": result, "raw": ""})
    return result, steps


# --------------------------------------------------------------------------- #
# CLI harness + acceptance check
# --------------------------------------------------------------------------- #
EXPECTED = {
    "CL-001": "APPROVE", "CL-002": "APPROVE", "CL-003": "APPROVE",
    "CL-004": "APPROVE", "CL-005": "FLAG_FOR_REVIEW", "CL-006": "FLAG_FOR_REVIEW",
    "CL-007": "FLAG_FOR_REVIEW", "CL-008": "FLAG_FOR_REVIEW",
    "CL-009": "FLAG_FOR_REVIEW", "CL-010": "APPROVE",
}
EXPECTED_ANOMALY = {"CL-007"}  # the only claim whose amount_anomaly must fire
# Claims that should APPROVE but route to a human on CONFIDENCE ALONE (no flag) —
# this is what makes pillar ② (confidence-as-a-threshold) visibly do work.
EXPECTED_SUBCUTOFF_ROUTE = {"CL-010"}


def _print_steps(steps):
    for s in steps:
        if s["kind"] == "tool":
            obs = s["observation"]
            if s["tool"] == "lookup_policy":
                summary = f"{obs.get('rule_count')} rule(s) for CPT {obs.get('cpt')}"
            elif s["tool"] == "check_eligibility":
                summary = f"active={obs.get('active')} covered={obs.get('covered')}"
            elif s["tool"] == "amount_anomaly":
                summary = (f"z={obs.get('z_score')} (mean={obs.get('mean')}, "
                           f"std={obs.get('std')}) anomaly={obs.get('is_anomaly')}")
            else:
                summary = json.dumps(obs)
            print(f"    -> tool: {s['tool']}({s['args']})  ::  {summary}")
        else:
            r = s["result"]
            print(f"    -> VERDICT: {r['decision']}  conf={r['confidence']:.2f}  "
                  f"routing={r['routing']}")
            for reason in r.get("reasons", []):
                print(f"         - {reason}")
            print(f"       action: {r.get('recommended_action', '')}")


def _anomaly_fired(steps) -> bool:
    for s in steps:
        if s["kind"] == "tool" and s["tool"] == "amount_anomaly":
            if s["observation"].get("is_anomaly"):
                return True
    return False


def main():
    _load_env()
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. See poc/.env.example.")
        return 1
    client = Groq()

    target = sys.argv[1] if len(sys.argv) > 1 else None
    claims = [c for c in CLAIMS if c["claim_id"] == target] if target else CLAIMS
    if target and not claims:
        print(f"No such claim: {target}")
        return 1

    passes = 0
    checks = 0
    for claim in claims:
        cid = claim["claim_id"]
        print(f"\n=== {cid}  CPT {claim['cpt']}  ICD {claim['icd']}  "
              f"{claim['patient_sex']}/{claim['patient_age']}y  "
              f"${claim['billed_amount']:.2f}  ({claim.get('_note', '')}) ===")
        result, steps = review_claim(claim, client)
        _print_steps(steps)

        # Acceptance checks (only meaningful when running the full set).
        exp = EXPECTED.get(cid)
        if exp:
            checks += 1
            ok = result["decision"] == exp
            anom_ok = _anomaly_fired(steps) == (cid in EXPECTED_ANOMALY)
            # For the sub-cutoff claim, the verdict must be APPROVE yet routing
            # must still be ROUTE_TO_HUMAN purely because confidence < threshold.
            route_ok = (cid not in EXPECTED_SUBCUTOFF_ROUTE) or (
                result["decision"] == "APPROVE"
                and result["confidence"] < CONFIDENCE_THRESHOLD
                and result.get("routing") == "ROUTE_TO_HUMAN")
            passes += 1 if (ok and anom_ok and route_ok) else 0
            verdict = "PASS" if (ok and anom_ok and route_ok) else "FAIL"
            detail = "" if ok else f" (expected {exp}, got {result['decision']})"
            if not anom_ok:
                detail += " (anomaly-fire mismatch)"
            if not route_ok:
                detail += (f" (expected sub-cutoff route to human, got "
                           f"conf={result['confidence']:.2f}/{result.get('routing')})")
            print(f"    [{verdict}]{detail}")
        time.sleep(0.3)  # be gentle with the free-tier rate limit

    if checks:
        print(f"\n================  {passes}/{checks} claims matched expected  ================")
        return 0 if passes == checks else 2
    return 0


# --------------------------------------------------------------------------- #
# Streamlit demo UI (Phase 3). Renders on top of review_claim(); the CLI above
# is unchanged. The three report pillars are made visible on screen:
#   (1) the ordered tool-call cards ARE the audit trail / chain-of-thought,
#   (2) a labeled, adjustable confidence cutoff drives AUTO-CLEAR vs ROUTE,
#   (3) amount_anomaly is shown as an independent second signal + chart.
# --------------------------------------------------------------------------- #
def run_streamlit_app():
    import altair as alt
    import pandas as pd
    import streamlit as st
    from scipy.stats import norm

    st.set_page_config(page_title="Agentic Claim-Review Assistant",
                       page_icon="🩺", layout="wide")
    _load_env()

    @st.cache_resource
    def _client():
        return Groq()

    # ---- Header + standing disclaimer ----
    st.title("🩺 Agentic Claim-Review Assistant")
    st.caption("Cotiviti Intern Assessment · Topic 2 — agentic generative AI for "
               "Treatment / Payment / Operations")
    st.warning(
        "**Synthetic data only — no PHI.**  Human-in-the-loop: the AI *recommends*, "
        "a human *decides*.  `lookup_policy` is a stand-in for a production RAG "
        "retrieval step (the POC does not do real RAG).", icon="⚠️")

    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it to poc/.env (or your shell), then reload.")
        st.stop()

    # ---- Sidebar: claim picker + the routing threshold control ----
    with st.sidebar:
        st.header("⚙️ Configuration")
        labels = {c["claim_id"]: f"{c['claim_id']} · CPT {c['cpt']} · {c.get('_note', '')[:34]}"
                  for c in CLAIMS}
        selected_id = st.selectbox("Sample claim", [c["claim_id"] for c in CLAIMS],
                                   format_func=lambda x: labels[x])
        threshold = st.slider("AUTO-CLEAR confidence cutoff", 0.50, 0.99,
                              CONFIDENCE_THRESHOLD, 0.01,
                              help="Only APPROVE verdicts at/above this confidence are "
                                   "auto-cleared; everything else routes to a human.")
        st.markdown(f"**Model:** `{MODEL}`")
        st.markdown(f"**Anomaly cutoff:** > {ANOMALY_SIGMA:g}σ (z-score)")
        st.divider()
        st.caption("Pillars shown below: ① tool-call audit trail · "
                   "② confidence → routing · ③ independent anomaly signal.")

    claim = next(c for c in CLAIMS if c["claim_id"] == selected_id)

    # ---- Selected-claim card ----
    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"Claim {claim['claim_id']}")
        meta = {k: v for k, v in claim.items() if not k.startswith("_")}
        st.dataframe(pd.DataFrame(meta.items(), columns=["field", "value"]),
                     hide_index=True, use_container_width=True)
    with right:
        st.subheader("Scenario (not shown to the agent)")
        st.info(claim.get("_note", "—"))

    # ---- Run the agent on click; cache the result so slider reruns are free ----
    if st.button("🔍 Review claim", type="primary", use_container_width=True):
        try:
            with st.spinner("Agent is reviewing the claim… (Groq + 3 tool calls)"):
                result, steps = review_claim(claim, _client())
            st.session_state["review"] = {"claim_id": selected_id, "result": result, "steps": steps}
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg or "429" in msg:
                st.error("Groq rate limit reached (free tier). Wait a moment, or set "
                         "`GROQ_MODEL=llama-3.1-8b-instant` to use the lighter fallback model.")
            else:
                st.error(f"Agent call failed: {msg}")
            st.stop()

    review = st.session_state.get("review")
    if not review or review["claim_id"] != selected_id:
        st.info("Pick a claim and press **Review claim** to run the agent.")
        return
    result, steps = review["result"], review["steps"]

    # ---- (1) Audit trail: one card per tool call + observation ----
    st.subheader("① Agent reasoning — audit trail")
    for s in steps:
        if s["kind"] != "tool":
            continue
        obs = s["observation"]
        with st.container(border=True):
            st.markdown(f"**🔧 `{s['tool']}`**  ·  args `{json.dumps(s['args'])}`")
            if s["tool"] == "lookup_policy":
                rules = obs.get("rules", [])
                st.caption(f"Retrieved {obs.get('rule_count', 0)} policy rule(s) "
                           f"for CPT {obs.get('cpt')}  (RAG stand-in)")
                if rules:
                    st.dataframe(
                        pd.DataFrame([{"rule_id": r["rule_id"], "category": r["category"],
                                       "detail": r["description"]} for r in rules]),
                        hide_index=True, use_container_width=True)
            elif s["tool"] == "check_eligibility":
                st.caption(f"active = **{obs.get('active')}**  ·  covered = "
                           f"**{obs.get('covered')}**" + (f"  ·  {obs['note']}" if obs.get("note") else ""))
            elif s["tool"] == "amount_anomaly":
                st.caption(f"z = **{obs.get('z_score')}**  (mean {obs.get('mean')}, "
                           f"std {obs.get('std')}, n={obs.get('n')})  ·  "
                           f"anomaly = **{obs.get('is_anomaly')}**")

    # ---- (2) Verdict + confidence-as-routing ----
    decision = result.get("decision", "FLAG_FOR_REVIEW")
    confidence = float(result.get("confidence", 0.0))
    routing = compute_routing(decision, confidence, threshold)

    st.subheader("② Verdict & routing")
    # High-contrast verdict banner (legible on a screen recording): green for
    # APPROVE, red reserved for FLAG.
    bg, icon, label = (("#157347", "✅", "APPROVE") if decision == "APPROVE"
                       else ("#C0392B", "🚩", "FLAG FOR REVIEW"))
    st.markdown(
        f"<div style='background:{bg};color:#fff;padding:14px 18px;border-radius:8px;"
        f"font-size:1.5rem;font-weight:800;letter-spacing:.5px;'>{icon}&nbsp;&nbsp;{label}</div>",
        unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Confidence: {confidence:.0%}**")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(f"AUTO-CLEAR cutoff = {threshold:.0%}")
    with c2:
        if routing == "AUTO_CLEAR":
            st.success(f"**Routing: AUTO-CLEAR**\n\nHigh-confidence approval "
                       f"({confidence:.0%} ≥ {threshold:.0%} cutoff) — no human needed.")
        elif decision != "APPROVE":
            st.error("**Routing: ROUTE TO HUMAN**\n\nClaim was flagged — "
                     "routing to a human reviewer.")
        else:
            st.error(f"**Routing: ROUTE TO HUMAN**\n\nApproved, but confidence "
                     f"({confidence:.0%}) is below the auto-clear cutoff "
                     f"({threshold:.0%}) — routing to a human reviewer.")

    st.markdown("**Reasons**")
    for r in result.get("reasons", []):
        st.markdown(f"- {r}")
    st.markdown(f"**Recommended action:** {result.get('recommended_action', '—')}")

    if result.get("rule_checks"):
        # Sanitize: the eligibility / anomaly checks have no rule_id, which would
        # otherwise render as literal "None" in the table.
        clean = [{"rule_id": (c.get("rule_id") or c.get("category") or "—"),
                  "category": (c.get("category") or "—"),
                  "status": c.get("status", "—"),
                  "detail": c.get("detail", "")} for c in result["rule_checks"]]
        with st.expander("Per-rule check detail"):
            st.dataframe(pd.DataFrame(clean), hide_index=True, use_container_width=True)

    # ---- (3) Independent anomaly signal + chart ----
    anom = next((s["observation"] for s in steps
                 if s["kind"] == "tool" and s["tool"] == "amount_anomaly"), None)
    st.subheader("③ Independent statistical signal (runs alongside the LLM)")
    if not anom or anom.get("n", 0) == 0:
        st.caption("No billing history available for this CPT.")
        return

    z = anom.get("z_score", 0.0)
    m1, m2, m3 = st.columns(3)
    m1.metric("This claim", f"${claim['billed_amount']:,.0f}")
    m2.metric("History mean", f"${anom['mean']:,.0f}")
    m3.metric("z-score", f"{z:.2f}", delta=("ANOMALY" if anom.get("is_anomaly") else "normal"),
              delta_color=("inverse" if anom.get("is_anomaly") else "off"))

    hist = BILLING_HISTORY.get(str(claim["cpt"]), [])
    dfh = pd.DataFrame({"amount": hist})
    bars = alt.Chart(dfh).mark_bar(opacity=0.8, color="#4C78A8").encode(
        x=alt.X("amount:Q", bin=alt.Bin(maxbins=18),
                title=f"Historical billed $ for CPT {claim['cpt']} (n={len(hist)})"),
        y=alt.Y("count():Q", title="claims"))
    marker = alt.Chart(pd.DataFrame({"amount": [claim["billed_amount"]]})).mark_rule(
        color="#E45756", size=3).encode(x="amount:Q")
    st.altair_chart(bars + marker, use_container_width=True)

    tail = float(norm.sf(abs(z)))  # one-sided tail probability under a normal fit
    if anom.get("is_anomaly"):
        st.error(f"🚩 Amount is **{z:.1f}σ** above the mean — a >~{ANOMALY_SIGMA:g}σ outlier "
                 f"(tail probability ≈ {tail:.1e}). This fires even though the hard $ cap passed.")
    else:
        st.caption(f"Amount is within normal range ({z:+.2f}σ, tail prob ≈ {tail:.2f}); "
                   f"the anomaly signal does not fire.")


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_in_streamlit():
    run_streamlit_app()
elif __name__ == "__main__":
    sys.exit(main())
