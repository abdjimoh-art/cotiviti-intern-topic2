# Agentic Claim-Review Assistant

**Cotiviti Intern Assessment — Topic 2: Clinical Decision Making & Pattern
Recognition** (agentic generative AI for Treatment / Payment / Operations).

An LLM agent (Groq `llama-3.3-70b-versatile`, **native tool-calling**) reviews a
synthetic healthcare claim. It reasons step-by-step against coding/payment rules
by calling three tools, then returns a structured verdict:

```
{ decision: APPROVE | FLAG_FOR_REVIEW, reasons[], confidence, recommended_action }
```

A high-confidence clean claim is **AUTO-CLEARED**; anything flagged or uncertain
is **ROUTED TO A HUMAN**.

> ⚠️ **Synthetic data only — no PHI.** Human-in-the-loop: the AI *recommends*, a
> human *decides*. `lookup_policy` is a stand-in for a production RAG retrieval
> step — the POC does **not** do real RAG.

---

## Why this design (the demo *is* the argument)

The companion report argues for "agentic claim review as a human-in-the-loop
system with a full audit trail." The POC makes that argument visible on screen
through three pillars:

| # | Pillar | Where you see it |
|---|--------|------------------|
| ① | **Auditable by construction** — every tool call + observation is a recorded step, so the chain-of-thought *is* the audit trail. | "Agent reasoning — audit trail" cards |
| ② | **Confidence as a routing threshold** — the report's fix for LLM overconfidence (Tian et al., 2025): calibrate confidence and route on it. Only high-confidence APPROVALs auto-clear. | The adjustable AUTO-CLEAR cutoff + routing banner |
| ③ | **Defense in depth** — `amount_anomaly` is an *independent* classical (z-score) check that runs alongside the LLM, so the verdict never rests on the model alone. | "Independent statistical signal" chart |

---

## Architecture

```
Streamlit UI  (poc/app.py)
  └─ pick a synthetic claim → "Review claim"
       └─ Agent loop (Groq + native tool-use)
            ├─ lookup_policy(cpt)             → coding/payment rules for that CPT   (RAG stand-in)
            ├─ check_eligibility(member_id)   → active / covered
            └─ amount_anomaly(cpt, amount)    → z-score vs per-CPT billing history  (independent signal)
       └─ strict JSON: { decision, reasons[], confidence, recommended_action }
  └─ render: step cards · verdict banner · confidence→routing · anomaly chart
```

No LangChain/LlamaIndex — the agent loop is hand-written to show first principles.

---

## Run it

```bash
# 1. install (Python 3.11+)
pip install -r poc/requirements.txt

# 2. add your free Groq key  (https://console.groq.com/keys)
cp poc/.env.example poc/.env          # then edit poc/.env and set GROQ_API_KEY

# 3a. verify the key + tool-calling round-trip
python poc/smoke_test.py

# 3b. run the agent over all 10 claims in the terminal (with acceptance check)
python poc/app.py                     # or:  python poc/app.py CL-007  (one claim)

# 3c. launch the demo UI
streamlit run poc/app.py              # opens http://localhost:8501
```

The app reads `GROQ_API_KEY` from the environment (or `poc/.env`). The key is
**never** committed — see *Security* below.

> Groq's free tier has a daily token cap. If the 70B model is rate-limited, set
> `GROQ_MODEL=llama-3.1-8b-instant` to switch to the lighter fallback (the demo
> is tuned for the 70B model, which reasons more reliably over the rules).

---

## What's in the box

```
poc/
  app.py                 # single-file Streamlit app + agent loop + 3 tools + CLI harness
  smoke_test.py          # one Groq tool-call to verify your key
  rules.json             # 15 hand-authored coding/payment rules
  claims.json            # 10 synthetic claims (the spread below)
  billing_history.json   # ~24 prior billed amounts per CPT (for the z-score)
  requirements.txt       # streamlit, groq, pandas, numpy, scipy
  .env.example           # GROQ_API_KEY=
report/                  # 2-page written report (Word) + bibliography
slides/                  # PowerPoint overview + screenshot assets
video/                   # recorded walkthrough (MP4)
```

### The 10 claims & expected behavior

| Claim | Case | Expected | Why |
|-------|------|----------|-----|
| CL-001…004 | clean | APPROVE → AUTO-CLEAR | all checks pass |
| CL-005 | dx/procedure mismatch | FLAG | ICD `J45.909` (asthma) not in colonoscopy's allowed-dx (**R-005**) |
| CL-006 | sex conflict | FLAG | male patient on female-only delivery code 59400 (**R-001**) |
| CL-007 | amount outlier | FLAG | passes every hard rule, but **amount_anomaly z≈5.8** |
| CL-008 | ineligible member | FLAG | coverage terminated |
| CL-009 | age edge | FLAG | screening colonoscopy at age 39 < guideline age 45 (**R-004**) |
| CL-010 | near-miss amount | APPROVE → **ROUTE TO HUMAN** | z≈2.0: anomaly doesn't fire, but the elevated amount caps confidence below the cutoff → routed on **confidence alone** (pillar ②) |

Running `python poc/app.py` prints each reasoning step and a
`10/10 claims matched expected` acceptance summary (reproducible in practice at
temperature 0; CL-010's sub-cutoff routing is enforced deterministically in
code, not left to the model's self-reported confidence).

---

## The three tools

| Tool | Signature | Notes |
|------|-----------|-------|
| `lookup_policy` | `(cpt) → {rules[]}` | Filters `rules.json` for the CPT. **Stand-in for production RAG** — in a real system this would semantically retrieve policy text from a vector store. |
| `check_eligibility` | `(member_id) → {active, covered}` | Synthetic member/coverage data. |
| `amount_anomaly` | `(cpt, billed_amount) → {is_anomaly, z_score, mean, std}` | Population z-score vs the CPT's billing history; flags `> 3σ`. **Independent of the LLM.** |

---

## Security (this repo is public)

- `.env` is git-ignored; only `.env.example` (no value) is committed.
- The key is read from `os.environ["GROQ_API_KEY"]` — never hardcoded.
- No PHI: 100% synthetic data, generated for this demo.

---

## Out of scope (kept simple on purpose)

RAG/vector DB, real claims data, auth, persistence, multi-page UI, and cloud
deploy are intentionally omitted. RAG, clustering, and broader anomaly methods
are *discussed in the report*, not built here — the prompt grades how well the
POC "hacks to prove a concept," not its complexity.

---

## Stack

Python · [Groq](https://groq.com) `llama-3.3-70b-versatile` (OpenAI-compatible
native tool-calling) · Streamlit · pandas · numpy · scipy · Altair.
