# AP-Resolve

An OpenEnv-compatible Gymnasium-style environment that simulates an **Accounts Payable (AP) invoice exception-handling** workflow. An agent operates as a finance operations analyst inside a simulated ERP inbox, reviewing invoices, purchase orders, goods-receipt notes, vendor records, and policy documents before making a final disposition decision.

---

## Overview

AP-Resolve is designed for RL research on multi-step document reasoning with real-world financial consequences. Key features:

- **Dense reward shaping** — rewards correct procedural steps, not just the final answer
- **Deterministic, LLM-free grading** — reproducible hackathon evaluation
- **Three progressively harder tasks** — straight-through approval, mismatch detection, complex multi-signal reasoning
- **Audit-risk mechanic** — penalizes reckless approvals, mirroring real-world consequences

---

## Observation Space

Each step returns an `APObservation` with:

| Field | Type | Description |
|---|---|---|
| `case_id` | str | Unique case identifier |
| `task_id` | str | Task name (easy/medium/hard) |
| `visible_documents` | list[str] | Available document names |
| `current_view` | str \| None | Content of the last opened document |
| `extracted_facts` | dict | Fields extracted so far |
| `pending_issues` | list[str] | Notes added by the agent |
| `action_history` | list[str] | History of actions taken |
| `steps_remaining` | int | Steps left before budget exhaustion |
| `message` | str | Human-readable result of the last action |
| `done` | bool | Whether the episode has ended |
| `reward` | float | Reward for this step |

---

## Action Space

`APAction` has three fields:

| Field | Type | Description |
|---|---|---|
| `action_type` | ActionType | One of 12 valid action types (see below) |
| `target` | str \| None | Primary argument (document name, field name, note text) |
| `params` | dict | Additional parameters (e.g. `field_b` for compare_fields) |

### Action Types

| Action | Description |
|---|---|
| `open_document` | Open a document by name |
| `extract_field` | Extract a named field from the current document |
| `compare_fields` | Compare two previously extracted fields |
| `calculate_total` | Compute total from extracted line items |
| `check_policy` | Look up a policy rule |
| `search_history` | Search invoice history for duplicates |
| `request_vendor_info` | Contact vendor for clarification |
| `add_note` | Add a note to the case |
| `approve_invoice` | **Terminal** — approve the invoice |
| `hold_invoice` | **Terminal** — place invoice on hold |
| `reject_invoice` | **Terminal** — reject the invoice |
| `escalate_case` | **Terminal** — escalate to senior reviewer |

---

## State Model

`APState` (full internal state, available via `/state`):

| Field | Description |
|---|---|
| `case_id` / `task_id` | Episode identifiers |
| `hidden_ground_truth` | Correct disposition and critical issue flag |
| `extracted_facts` | All fields extracted during the episode |
| `checks_completed` | Which documents have been opened |
| `decision` | Final disposition (set on terminal action) |
| `notes` | Agent notes |
| `vendor_contacted` | Whether vendor was contacted |
| `reward_so_far` | Cumulative reward (clipped to [0,1]) |
| `step_count` | Steps taken |

---

## Task Descriptions

### easy_straight_through
Invoice, PO, and GRN all match. Correct disposition: `approve_invoice`. Step budget: 15.

### medium_mismatch
Invoice quantity exceeds PO authorization (SSD quantity discrepancy). Correct disposition: `hold_invoice`. Step budget: 18.

### hard_duplicate_partial
Combines a potential duplicate invoice, partial goods receipt (60% complete), and an early-payment discount rule. Correct disposition: `hold_invoice`. Step budget: 20.

---

## Reward Design

| Event | Delta |
|---|---|
| Open relevant document (first time) | +0.05 |
| Extract critical field | +0.10 |
| Complete required field comparison | +0.10 |
| Identify core issue | +0.15 |
| Re-open already-opened document | -0.03 |
| Irrelevant action | -0.05 |
| Each step over soft threshold | -0.02 |
| Correct final disposition | +0.30 to +0.40 |
| Evidence-backed note on terminal | +0.10 bonus |
| Audit-risk: approve with critical issues | -0.40 |

All cumulative rewards are clipped to **[0.0, 1.0]**.

---

## Grader Logic

The `Grader` computes a composite score from four weighted sub-scores:

```
composite = 0.40 * decision_score
          + 0.25 * evidence_score
          + 0.20 * workflow_score
          + 0.15 * efficiency_score
```

- **decision_score** (0 or 1): Correct disposition match
- **evidence_score** ([0,1]): Fraction of required evidence fields present
- **workflow_score** ([0,1]): Fraction of required document checks completed
- **efficiency_score** ([0,1]): Steps remaining / step budget

An `audit_failure` flag is set when the agent approves an invoice with critical unresolved issues.

---

## Setup

```bash
# Install dependencies
pip install -e OpenEnv-main/

# Run tests
cd OpenEnv-main
python -m pytest envs/ap_resolve_env/tests/ -v
```

---

## Docker Usage

```bash
# Build
docker build -f envs/ap_resolve_env/server/Dockerfile -t ap-resolve-env .

# Run
docker run -p 7860:7860 ap-resolve-env

# Health check
curl http://localhost:7860/health
```

---

## HF Space Deployment

Push the `ap_resolve_env` directory to a Hugging Face Space with `runtime: fastapi` and `app: server.app:app`. The `openenv.yaml` manifest is already configured.

---

## Baseline Script Usage

```bash
# Start the server first
uvicorn envs.ap_resolve_env.server.app:app --host 0.0.0.0 --port 7860

# Run baseline
API_BASE_URL=http://localhost:7860 python -m envs.ap_resolve_env.inference
```

Optional env vars:
- `MODEL_NAME` — label for logging (default: `baseline-rule-agent`)
- `HF_TOKEN` — Hugging Face token for authenticated Spaces
- `TASK_ID` — pin all resets to a specific task
- `ENABLE_WEB_INTERFACE=true` — enable web debug UI at `/web`

---

## Baseline Scores

| Task | Score |
|---|---|
| easy_straight_through | ≥ 0.70 |
| medium_mismatch | ~0.40–0.55 |
| hard_duplicate_partial | ~0.35–0.50 |

---

## Known Limitations

- Field extraction uses regex heuristics; unusual document formats may fail to extract some fields
- The baseline agent is rule-based and does not use an LLM
- Vendor responses are simulated (pre-authored in fixture files)
- The web debug interface requires `ENABLE_WEB_INTERFACE=true` and `openenv-core` web extras
