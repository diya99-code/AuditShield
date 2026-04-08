---
title: AuditShield
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AuditShield (AP-Resolve) - AI Auditor Reinforcement Learning Environment

AuditShield is a high-fidelity reinforcement learning environment designed to train and benchmark AI agents in accounts payable auditing. Agents must process invoices, match them against purchase orders and receipts, and identify complex exceptions like duplicate risks or policy violations.

## 🎯 Task Tiers

| Difficulty | Task Name | Scenario | Target Action |
| :--- | :--- | :--- | :--- |
| **Easy** | `easy_straight_through` | 3-way match, clean invoice | `approve_invoice` |
| **Medium** | `medium_mismatch` | Quantity/Price discrepancy | `hold_invoice` |
| **Hard** | `hard_duplicate_partial` | Duplicate risk + partial receipt + policy | `hold_invoice` / `escalate` |

## ⚖️ Grading System (Unified Grader)

Each episode is scored from **0.0 to 1.0** based on four deterministic buckets:

1. **Decision Correctness (40%)**: Does the final action match the ground truth? (Auto-zero if unsafe approval on high-risk case).
2. **Required Checks (25%)**: Proportion of mandatory documents opened and compared.
3. **Issue Identification (20%)**: Did the agent correctly tag the specific core issues (e.g., `quantity_mismatch`)?
4. **Efficiency & Safety (15%)**: Penalties for unnecessary loops, budget overruns, or redundant actions.

## 💰 Reward Table

AuditShield uses a **dense reward function** to encourage meaningful investigation:

| Event | Reward |
| :--- | :--- |
| Open relevant unseen document | +0.05 |
| Extract critical field correctly | +0.05 |
| Complete relevant comparison | +0.10 |
| Detect core issue | +0.10 |
| Correct final decision | +0.40 |
| Finish within step budget | +0.05 |
| **Penalty** | **Value** |
| Wrong final decision | -0.30 |
| Unsafe approval on bad invoice | -0.40 |
| Redundant document re-opening | -0.03 |
| Repeated looping behavior | -0.10 |
| Skipping required checks | -0.20 |

## 🚀 Quick Start (Local Development)

### 1. Install Dependencies
```bash
pip install -e ./openenv_src
pip install fastapi uvicorn gradio
```

### 2. Launch the Environment (Playground)
```bash
$env:HF_TOKEN="your_token"; python -m uvicorn envs.ap_resolve_env.server.app:app --host 0.0.0.0 --port 7860
```
Open [http://localhost:7860/web](http://localhost:7860/web) to access the Gradio Playground with the **Task Selector**.

### 3. Run Benchmark
```bash
$env:HF_TOKEN="your_token"; python inference_benchmark.py
```

## 📂 Project Structure
- `envs/ap_resolve_env/fixtures/`: Task definitions and ground truths.
- `envs/ap_resolve_env/server/ap_environment.py`: Primary RL environment loop.
- `envs/ap_resolve_env/graders.py`: Deterministic scoring logic.
- `envs/ap_resolve_env/rewards.py`: Dense reward calculation.
- `inference_benchmark.py`: Standardized logging script for hackathon evaluation.

---
*Built for the Agentic AI Hackathon 2024.*
