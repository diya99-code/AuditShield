"""
Baseline inference script for AP-Resolve.

Runs a deterministic rule-based agent against all three tasks and prints
a summary table with per-task scores and overall mean.

Usage:
    API_BASE_URL=http://localhost:8000 python -m envs.ap_resolve_env.inference

Environment variables:
    API_BASE_URL   - Base URL of the AP-Resolve server (required)
    MODEL_NAME     - Model name for logging (optional, default: "baseline-rule-agent")
    HF_TOKEN       - Hugging Face token (optional, for HF Space auth)
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Deterministic baseline agent
# ---------------------------------------------------------------------------

FIXED_SEED = 42
TASK_ORDER = ["easy_straight_through", "medium_mismatch", "hard_duplicate_partial"]

PROMPT_TEMPLATE = """
You are an AP analyst. Your job is to review invoice documents and make a disposition decision.

INSTRUCTIONS:
1. Open and read the invoice document
2. Open and read the purchase order
3. Open and read the goods receipt note
4. Extract key fields: invoice_amount, po_amount, vendor_id
5. Compare invoice_amount with po_amount
6. Check invoice history for duplicates
7. Make your final decision based on the evidence

Available actions: open_document, extract_field, compare_fields, search_history, add_note,
approve_invoice, hold_invoice, reject_invoice, escalate_case
"""


def baseline_agent_policy(
    obs: Dict[str, Any],
    step: int,
    task_id: str,
) -> Dict[str, Any]:
    """
    Deterministic rule-based baseline agent.

    Follows a fixed script: open docs → extract fields → compare → decide.
    """
    extracted = obs.get("extracted_facts", {})
    action_history = obs.get("action_history", [])
    visible_docs = obs.get("visible_documents", [])

    # Step-based deterministic script
    script = [
        {"action_type": "open_document", "target": "invoice"},
        {"action_type": "extract_field", "target": "invoice_amount"},
        {"action_type": "extract_field", "target": "vendor_id"},
        {"action_type": "open_document", "target": "purchase_order"},
        {"action_type": "extract_field", "target": "po_amount"},
        {"action_type": "open_document", "target": "goods_receipt"},
        {"action_type": "compare_fields", "target": "invoice_amount", "params": {"field_b": "po_amount"}},
        {"action_type": "search_history"},
        {"action_type": "add_note", "target": "Reviewed invoice, PO, and GRN. Checked history."},
    ]

    if step < len(script):
        return script[step]

    # Decision logic based on extracted facts
    inv_amount = extracted.get("invoice_amount")
    po_amount = extracted.get("po_amount")
    duplicate_flag = extracted.get("duplicate_flag", False)

    if duplicate_flag:
        return {"action_type": "hold_invoice"}

    if inv_amount is not None and po_amount is not None:
        try:
            diff_pct = abs(float(inv_amount) - float(po_amount)) / float(po_amount) * 100
            if diff_pct > 2.0:
                return {"action_type": "hold_invoice"}
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return {"action_type": "approve_invoice"}


def run_episode(
    base_url: str,
    task_id: str,
    model_name: str,
    hf_token: Optional[str] = None,
) -> float:
    """Run one episode against the server and return the final score."""
    import urllib.request
    import urllib.error

    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    def post(endpoint: str, data: dict) -> dict:
        url = f"{base_url.rstrip('/')}{endpoint}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    # Reset
    reset_resp = post("/reset", {"task_id": task_id})
    obs = reset_resp.get("observation", reset_resp)
    print(f"\n[{task_id}] Episode started. Case: {obs.get('case_id', '?')}")

    total_reward = 0.0
    step = 0

    while True:
        action = baseline_agent_policy(obs, step, task_id)
        print(f"  Step {step+1}: action={action.get('action_type')} target={action.get('target', '')}")

        step_resp = post("/step", {"action": action})
        obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", 0.0) or 0.0
        done = step_resp.get("done", obs.get("done", False))

        print(f"           reward={reward:.3f} done={done} msg={obs.get('message', '')[:80]}")

        if done:
            total_reward = reward
            break

        step += 1
        if step > 25:
            print("  [WARNING] Max steps exceeded, forcing hold")
            post("/step", {"action": {"action_type": "hold_invoice"}})
            break

    print(f"[{task_id}] Final score: {total_reward:.4f}")
    return total_reward


def main() -> None:
    random.seed(FIXED_SEED)

    base_url = os.environ.get("API_BASE_URL", "")
    model_name = os.environ.get("MODEL_NAME", "baseline-rule-agent")
    hf_token = os.environ.get("HF_TOKEN")

    if not base_url:
        print("ERROR: API_BASE_URL environment variable is required.", file=sys.stderr)
        print("Usage: API_BASE_URL=http://localhost:8000 python -m envs.ap_resolve_env.inference")
        sys.exit(1)

    print(f"AP-Resolve Baseline Inference")
    print(f"Model: {model_name}")
    print(f"Server: {base_url}")
    print(f"Tasks: {TASK_ORDER}")
    print("=" * 60)

    scores: Dict[str, float] = {}
    for task_id in TASK_ORDER:
        try:
            score = run_episode(base_url, task_id, model_name, hf_token)
            scores[task_id] = score
        except Exception as exc:
            print(f"[{task_id}] ERROR: {exc}")
            scores[task_id] = 0.0

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Task':<35} {'Score':>8}")
    print("-" * 60)
    for task_id, score in scores.items():
        print(f"{task_id:<35} {score:>8.4f}")
    print("-" * 60)
    mean_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"{'MEAN':<35} {mean_score:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
