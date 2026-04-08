import os
import sys

# Add paths to sys.path
sys.path.append(os.path.join(os.getcwd(), "openenv_src", "src"))
sys.path.append(os.getcwd())

from envs.ap_resolve_env.env import APEnvironment
from envs.ap_resolve_env.models import APAction

def run_hard_task():
    env = APEnvironment()
    task_id = "hard_duplicate_partial"
    obs = env.reset(task_id=task_id)
    print(f"Started task: {task_id}")
    
    # script from inference.py
    actions = [
        {"action_type": "open_document", "target": "invoice"},
        {"action_type": "extract_field", "target": "invoice_amount"},
        {"action_type": "extract_field", "target": "vendor_id"},
        {"action_type": "open_document", "target": "purchase_order"},
        {"action_type": "extract_field", "target": "po_amount"},
        {"action_type": "open_document", "target": "goods_receipt"},
        {"action_type": "extract_field", "target": "received_qty"},
        {"action_type": "compare_fields", "target": "invoice_amount", "params": {"field_b": "po_amount"}},
        {"action_type": "open_document", "target": "invoice_history"},
        {"action_type": "search_history"},
        {"action_type": "open_document", "target": "ap_policy"},
        {"action_type": "check_policy"},
        {"action_type": "add_note", "target": "Reviewed all docs. Found potential duplicate and partial delivery."},
        {"action_type": "hold_invoice"}
    ]
    
    total_reward = 0.0
    for i, act_data in enumerate(actions):
        action = APAction(**act_data)
        obs = env.step(action)
        print(f"Step {i+1}: {act_data['action_type']} -> reward delta: {obs.reward}")
        if obs.done:
            total_reward = obs.reward
            break
            
    print(f"\nFinal Score: {total_reward}")
    if 0.0 <= total_reward <= 1.0:
        print("SUCCESS: Score is between 0 and 1.")
    else:
        print("FAILURE: Score is NOT between 0 and 1.")

if __name__ == "__main__":
    run_hard_task()
