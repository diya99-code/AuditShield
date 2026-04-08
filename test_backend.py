import httpx
import json
import sys

# Your live server URL
BASE_URL = "https://d0907-auditshield.hf.space"

def test_audit():
    print(f"Connecting to AuditShield Backend at {BASE_URL}...")
    
    try:
        # 1. Reset the environment (Start a new Case)
        response = httpx.post(f"{BASE_URL}/reset", json={"task_id": "easy_straight_through"}, timeout=10)
        response.raise_for_status()
        obs = response.json()
        
        # OpenEnv responses sometimes nest the observation
        if "observation" in obs:
            obs = obs["observation"]
            
        print(f"\n[START] New Case: {obs.get('case_id', 'unknown')}")
        print(f"Visible Docs: {obs.get('visible_documents', [])}")

        # 2. Open the Invoice (Our first action)
        action = {"action_type": "open_document", "target": "invoice"}
        response = httpx.post(f"{BASE_URL}/step", json={"action": action}, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # Handle observation nesting in StepResult
        if "observation" in result:
            obs_after = result["observation"]
        else:
            obs_after = result
            
        print(f"\n[ACTION] Opening Invoice...")
        print(f"Message from World: {obs_after.get('message', 'No message')}")
        
        # Peek at the Invoice Content!
        current_view = obs_after.get("current_view")
        if current_view:
            print("\n--- INVOICE CONTENT ---")
            print(current_view[:300] + "...")
            print("-----------------------")
        else:
            print("\n[WARNING] No document content returned. Check if the document name 'invoice' exists.")

        # 3. Ask for the State (The 'Internal' Truth)
        state_resp = httpx.get(f"{BASE_URL}/state", timeout=10)
        state_resp.raise_for_status()
        state = state_resp.json()
        print(f"\n[STATE] Hidden Ground Truth Disposition: {state.get('hidden_ground_truth', {}).get('disposition', 'N/A')}")
        
    except httpx.HTTPStatusError as exc:
        print(f"\n[ERROR] HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
    except Exception as exc:
        print(f"\n[ERROR] An error occurred: {exc}")

if __name__ == "__main__":
    test_audit()
