import httpx
import json
import sys

# TEST LOCAL SERVER
BASE_URL = "http://localhost:8000"

def test_audit():
    print(f"Connecting to Local AuditShield Backend at {BASE_URL}...")
    
    try:
        # 1. Reset the environment (Start a new Case)
        response = httpx.post(f"{BASE_URL}/reset", json={"task_id": "easy_straight_through"}, timeout=10)
        response.raise_for_status()
        obs_resp = response.json()
        
        # OpenEnv responses sometimes nest the observation
        obs = obs_resp.get("observation", obs_resp)
            
        print(f"\n[START] New Case: {obs.get('case_id', 'unknown')}")
        print(f"Visible Docs: {obs.get('visible_documents', [])}")

        # 2. Open the Invoice (Our first action)
        action = {"action_type": "open_document", "target": "invoice"}
        response = httpx.post(f"{BASE_URL}/step", json={"action": action}, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # Handle observation nesting in StepResult
        obs_after = result.get("observation", result)
            
        print(f"\n[ACTION] Opening Invoice...")
        print(f"Message from World: {obs_after.get('message', 'No message')}")
        
    except httpx.HTTPStatusError as exc:
        print(f"\n[ERROR] HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
    except Exception as exc:
        print(f"\n[ERROR] An error occurred: {exc}")

if __name__ == "__main__":
    test_audit()
