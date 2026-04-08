---
title: AP-Resolve
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Accounts Payable invoice exception-handling RL environment
---

# AP-Resolve

An OpenEnv-compatible Gymnasium-style environment that simulates an **Accounts Payable (AP) invoice exception-handling** workflow. An agent operates as a finance operations analyst inside a simulated ERP inbox, reviewing invoices, purchase orders, goods-receipt notes, vendor records, and policy documents before making a final disposition decision.

## Quick Start

```python
from envs.ap_resolve_env.client import APClient
from envs.ap_resolve_env.models import APAction

async with APClient(base_url="https://<your-space>.hf.space") as env:
    result = await env.reset(task_id="easy_straight_through")
    while not result.done:
        action = APAction(action_type="open_document", target="invoice")
        result = await env.step(action)
    print(f"Final score: {result.reward:.4f}")
```

## Web Interface

Visit `/web` for the interactive Gradio playground to manually test the environment.

## API Endpoints

- `GET /health` — health check
- `GET /schema` — action/observation JSON schemas
- `POST /reset` — reset episode (body: `{"task_id": "easy_straight_through"}`)
- `POST /step` — execute action (body: `{"action": {"action_type": "open_document", "target": "invoice"}}`)
- `GET /state` — full internal state (for debugging)
- `WS /ws` — WebSocket for persistent sessions

## Tasks

| Task | Difficulty | Correct Disposition | Budget |
|---|---|---|---|
| `easy_straight_through` | Easy | `approve_invoice` | 15 steps |
| `medium_mismatch` | Medium | `hold_invoice` | 18 steps |
| `hard_duplicate_partial` | Hard | `hold_invoice` | 20 steps |

## Baseline Scores

| Task | Score |
|---|---|
| easy_straight_through | ≥ 0.70 |
| medium_mismatch | ~0.40–0.55 |
| hard_duplicate_partial | ~0.35–0.50 |

See the full [README](envs/ap_resolve_env/README.md) for complete documentation.
