# Design Document: AP-Resolve

## Overview

AP-Resolve is an OpenEnv-compatible Gymnasium-style environment that simulates an Accounts Payable (AP) invoice exception-handling workflow. An agent operates as a finance operations analyst inside a simulated company ERP inbox, receiving invoices, purchase orders, goods-receipt notes, vendor messages, and policy rules, then deciding whether to approve, hold, reject, request clarification, or escalate the invoice.

The environment is implemented as a FastAPI server packaged in a Docker container and deployable as a Hugging Face Space. It exposes the standard OpenEnv `reset()` / `step()` / `state()` API and uses fully typed Pydantic models for all data exchange.

Key design goals:
- Dense reward shaping that rewards correct procedural behavior, not just the final answer
- Deterministic, LLM-free grading for reproducible hackathon evaluation
- Three progressively harder tasks covering straight-through approval, mismatch detection, and complex multi-signal reasoning
- Audit-risk mechanic that penalizes reckless approvals, mirroring real-world consequences

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              RL Training Loop / Inference Script        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  APClient (EnvClient[APAction, APObservation])   │   │
│  └──────────────────────┬───────────────────────────┘   │
└─────────────────────────┼───────────────────────────────┘
                          │ HTTP (reset, step, state)
┌─────────────────────────▼───────────────────────────────┐
│              Docker Container / HF Space                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  FastAPI App (create_fastapi_app)                │   │
│  │  ┌────────────────────────────────────────────┐  │   │
│  │  │  APEnvironment (Environment)               │  │   │
│  │  │  ┌──────────────┐  ┌────────────────────┐  │  │   │
│  │  │  │ TaskRegistry │  │ DocumentWorkspace  │  │  │   │
│  │  │  └──────────────┘  └────────────────────┘  │  │   │
│  │  │  ┌──────────────┐  ┌────────────────────┐  │  │   │
│  │  │  │ ActionHandler│  │ RewardCalculator   │  │  │   │
│  │  │  └──────────────┘  └────────────────────┘  │  │   │
│  │  │  ┌──────────────┐                           │  │   │
│  │  │  │    Grader    │                           │  │   │
│  │  │  └──────────────┘                           │  │   │
│  │  └────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Request Flow

1. Client calls `reset(task_id?)` → APEnvironment loads fixture, initializes APState, returns APObservation
2. Client calls `step(APAction)` → ActionHandler executes action, RewardCalculator computes delta reward, APEnvironment returns updated APObservation
3. On terminal action → Grader computes final composite score, episode ends with `done=True`
4. Client calls `state()` → APEnvironment returns full APState for debugging

---

## Components and Interfaces

### APEnvironment

The central server-side class extending `openenv_core.Environment`.

```python
class APEnvironment(Environment):
    def __init__(self):
        self._task_registry = TaskRegistry()
        self._state: APState = APState()
        self._workspace: DocumentWorkspace = None
        self._action_handler = ActionHandler()
        self._reward_calc = RewardCalculator()
        self._grader = Grader()

    def reset(self, task_id: str | None = None) -> APObservation: ...
    def step(self, action: APAction) -> APObservation: ...

    @property
    def state(self) -> APState: ...
```

### APClient

Client-side class extending `EnvClient[APAction, APObservation, APState]`.

```python
class APClient(EnvClient[APAction, APObservation, APState]):
    def _step_payload(self, action: APAction) -> dict: ...
    def _parse_result(self, payload: dict) -> StepResult[APObservation]: ...
    def _parse_state(self, payload: dict) -> APState: ...
```

### TaskRegistry

Loads and indexes fixture files from `fixtures/`.

```python
class TaskRegistry:
    TASKS = ["easy_straight_through", "medium_mismatch", "hard_duplicate_partial"]

    def get(self, task_id: str) -> CaseFixture: ...
    def random(self) -> CaseFixture: ...
    def list_ids(self) -> list[str]: ...
```

### DocumentWorkspace

Manages the set of documents available in the current episode and tracks which have been opened.

```python
class DocumentWorkspace:
    def open(self, doc_name: str) -> str | None: ...          # returns content or None
    def extract_field(self, field: str) -> Any | None: ...    # returns value or None
    def list_available(self) -> list[str]: ...
    def already_opened(self, doc_name: str) -> bool: ...
```

### ActionHandler

Dispatches each `APAction` to the appropriate workspace or state mutation method and returns a raw result dict.

```python
class ActionHandler:
    def handle(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
    ) -> ActionResult: ...
```

`ActionResult` carries: `success: bool`, `message: str`, `new_facts: dict`, `issues_resolved: list[str]`, `is_terminal: bool`.

### RewardCalculator

Computes the reward delta for a single step given the action, result, and current state.

```python
class RewardCalculator:
    def compute(
        self,
        action: APAction,
        result: ActionResult,
        state: APState,
        fixture: CaseFixture,
    ) -> float: ...
```

Reward rules (all clipped to [0.0, 1.0] cumulatively):

| Event | Delta |
|---|---|
| Open relevant document (first time) | +0.05 |
| Extract correct critical field | +0.10 |
| Complete required field comparison | +0.10 |
| Identify core issue correctly | +0.15 |
| Re-open already-opened document | -0.03 |
| Irrelevant action | -0.05 |
| Invalid action type | -0.05 |
| Each step over soft threshold | -0.02 |
| Correct final disposition | +0.30 to +0.40 |
| Evidence-backed note on terminal | +0.10 bonus |
| Audit-risk: approve with critical issues | -0.40 |

### Grader

Deterministic, LLM-free scoring component invoked at episode end.

```python
@dataclass
class GradeResult:
    composite_score: float          # weighted sum, clipped [0,1]
    decision_score: float           # 0.0 or 1.0
    evidence_score: float           # [0,1] fraction of required evidence present
    workflow_score: float           # [0,1] fraction of required checks completed
    efficiency_score: float         # [0,1] steps remaining / budget
    audit_failure: bool             # True if wrong approval on critical case
    details: dict                   # per-field breakdown for transparency

class Grader:
    WEIGHTS = {
        "decision": 0.40,
        "evidence": 0.25,
        "workflow": 0.20,
        "efficiency": 0.15,
    }

    def grade(self, state: APState, fixture: CaseFixture) -> GradeResult: ...
```

Composite score formula:
```
composite = (
    0.40 * decision_score +
    0.25 * evidence_score +
    0.20 * workflow_score +
    0.15 * efficiency_score
)
composite = min(1.0, max(0.0, composite))
```

---

## Data Models

### APAction

```python
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel

ActionType = Literal[
    "open_document", "extract_field", "compare_fields", "calculate_total",
    "check_policy", "search_history", "request_vendor_info", "add_note",
    "approve_invoice", "hold_invoice", "reject_invoice", "escalate_case",
]

class APAction(BaseModel):
    action_type: ActionType
    target: Optional[str] = None
    params: Dict[str, Any] = {}
```

### APObservation

```python
from typing import Optional, List, Dict, Any
from openenv_core import Observation

class APObservation(Observation):
    case_id: str
    task_id: str
    visible_documents: List[str]
    current_view: Optional[str] = None
    extracted_facts: Dict[str, Any] = {}
    pending_issues: List[str] = []
    action_history: List[str] = []
    steps_remaining: int
    message: str
```

### APState

```python
from typing import Optional, List, Dict, Any
from openenv_core import State

class APState(State):
    case_id: str = ""
    task_id: str = ""
    hidden_ground_truth: Dict[str, Any] = {}
    extracted_facts: Dict[str, Any] = {}
    checks_completed: Dict[str, bool] = {}
    decision: Optional[str] = None
    notes: List[str] = []
    vendor_contacted: bool = False
    reward_so_far: float = 0.0
    step_count: int = 0
```

### CaseFixture

```python
@dataclass
class CaseFixture:
    task_id: str
    case_id: str
    documents: Dict[str, str]           # doc_name -> content string
    critical_fields: List[str]          # fields that must be extracted
    required_checks: List[str]          # document names that must be opened
    ground_truth_disposition: str       # correct final action_type
    required_evidence_fields: List[str] # fields that must appear in notes/facts
    has_critical_issues: bool           # True for medium/hard cases
    soft_step_threshold: int
    max_steps: int
```

### Task Fixture Schemas

Each fixture JSON file under `fixtures/` follows this structure:

```json
{
  "task_id": "easy_straight_through",
  "case_id": "CASE-001",
  "documents": {
    "invoice": "...",
    "purchase_order": "...",
    "goods_receipt": "...",
    "vendor_master": "...",
    "ap_policy": "...",
    "invoice_history": "..."
  },
  "critical_fields": ["invoice_amount", "po_amount", "received_qty", "billed_qty", "vendor_id"],
  "required_checks": ["invoice", "purchase_order", "goods_receipt"],
  "ground_truth_disposition": "approve_invoice",
  "required_evidence_fields": ["invoice_amount", "po_amount", "vendor_id"],
  "has_critical_issues": false,
  "soft_step_threshold": 8,
  "max_steps": 15
}
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Reset produces clean state

*For any* sequence of actions taken in an episode, calling `reset()` afterwards should return an observation with an empty `action_history`, `extracted_facts = {}`, `vendor_contacted = False`, and `decision = None` in the resulting state.

**Validates: Requirements 1.4**

---

### Property 2: APObservation schema invariant

*For any* action submitted to a running episode, the returned `APObservation` should contain all required fields: `case_id`, `task_id`, `visible_documents`, `extracted_facts`, `pending_issues`, `action_history`, `steps_remaining`, and `message`.

**Validates: Requirements 2.3**

---

### Property 3: APState schema invariant

*For any* call to `state()` during or after an episode, the returned `APState` should contain all required fields: `case_id`, `task_id`, `hidden_ground_truth`, `extracted_facts`, `checks_completed`, `notes`, `vendor_contacted`, `reward_so_far`, and `step_count`.

**Validates: Requirements 2.4**

---

### Property 4: Invalid action type is rejected without termination

*For any* string not in the allowed `ActionType` literal union submitted as `action_type`, the environment should return an observation with `done=False` and a negative reward delta, keeping the episode alive.

**Validates: Requirements 2.5**

---

### Property 5: Task ID round-trip

*For any* valid `task_id` in the task registry, calling `reset(task_id=task_id)` should return an observation whose `task_id` field equals the requested `task_id`.

**Validates: Requirements 3.2**

---

### Property 6: All fixtures contain required document keys

*For any* fixture loaded from the task registry, the `documents` dict should contain all of: `invoice`, `purchase_order`, `goods_receipt`, `vendor_master`, `ap_policy`, `invoice_history`.

**Validates: Requirements 3.7**

---

### Property 7: open_document populates current_view

*For any* case and any document name in `visible_documents`, submitting `open_document` with that name should return an observation where `current_view` is a non-empty string.

**Validates: Requirements 4.1**

---

### Property 8: extract_field round-trip

*For any* case where a document has been opened and a valid field name is extracted, the field should appear in `extracted_facts` in the subsequent observation, and calling `state()` should reflect the same value.

**Validates: Requirements 4.3**

---

### Property 9: add_note appends to notes list

*For any* note text submitted via `add_note`, the `notes` list in `APState` should grow by exactly one entry containing that text.

**Validates: Requirements 4.9**

---

### Property 10: Terminal actions set done=True and record decision

*For any* terminal action type (`approve_invoice`, `hold_invoice`, `reject_invoice`, `escalate_case`), the returned observation should have `done=True`, and `state().decision` should equal the submitted `action_type`.

**Validates: Requirements 5.1, 5.4**

---

### Property 11: Step budget exhaustion terminates episode

*For any* episode where the agent takes exactly `max_steps` actions without a terminal action, the environment should return `done=True` and `reward=0.0` on the final step.

**Validates: Requirements 5.5**

---

### Property 12: Reward is always in [0.0, 1.0]

*For any* sequence of actions in any episode, the cumulative `reward_so_far` in `APState` should always remain in the range [0.0, 1.0].

**Validates: Requirements 6.9**

---

### Property 13: Grader composite score equals weighted sub-scores

*For any* `GradeResult`, the `composite_score` should equal `0.40 * decision_score + 0.25 * evidence_score + 0.20 * workflow_score + 0.15 * efficiency_score`, clipped to [0.0, 1.0].

**Validates: Requirements 7.1**

---

### Property 14: Grader decision score is binary

*For any* grading call, `decision_score` should be exactly 1.0 when the agent's disposition matches the ground-truth disposition, and exactly 0.0 otherwise.

**Validates: Requirements 7.2**

---

### Property 15: Grader evidence score reflects required field coverage

*For any* set of extracted facts and notes, `evidence_score` should equal the fraction of `required_evidence_fields` that are present, in the range [0.0, 1.0].

**Validates: Requirements 7.3**

---

### Property 16: Audit failure on wrong approval of critical case

*For any* case where `has_critical_issues=True` and the agent submits `approve_invoice`, the `GradeResult` should have `audit_failure=True` and `decision_score=0.0`.

**Validates: Requirements 5.3, 7.7**

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Invalid `action_type` | Return error observation, apply -0.05 penalty, episode continues |
| `open_document` with unknown name | Return error message, apply -0.05 penalty, episode continues |
| `extract_field` before opening any document | Return error message, no reward delta |
| `compare_fields` with un-extracted fields | Return error message listing missing fields |
| `reset()` called mid-episode | Allowed; clears all state and starts fresh |
| `step()` after `done=True` | Return observation with `done=True`, `reward=0.0`, no state change |
| Fixture file missing or malformed | Raise `ValueError` at startup during registry initialization |
| Unknown `task_id` in `reset()` | Raise `ValueError` with list of valid task IDs |

---

## Testing Strategy

### Dual Testing Approach

Both unit tests and property-based tests are required. They are complementary:
- Unit tests verify specific examples, edge cases, and integration points
- Property-based tests verify universal properties hold across all inputs

### Property-Based Testing Library

Use **Hypothesis** (Python) for all property-based tests.

```bash
pip install hypothesis
```

Each property test runs a minimum of 100 iterations (Hypothesis default `max_examples=100`).

### Property Test Annotation Format

Each property test must be annotated with a comment referencing the design property:

```python
# Feature: ap-resolve, Property N: <property_text>
@given(...)
@settings(max_examples=100)
def test_property_N_name(...):
    ...
```

### Unit Test Coverage

Unit tests should cover:
- `TaskRegistry.get()` with valid and invalid task IDs
- `DocumentWorkspace.open()` with valid and invalid document names
- `Grader.grade()` with known fixture inputs and expected outputs
- `RewardCalculator.compute()` for each reward rule in isolation
- `APEnvironment.reset()` and `APEnvironment.step()` for each action type
- Fixture loading and schema validation for all three task fixtures
- Health check endpoint returns 200

### Integration Tests

- Full episode run for each of the three tasks using the baseline agent policy
- Verify baseline easy-task score >= 0.70
- Verify Docker container starts and `/health` returns 200 within 30 seconds

### Test File Structure

```
tests/
├── test_models.py          # APAction, APObservation, APState schema tests
├── test_task_registry.py   # TaskRegistry unit tests
├── test_document_workspace.py
├── test_action_handler.py
├── test_reward_calculator.py
├── test_grader.py
├── test_environment.py     # APEnvironment reset/step/state
├── test_properties.py      # All Hypothesis property-based tests
└── test_integration.py     # Full episode runs
```
