# Requirements Document

## Introduction

AP-Resolve is an OpenEnv-compatible Gymnasium-style environment that simulates an Accounts Payable (AP) invoice exception-handling workflow. An agent operates as a finance operations analyst inside a simulated company ERP inbox. It receives invoices, purchase orders, goods-receipt notes, vendor messages, and policy rules, then decides whether to approve, hold, reject, request clarification, or escalate the invoice. The environment exposes `reset()`, `step()`, and `state()` APIs, runs as a containerized FastAPI server, and is deployable as a Hugging Face Space.

## Glossary

- **AP_Environment**: The server-side OpenEnv environment implementing the AP-Resolve simulation.
- **AP_Client**: The client-side `EnvClient` subclass used by RL training loops to interact with AP_Environment.
- **APAction**: Typed Pydantic/dataclass model representing a single agent action.
- **APObservation**: Typed Pydantic/dataclass model representing what the agent perceives after each step.
- **APState**: Typed model representing the full internal simulator state (used for debugging and grading).
- **Case**: A single AP episode containing an invoice, PO, GRN, vendor record, policy, and optional history.
- **Task**: A named scenario (easy / medium / hard) with a hidden ground-truth disposition and scoring rubric.
- **Disposition**: The final agent decision: one of `approve_invoice`, `hold_invoice`, `reject_invoice`, `escalate_case`.
- **Grader**: A deterministic scoring component that evaluates agent trajectories against ground-truth.
- **Reward**: A scalar in [0.0, 1.0] returned per step, shaped to encourage correct procedural behavior.
- **Episode**: One full interaction sequence from `reset()` to a terminal action or step-budget exhaustion.
- **ERP_Inbox**: The simulated document workspace the agent operates within.
- **Fixture**: A hand-authored JSON case file stored under `fixtures/` with hidden ground truth.

---

## Requirements

### Requirement 1: Core OpenEnv API Compliance

**User Story:** As an RL framework developer, I want AP-Resolve to expose standard `reset()`, `step()`, and `state()` APIs, so that I can integrate it with any OpenEnv-compatible training loop without custom adapters.

#### Acceptance Criteria

1. THE AP_Environment SHALL implement `reset() -> APObservation` that initializes a new episode and returns the initial observation.
2. THE AP_Environment SHALL implement `step(action: APAction) -> APObservation` that executes one agent action and returns the resulting observation.
3. THE AP_Environment SHALL implement a `state` property returning an `APState` object reflecting the full current simulator state.
4. WHEN `reset()` is called, THE AP_Environment SHALL clear all episode state and load a fresh Case from the task registry.
5. WHEN `step()` is called after the episode is terminal, THE AP_Environment SHALL return an observation with `done=True` and `reward=0.0`.
6. THE AP_Environment SHALL be served as a FastAPI application compatible with the `openenv-core` `create_fastapi_app` helper.

---

### Requirement 2: Typed Action, Observation, and State Models

**User Story:** As an environment developer, I want all data exchanged between agent and environment to use typed Pydantic models, so that the environment passes OpenEnv validator checks and is easy to inspect and debug.

#### Acceptance Criteria

1. THE APAction model SHALL include a `action_type` field constrained to the literal union: `open_document`, `extract_field`, `compare_fields`, `calculate_total`, `check_policy`, `search_history`, `request_vendor_info`, `add_note`, `approve_invoice`, `hold_invoice`, `reject_invoice`, `escalate_case`.
2. THE APAction model SHALL include optional `target` (str) and `params` (Dict[str, Any]) fields.
3. THE APObservation model SHALL include: `case_id`, `task_id`, `visible_documents` (list of document names), `current_view` (optional document content string), `extracted_facts` (dict), `pending_issues` (list of strings), `action_history` (list of strings), `steps_remaining` (int), and `message` (str).
4. THE APState model SHALL include: `case_id`, `task_id`, `hidden_ground_truth` (dict), `extracted_facts` (dict), `checks_completed` (dict of bool flags), `decision` (optional str), `notes` (list of str), `vendor_contacted` (bool), `reward_so_far` (float), and `step_count` (int).
5. WHEN an action with an invalid `action_type` is submitted, THE AP_Environment SHALL return an observation with an error message and a penalty reward without terminating the episode.

---

### Requirement 3: Task Registry and Case Fixtures

**User Story:** As a researcher, I want the environment to support at least three progressively harder tasks loaded from fixture files, so that I can benchmark agents across a difficulty spectrum.

#### Acceptance Criteria

1. THE Task_Registry SHALL contain at least three tasks: `easy_straight_through`, `medium_mismatch`, and `hard_duplicate_partial`.
2. WHEN `reset(task_id=...)` is called with a valid task ID, THE AP_Environment SHALL load the corresponding fixture and initialize the episode with that case.
3. WHEN `reset()` is called without a `task_id`, THE AP_Environment SHALL select a task uniformly at random from the registry.
4. THE easy_straight_through task SHALL present a case where invoice, PO, and GRN all match, and the correct disposition is `approve_invoice`.
5. THE medium_mismatch task SHALL present a case with a quantity or tax discrepancy, where the correct disposition is `hold_invoice` or `reject_invoice` per policy.
6. THE hard_duplicate_partial task SHALL present a case combining a potential duplicate invoice, partial goods receipt, and an early-payment discount rule, requiring multi-step reasoning to resolve.
7. EACH fixture file SHALL include: invoice document, PO document, GRN document, vendor master record, AP policy document, invoice history, and hidden ground-truth disposition with required evidence fields.

---

### Requirement 4: Action Execution and Document Workspace

**User Story:** As an agent, I want to inspect documents and extract information step by step, so that I can gather evidence before making a final disposition decision.

#### Acceptance Criteria

1. WHEN the agent submits `open_document` with a valid document name, THE AP_Environment SHALL return the document content in `current_view` of the observation.
2. WHEN the agent submits `open_document` with an invalid document name, THE AP_Environment SHALL return an error message and apply a penalty reward.
3. WHEN the agent submits `extract_field` with a valid field name and a document has been opened, THE AP_Environment SHALL populate `extracted_facts` with the field value and return it in the observation.
4. WHEN the agent submits `compare_fields` naming two previously extracted fields, THE AP_Environment SHALL return a comparison result (match / mismatch / delta) in the observation message.
5. WHEN the agent submits `calculate_total`, THE AP_Environment SHALL compute the invoice total from extracted line items and return the result.
6. WHEN the agent submits `check_policy` with a policy rule name, THE AP_Environment SHALL return the relevant policy text from the AP policy document.
7. WHEN the agent submits `search_history`, THE AP_Environment SHALL return a summary of past invoices from the same vendor, including any potential duplicate matches.
8. WHEN the agent submits `request_vendor_info` with a clarification message, THE AP_Environment SHALL record the request, set `vendor_contacted=True` in state, and return a simulated vendor response.
9. WHEN the agent submits `add_note` with note text, THE AP_Environment SHALL append the note to the `notes` list in state and confirm in the observation.

---

### Requirement 5: Terminal Disposition Actions

**User Story:** As an agent, I want to submit a final disposition decision with supporting evidence, so that the episode can be graded and a final reward computed.

#### Acceptance Criteria

1. WHEN the agent submits `approve_invoice`, `hold_invoice`, `reject_invoice`, or `escalate_case`, THE AP_Environment SHALL mark the episode as terminal (`done=True`).
2. WHEN a terminal disposition is submitted, THE AP_Environment SHALL invoke the Grader and return the final composite score as the reward.
3. WHEN `approve_invoice` is submitted on a case with unresolved critical issues, THE AP_Environment SHALL apply an audit-risk penalty of -0.40 and terminate the episode immediately.
4. WHEN a terminal disposition is submitted, THE APState SHALL record the `decision` field with the chosen disposition string.
5. WHEN the step budget is exhausted without a terminal action, THE AP_Environment SHALL terminate the episode with `done=True` and `reward=0.0`.

---

### Requirement 6: Dense Reward Shaping

**User Story:** As an RL researcher, I want the environment to emit dense intermediate rewards throughout the episode, so that agents receive learning signal for correct procedural steps, not just the final decision.

#### Acceptance Criteria

1. WHEN the agent opens a relevant document for the first time, THE Reward_Calculator SHALL add +0.05 to the step reward.
2. WHEN the agent correctly extracts a critical field, THE Reward_Calculator SHALL add +0.10 to the step reward.
3. WHEN the agent completes a required field comparison, THE Reward_Calculator SHALL add +0.10 to the step reward.
4. WHEN the agent correctly identifies the core issue (mismatch, duplicate, or policy violation), THE Reward_Calculator SHALL add +0.15 to the step reward.
5. WHEN the agent re-opens a document it has already opened in the same episode, THE Reward_Calculator SHALL subtract -0.03 from the step reward.
6. WHEN the agent submits an irrelevant action (action type valid but target not applicable to the case), THE Reward_Calculator SHALL subtract -0.05 from the step reward.
7. WHEN the agent exceeds the soft step threshold (defined per task), THE Reward_Calculator SHALL subtract -0.02 per additional step.
8. WHEN the agent submits a terminal disposition action, THE Reward_Calculator SHALL add the final decision score (up to +0.40 for correct disposition).
9. THE Reward_Calculator SHALL clip all cumulative rewards to the range [0.0, 1.0].

---

### Requirement 7: Deterministic Grader

**User Story:** As a hackathon judge, I want the grader to produce reproducible scores using only structured checks, so that evaluation results are fair and auditable without relying on LLM judges.

#### Acceptance Criteria

1. THE Grader SHALL compute a composite score from four weighted sub-scores: `decision_score` (weight 0.40), `evidence_score` (weight 0.25), `workflow_score` (weight 0.20), and `efficiency_score` (weight 0.15).
2. THE Grader SHALL set `decision_score` to 1.0 if the final disposition matches the ground-truth disposition, and 0.0 otherwise.
3. THE Grader SHALL compute `evidence_score` by checking whether the agent's extracted facts and notes contain all required evidence fields specified in the fixture's ground truth.
4. THE Grader SHALL compute `workflow_score` by verifying that all required document checks were completed before the terminal action.
5. THE Grader SHALL compute `efficiency_score` based on the ratio of steps used to the step budget, rewarding agents that finish under budget.
6. THE Grader SHALL return a structured `GradeResult` object containing the composite score and all sub-scores for transparency.
7. WHEN the agent incorrectly approves an invoice with critical unresolved issues, THE Grader SHALL set `decision_score` to 0.0 and flag an `audit_failure` in the `GradeResult`.

---

### Requirement 8: Baseline Inference Script

**User Story:** As a hackathon participant, I want a reproducible baseline inference script that runs the agent against all three tasks, so that I can verify the environment works end-to-end and have a score to beat.

#### Acceptance Criteria

1. THE Baseline_Script SHALL accept environment variables `API_BASE_URL`, `MODEL_NAME`, and optionally `HF_TOKEN` for configuration.
2. THE Baseline_Script SHALL run each task in a fixed order with a fixed random seed for reproducibility.
3. THE Baseline_Script SHALL use a deterministic prompt template that instructs the agent to inspect documents before deciding.
4. THE Baseline_Script SHALL log each action, observation, and reward to stdout in a structured format.
5. THE Baseline_Script SHALL print a final summary table with per-task scores and an overall mean score.
6. WHEN the baseline agent runs the easy task, THE Baseline_Script SHALL achieve a score of 0.70 or higher.

---

### Requirement 9: Docker and Hugging Face Space Deployment

**User Story:** As a hackathon judge, I want to run AP-Resolve as a Docker container or Hugging Face Space, so that I can evaluate it in a standardized, reproducible environment.

#### Acceptance Criteria

1. THE AP_Environment SHALL be packaged with a `Dockerfile` that builds from `openenv-base:latest` and exposes port 8000.
2. THE AP_Environment SHALL include an `openenv.yaml` manifest with `spec_version`, `name`, `type`, `runtime`, `app`, and `port` fields.
3. WHEN the Docker container starts, THE AP_Environment SHALL be reachable at `GET /health` within 30 seconds.
4. THE AP_Environment SHALL include a `README.md` with sections covering: overview, observation space, action space, state model, task descriptions, reward design, grader logic, setup, Docker usage, HF Space deployment, baseline script usage, and baseline scores.
5. WHERE a `TASK_ID` environment variable is set, THE AP_Environment SHALL default all `reset()` calls without explicit `task_id` to that task.

---

### Requirement 10: Web Debug Interface

**User Story:** As a developer, I want an optional web interface to interact with the environment manually, so that I can debug cases and verify environment behavior without writing code.

#### Acceptance Criteria

1. WHERE the environment variable `ENABLE_WEB_INTERFACE=true` is set, THE AP_Environment SHALL serve a web UI at `/web` using the `create_web_interface_app` helper from `openenv-core`.
2. WHEN the web interface is active, THE AP_Environment SHALL display the current observation, extracted facts, action history, and steps remaining in real time.
3. WHEN the web interface is active, THE AP_Environment SHALL provide a form for submitting any valid APAction.
