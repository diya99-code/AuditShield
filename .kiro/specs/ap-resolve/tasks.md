# Implementation Plan: AP-Resolve

## Overview

Build AP-Resolve as an OpenEnv-compatible environment following the framework's standard structure: typed models, FastAPI server, Docker packaging, and a baseline inference script. Tasks are ordered so each step produces runnable, integrated code.

## Tasks

- [x] 1. Project scaffold and typed data models
  - Create `envs/ap_resolve_env/` directory structure matching OpenEnv conventions
  - Implement `APAction`, `APObservation`, `APState`, `CaseFixture`, `GradeResult` in `models.py` using Pydantic v2
  - Implement `__init__.py` exporting all public types
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 1.1 Write unit tests for model schema validation
  - Test `APAction` rejects invalid `action_type` values
  - Test all required fields are present on `APObservation` and `APState`
  - _Requirements: 2.1, 2.3, 2.4_

- [x] 1.2 Write property test: APObservation schema invariant (Property 2)
  - **Property 2: APObservation schema invariant**
  - **Validates: Requirements 2.3**

- [x] 1.3 Write property test: APState schema invariant (Property 3)
  - **Property 3: APState schema invariant**
  - **Validates: Requirements 2.4**

- [x] 2. Task fixtures and TaskRegistry
  - Author three JSON fixture files under `fixtures/`: `easy_straight_through.json`, `medium_mismatch.json`, `hard_duplicate_partial.json`
  - Each fixture must include all six document types, critical fields, required checks, ground-truth disposition, and `has_critical_issues` flag
  - Implement `TaskRegistry` in `tasks.py` with `get(task_id)`, `random()`, and `list_ids()` methods
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 2.1 Write unit tests for TaskRegistry
  - Test `get()` with valid and invalid task IDs
  - Test `random()` always returns a valid task ID
  - Test all three fixture files load without error
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 2.2 Write property test: Task ID round-trip (Property 5)
  - **Property 5: Task ID round-trip**
  - **Validates: Requirements 3.2**

- [x] 2.3 Write property test: All fixtures contain required document keys (Property 6)
  - **Property 6: All fixtures contain required document keys**
  - **Validates: Requirements 3.7**

- [x] 3. DocumentWorkspace and ActionHandler
  - Implement `DocumentWorkspace` in `workspace.py` with `open()`, `extract_field()`, `list_available()`, and `already_opened()` methods
  - Implement `ActionHandler` in `action_handler.py` dispatching all 12 action types to workspace/state mutations, returning `ActionResult`
  - Handle all error cases: invalid document name, extract before open, compare un-extracted fields
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9_

- [x] 3.1 Write unit tests for DocumentWorkspace
  - Test `open()` with valid and invalid document names
  - Test `extract_field()` before and after opening a document
  - Test `already_opened()` tracking
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 3.2 Write property test: open_document populates current_view (Property 7)
  - **Property 7: open_document populates current_view**
  - **Validates: Requirements 4.1**

- [x] 3.3 Write property test: extract_field round-trip (Property 8)
  - **Property 8: extract_field round-trip**
  - **Validates: Requirements 4.3**

- [x] 3.4 Write property test: add_note appends to notes list (Property 9)
  - **Property 9: add_note appends to notes list**
  - **Validates: Requirements 4.9**

- [x] 4. RewardCalculator
  - Implement `RewardCalculator` in `rewards.py` applying all reward rules from the design document
  - Implement cumulative reward clipping to [0.0, 1.0]
  - Implement audit-risk penalty (-0.40) for wrong approval on critical cases
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9_

- [x] 4.1 Write unit tests for RewardCalculator
  - Test each reward rule in isolation with a known fixture
  - Test cumulative clipping stays in [0.0, 1.0]
  - Test audit-risk penalty triggers correctly
  - _Requirements: 6.1–6.9_

- [x] 4.2 Write property test: Reward always in [0.0, 1.0] (Property 12)
  - **Property 12: Reward is always in [0.0, 1.0]**
  - **Validates: Requirements 6.9**

- [x] 5. Grader
  - Implement `Grader` in `graders.py` computing `decision_score`, `evidence_score`, `workflow_score`, `efficiency_score`, and `composite_score`
  - Implement `audit_failure` flag for wrong approvals on critical cases
  - Return structured `GradeResult` dataclass
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 5.1 Write unit tests for Grader
  - Test each sub-score with known fixture inputs and expected outputs
  - Test `audit_failure` flag on critical case wrong approval
  - _Requirements: 7.1–7.7_

- [x] 5.2 Write property test: Grader composite score equals weighted sub-scores (Property 13)
  - **Property 13: Grader composite score equals weighted sub-scores**
  - **Validates: Requirements 7.1**

- [x] 5.3 Write property test: Grader decision score is binary (Property 14)
  - **Property 14: Grader decision score is binary**
  - **Validates: Requirements 7.2**

- [x] 5.4 Write property test: Grader evidence score reflects required field coverage (Property 15)
  - **Property 15: Grader evidence score reflects required field coverage**
  - **Validates: Requirements 7.3**

- [x] 5.5 Write property test: Audit failure on wrong approval of critical case (Property 16)
  - **Property 16: Audit failure on wrong approval of critical case**
  - **Validates: Requirements 5.3, 7.7**

- [x] 6. APEnvironment core — reset and step
  - Implement `APEnvironment` in `server/ap_environment.py` extending `openenv_core.Environment`
  - Wire `TaskRegistry`, `DocumentWorkspace`, `ActionHandler`, `RewardCalculator`, and `Grader` together
  - Implement `reset(task_id?)`, `step(action)`, and `state` property
  - Handle terminal actions, step-budget exhaustion, and post-terminal step calls
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 6.1 Write unit tests for APEnvironment reset and step
  - Test `reset()` clears all state fields
  - Test `step()` after `done=True` returns `done=True` and `reward=0.0`
  - Test each terminal action type sets `done=True`
  - _Requirements: 1.4, 1.5, 5.1, 5.4_

- [x] 6.2 Write property test: Reset produces clean state (Property 1)
  - **Property 1: Reset produces clean state**
  - **Validates: Requirements 1.4**

- [x] 6.3 Write property test: Terminal actions set done=True and record decision (Property 10)
  - **Property 10: Terminal actions set done=True and record decision**
  - **Validates: Requirements 5.1, 5.4**

- [x] 6.4 Write property test: Step budget exhaustion terminates episode (Property 11)
  - **Property 11: Step budget exhaustion terminates episode**
  - **Validates: Requirements 5.5**

- [x] 6.5 Write property test: Invalid action type rejected without termination (Property 4)
  - **Property 4: Invalid action type is rejected without termination**
  - **Validates: Requirements 2.5**

- [x] 7. Checkpoint — core environment tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. FastAPI server and APClient
  - Implement `server/app.py` using `create_fastapi_app(env, APAction, APObservation)` from `openenv_core`
  - Implement `client.py` extending `EnvClient[APAction, APObservation, APState]`
  - Add `ENABLE_WEB_INTERFACE` env-var check and `create_web_interface_app` integration
  - Add `TASK_ID` env-var support for default task selection
  - _Requirements: 1.6, 9.3, 9.5, 10.1_

- [x] 8.1 Write unit tests for FastAPI app
  - Test `/health` endpoint returns 200
  - Test `/reset`, `/step`, `/state` endpoints with valid payloads
  - _Requirements: 1.6, 9.3_

- [x] 9. Baseline inference script
  - Implement `inference.py` reading `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env vars
  - Implement deterministic prompt template and fixed-seed task ordering
  - Log each action, observation, and reward to stdout in structured format
  - Print final summary table with per-task scores and overall mean
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 9.1 Write integration test for baseline easy-task score
  - Run baseline agent on `easy_straight_through` task and assert score >= 0.70
  - _Requirements: 8.6_

- [x] 10. Docker packaging and openenv.yaml
  - Write `server/Dockerfile` building from `openenv-base:latest`, installing requirements, copying code, exposing port 8000
  - Write `openenv.yaml` with `spec_version`, `name`, `type`, `runtime`, `app`, `port` fields
  - Write `server/requirements.txt` with all runtime dependencies
  - _Requirements: 9.1, 9.2_

- [x] 11. README
  - Write `README.md` with all required sections: overview, observation space, action space, state model, task descriptions, reward design, grader logic, setup, Docker usage, HF Space deployment, baseline script usage, baseline scores, known limitations
  - _Requirements: 9.4_

- [x] 12. Final checkpoint — all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `max_examples=100`
- Checkpoints at tasks 7 and 12 ensure incremental validation
- The audit-risk mechanic (task 4, reward -0.40) is a key differentiator — implement it carefully
