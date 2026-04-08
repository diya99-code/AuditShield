"""
Property-based tests for AP-Resolve (Hypothesis).

Each test is annotated with the design property it validates.
Feature: ap-resolve
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from envs.ap_resolve_env.models import (
    APAction,
    APObservation,
    APState,
    ActionType,
    CaseFixture,
    GradeResult,
    TERMINAL_ACTIONS,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ACTION_TYPES: list[str] = [
    "open_document",
    "extract_field",
    "compare_fields",
    "calculate_total",
    "check_policy",
    "search_history",
    "request_vendor_info",
    "add_note",
    "approve_invoice",
    "hold_invoice",
    "reject_invoice",
    "escalate_case",
]

action_type_st = st.sampled_from(ACTION_TYPES)

ap_observation_st = st.builds(
    APObservation,
    case_id=st.text(min_size=1, max_size=20),
    task_id=st.text(min_size=1, max_size=30),
    visible_documents=st.lists(st.text(min_size=1, max_size=20), max_size=6),
    current_view=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    extracted_facts=st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(max_size=50), st.integers(), st.floats(allow_nan=False)),
        max_size=5,
    ),
    pending_issues=st.lists(st.text(min_size=1, max_size=50), max_size=5),
    action_history=st.lists(st.text(min_size=1, max_size=50), max_size=10),
    steps_remaining=st.integers(min_value=0, max_value=30),
    message=st.text(max_size=200),
)

ap_state_st = st.builds(
    APState,
    case_id=st.text(min_size=1, max_size=20),
    task_id=st.text(min_size=1, max_size=30),
    hidden_ground_truth=st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.text(max_size=50),
        max_size=5,
    ),
    extracted_facts=st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(max_size=50), st.integers()),
        max_size=5,
    ),
    checks_completed=st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.booleans(),
        max_size=5,
    ),
    decision=st.one_of(st.none(), st.sampled_from(ACTION_TYPES[-4:])),
    notes=st.lists(st.text(min_size=1, max_size=100), max_size=5),
    vendor_contacted=st.booleans(),
    reward_so_far=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    step_count=st.integers(min_value=0, max_value=30),
)

grade_result_st = st.builds(
    GradeResult,
    composite_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    decision_score=st.sampled_from([0.0, 1.0]),
    evidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    workflow_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    efficiency_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    audit_failure=st.booleans(),
)


# ---------------------------------------------------------------------------
# Property 2: APObservation schema invariant
# Feature: ap-resolve, Property 2: APObservation schema invariant
# ---------------------------------------------------------------------------


@given(ap_observation_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_2_ap_observation_schema_invariant(obs: APObservation):
    """
    For any APObservation, all required fields must be present and accessible.
    Validates: Requirements 2.3
    """
    required_fields = [
        "case_id",
        "task_id",
        "visible_documents",
        "extracted_facts",
        "pending_issues",
        "action_history",
        "steps_remaining",
        "message",
    ]
    for field_name in required_fields:
        assert hasattr(obs, field_name), f"Missing field: {field_name}"
        # Field must be accessible (not raise)
        _ = getattr(obs, field_name)

    # Type checks
    assert isinstance(obs.case_id, str)
    assert isinstance(obs.task_id, str)
    assert isinstance(obs.visible_documents, list)
    assert isinstance(obs.extracted_facts, dict)
    assert isinstance(obs.pending_issues, list)
    assert isinstance(obs.action_history, list)
    assert isinstance(obs.steps_remaining, int)
    assert isinstance(obs.message, str)


# ---------------------------------------------------------------------------
# Property 3: APState schema invariant
# Feature: ap-resolve, Property 3: APState schema invariant
# ---------------------------------------------------------------------------


@given(ap_state_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_3_ap_state_schema_invariant(state: APState):
    """
    For any APState, all required fields must be present and accessible.
    Validates: Requirements 2.4
    """
    required_fields = [
        "case_id",
        "task_id",
        "hidden_ground_truth",
        "extracted_facts",
        "checks_completed",
        "notes",
        "vendor_contacted",
        "reward_so_far",
        "step_count",
    ]
    for field_name in required_fields:
        assert hasattr(state, field_name), f"Missing field: {field_name}"
        _ = getattr(state, field_name)

    # Type checks
    assert isinstance(state.case_id, str)
    assert isinstance(state.task_id, str)
    assert isinstance(state.hidden_ground_truth, dict)
    assert isinstance(state.extracted_facts, dict)
    assert isinstance(state.checks_completed, dict)
    assert isinstance(state.notes, list)
    assert isinstance(state.vendor_contacted, bool)
    assert isinstance(state.reward_so_far, float)
    assert isinstance(state.step_count, int)


# ---------------------------------------------------------------------------
# Property 5: Task ID round-trip
# Feature: ap-resolve, Property 5: Task ID round-trip
# ---------------------------------------------------------------------------

from envs.ap_resolve_env.tasks_registry import TaskRegistry, TASK_IDS

_registry = TaskRegistry()

_REQUIRED_DOC_KEYS = {
    "invoice",
    "purchase_order",
    "goods_receipt",
    "vendor_master",
    "ap_policy",
    "invoice_history",
}


@given(st.sampled_from(TASK_IDS))
@settings(max_examples=100)
def test_property_5_task_id_round_trip(task_id: str):
    """
    For any valid task_id in the registry, reset(task_id=task_id) should return
    an observation whose task_id equals the requested task_id.
    Here we test the registry layer: get(task_id).task_id == task_id.
    Validates: Requirements 3.2
    """
    fixture = _registry.get(task_id)
    assert fixture.task_id == task_id


# ---------------------------------------------------------------------------
# Property 6: All fixtures contain required document keys
# Feature: ap-resolve, Property 6: All fixtures contain required document keys
# ---------------------------------------------------------------------------


@given(st.sampled_from(TASK_IDS))
@settings(max_examples=100)
def test_property_6_fixtures_contain_required_document_keys(task_id: str):
    """
    For any fixture loaded from the task registry, the documents dict should
    contain all of: invoice, purchase_order, goods_receipt, vendor_master,
    ap_policy, invoice_history.
    Validates: Requirements 3.7
    """
    fixture = _registry.get(task_id)
    missing = _REQUIRED_DOC_KEYS - set(fixture.documents.keys())
    assert not missing, f"Fixture '{task_id}' missing document keys: {missing}"


# ---------------------------------------------------------------------------
# Properties 7, 8, 9: DocumentWorkspace and ActionHandler
# ---------------------------------------------------------------------------

from envs.ap_resolve_env.workspace import DocumentWorkspace
from envs.ap_resolve_env.action_handler import ActionHandler
from envs.ap_resolve_env.models import APAction, APState

_handler = ActionHandler()


def _make_env_from_fixture(fixture):
    ws = DocumentWorkspace(fixture)
    state = APState(case_id=fixture.case_id, task_id=fixture.task_id)
    return ws, state


# ---------------------------------------------------------------------------
# Property 7: open_document populates current_view
# Feature: ap-resolve, Property 7: open_document populates current_view
# ---------------------------------------------------------------------------


@given(st.sampled_from(TASK_IDS))
@settings(max_examples=100)
def test_property_7_open_document_populates_current_view(task_id: str):
    """
    For any case and any document name in visible_documents, submitting
    open_document should return a non-empty content string.
    Validates: Requirements 4.1
    """
    fixture = _registry.get(task_id)
    ws, state = _make_env_from_fixture(fixture)

    for doc_name in fixture.documents.keys():
        content = ws.open(doc_name)
        assert content is not None, f"open({doc_name}) returned None"
        assert len(content.strip()) > 0, f"open({doc_name}) returned empty content"


# ---------------------------------------------------------------------------
# Property 8: extract_field round-trip
# Feature: ap-resolve, Property 8: extract_field round-trip
# ---------------------------------------------------------------------------


@given(st.sampled_from(TASK_IDS))
@settings(max_examples=100)
def test_property_8_extract_field_round_trip(task_id: str):
    """
    For any case where a document has been opened and a valid field is extracted,
    the field should appear in extracted_facts and state should reflect the same value.
    Validates: Requirements 4.3
    """
    fixture = _registry.get(task_id)
    ws, state = _make_env_from_fixture(fixture)

    # Open invoice and extract a known field
    ws.open("invoice")
    value = ws.extract_field("vendor_id")
    if value is not None:
        # Simulate state update (as ActionHandler does)
        state.extracted_facts["vendor_id"] = value
        # Round-trip: extracted value in workspace == value in state
        assert ws.get_extracted().get("vendor_id") == state.extracted_facts["vendor_id"]


# ---------------------------------------------------------------------------
# Property 9: add_note appends to notes list
# Feature: ap-resolve, Property 9: add_note appends to notes list
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(TASK_IDS),
    st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5),
)
@settings(max_examples=100)
def test_property_9_add_note_appends_to_notes_list(task_id: str, notes: list):
    """
    For any note text submitted via add_note, the notes list in APState
    should grow by exactly one entry containing that text.
    Validates: Requirements 4.9
    """
    fixture = _registry.get(task_id)
    ws, state = _make_env_from_fixture(fixture)

    for note_text in notes:
        before_len = len(state.notes)
        action = APAction(action_type="add_note", target=note_text)
        result = _handler.handle(action, ws, state, fixture)
        assert result.success is True
        assert len(state.notes) == before_len + 1
        assert state.notes[-1] == note_text


# ---------------------------------------------------------------------------
# Property 12: Reward is always in [0.0, 1.0]
# Feature: ap-resolve, Property 12: Reward is always in [0.0, 1.0]
# ---------------------------------------------------------------------------

from envs.ap_resolve_env.rewards import RewardCalculator
from envs.ap_resolve_env.models import ActionResult

_reward_calc = RewardCalculator()

action_result_st = st.builds(
    ActionResult,
    success=st.booleans(),
    message=st.text(max_size=50),
    is_invalid=st.booleans(),
    is_irrelevant=st.booleans(),
    opened_doc=st.one_of(st.none(), st.sampled_from(["invoice", "purchase_order", "goods_receipt"])),
    reopened_doc=st.booleans(),
    extracted_critical_field=st.booleans(),
    completed_comparison=st.booleans(),
    identified_core_issue=st.booleans(),
    is_terminal=st.just(False),  # keep non-terminal for simplicity
)


@given(
    st.sampled_from(TASK_IDS),
    st.lists(
        st.builds(
            APAction,
            action_type=st.sampled_from(ACTION_TYPES[:-4]),  # non-terminal
            target=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
        ),
        min_size=1,
        max_size=25,
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_reward_always_in_range(task_id: str, actions: list):
    """
    For any sequence of actions in any episode, the cumulative reward_so_far
    should always remain in the range [0.0, 1.0].
    Validates: Requirements 6.9
    """
    fixture = _registry.get(task_id)
    ws, state = _make_env_from_fixture(fixture)

    for action in actions:
        result = _handler.handle(action, ws, state, fixture)
        delta = _reward_calc.compute(action, result, state, fixture)
        state.reward_so_far = RewardCalculator.clip(state.reward_so_far + delta)
        state.step_count += 1

        assert 0.0 <= state.reward_so_far <= 1.0, (
            f"reward_so_far={state.reward_so_far} out of [0,1] after action {action.action_type}"
        )


# ---------------------------------------------------------------------------
# Properties 13–16: Grader
# ---------------------------------------------------------------------------

from envs.ap_resolve_env.graders import Grader

_grader = Grader()

TERMINAL_ACTION_TYPES = ["approve_invoice", "hold_invoice", "reject_invoice", "escalate_case"]
CRITICAL_TASK_IDS = ["medium_mismatch", "hard_duplicate_partial"]
NON_CRITICAL_TASK_IDS = ["easy_straight_through"]


# ---------------------------------------------------------------------------
# Property 13: Grader composite score equals weighted sub-scores
# Feature: ap-resolve, Property 13: Grader composite score equals weighted sub-scores
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(TASK_IDS),
    st.sampled_from(TERMINAL_ACTION_TYPES),
    st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100)
def test_property_13_composite_score_equals_weighted_sub_scores(
    task_id: str, decision: str, step_count: int
):
    """
    For any GradeResult, composite_score == 0.40*decision + 0.25*evidence +
    0.20*workflow + 0.15*efficiency, clipped to [0,1].
    Validates: Requirements 7.1
    """
    fixture = _registry.get(task_id)
    state = APState(
        case_id=fixture.case_id,
        task_id=fixture.task_id,
        decision=decision,
        step_count=step_count,
    )
    result = _grader.grade(state, fixture)

    expected = (
        0.40 * result.decision_score
        + 0.25 * result.evidence_score
        + 0.20 * result.workflow_score
        + 0.15 * result.efficiency_score
    )
    expected = max(0.0, min(1.0, expected))
    assert abs(result.composite_score - expected) < 1e-9


# ---------------------------------------------------------------------------
# Property 14: Grader decision score is binary
# Feature: ap-resolve, Property 14: Grader decision score is binary
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(TASK_IDS),
    st.sampled_from(TERMINAL_ACTION_TYPES),
)
@settings(max_examples=100)
def test_property_14_decision_score_is_binary(task_id: str, decision: str):
    """
    For any grading call, decision_score is exactly 1.0 or 0.0.
    Validates: Requirements 7.2
    """
    fixture = _registry.get(task_id)
    state = APState(
        case_id=fixture.case_id,
        task_id=fixture.task_id,
        decision=decision,
    )
    result = _grader.grade(state, fixture)
    assert result.decision_score in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Property 15: Grader evidence score reflects required field coverage
# Feature: ap-resolve, Property 15: Grader evidence score reflects required field coverage
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(TASK_IDS),
    st.sampled_from(TERMINAL_ACTION_TYPES),
)
@settings(max_examples=100)
def test_property_15_evidence_score_reflects_coverage(task_id: str, decision: str):
    """
    For any set of extracted facts, evidence_score equals the fraction of
    required_evidence_fields present, in [0.0, 1.0].
    Validates: Requirements 7.3
    """
    fixture = _registry.get(task_id)
    state = APState(
        case_id=fixture.case_id,
        task_id=fixture.task_id,
        decision=decision,
    )
    # Provide a random subset of required evidence fields
    import random
    provided = random.sample(
        fixture.required_evidence_fields,
        k=random.randint(0, len(fixture.required_evidence_fields)),
    )
    for field in provided:
        state.extracted_facts[field] = "test_value"

    result = _grader.grade(state, fixture)

    expected_fraction = len(provided) / len(fixture.required_evidence_fields)
    assert result.evidence_score == pytest.approx(expected_fraction, abs=1e-9)
    assert 0.0 <= result.evidence_score <= 1.0


# ---------------------------------------------------------------------------
# Property 16: Audit failure on wrong approval of critical case
# Feature: ap-resolve, Property 16: Audit failure on wrong approval of critical case
# ---------------------------------------------------------------------------


@given(st.sampled_from(CRITICAL_TASK_IDS))
@settings(max_examples=100)
def test_property_16_audit_failure_on_critical_case(task_id: str):
    """
    For any case where has_critical_issues=True and agent submits approve_invoice,
    GradeResult should have audit_failure=True and decision_score=0.0.
    Validates: Requirements 5.3, 7.7
    """
    fixture = _registry.get(task_id)
    assert fixture.has_critical_issues is True

    state = APState(
        case_id=fixture.case_id,
        task_id=fixture.task_id,
        decision="approve_invoice",
    )
    result = _grader.grade(state, fixture)
    assert result.audit_failure is True
    assert result.decision_score == 0.0


# ---------------------------------------------------------------------------
# Properties 1, 4, 10, 11: APEnvironment
# ---------------------------------------------------------------------------

from envs.ap_resolve_env.server.ap_environment import APEnvironment

TERMINAL_TYPES = list(TERMINAL_ACTIONS)
NON_TERMINAL_TYPES = [t for t in ACTION_TYPES if t not in TERMINAL_ACTIONS]


# ---------------------------------------------------------------------------
# Property 1: Reset produces clean state
# Feature: ap-resolve, Property 1: Reset produces clean state
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(TASK_IDS),
    st.lists(
        st.builds(
            APAction,
            action_type=st.sampled_from(NON_TERMINAL_TYPES),
            target=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
        ),
        min_size=0,
        max_size=5,
    ),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_1_reset_produces_clean_state(task_id: str, actions: list):
    """
    For any sequence of actions, calling reset() should return a clean state:
    empty action_history, extracted_facts={}, vendor_contacted=False, decision=None.
    Validates: Requirements 1.4
    """
    env = APEnvironment()
    env.reset(task_id=task_id)

    for action in actions:
        obs = env.step(action)
        if obs.done:
            break

    # Now reset
    obs = env.reset(task_id=task_id)
    state = env.state

    assert state.extracted_facts == {}
    assert state.notes == []
    assert state.decision is None
    assert state.vendor_contacted is False
    assert state.step_count == 0
    assert state.reward_so_far == 0.0


# ---------------------------------------------------------------------------
# Property 10: Terminal actions set done=True and record decision
# Feature: ap-resolve, Property 10: Terminal actions set done=True and record decision
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(TASK_IDS),
    st.sampled_from(TERMINAL_TYPES),
)
@settings(max_examples=100)
def test_property_10_terminal_actions_set_done(task_id: str, terminal_type: str):
    """
    For any terminal action type, the returned observation should have done=True,
    and state().decision should equal the submitted action_type.
    Validates: Requirements 5.1, 5.4
    """
    env = APEnvironment()
    env.reset(task_id=task_id)
    obs = env.step(APAction(action_type=terminal_type))
    assert obs.done is True
    assert env.state.decision == terminal_type


# ---------------------------------------------------------------------------
# Property 11: Step budget exhaustion terminates episode
# Feature: ap-resolve, Property 11: Step budget exhaustion terminates episode
# ---------------------------------------------------------------------------


@given(st.sampled_from(TASK_IDS))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_11_step_budget_exhaustion(task_id: str):
    """
    For any episode where the agent takes max_steps non-terminal actions,
    the environment should return done=True and reward=0.0 on the final step.
    Validates: Requirements 5.5
    """
    env = APEnvironment()
    env.reset(task_id=task_id)
    fixture = env._fixture
    max_steps = fixture.max_steps

    last_obs = None
    for _ in range(max_steps):
        last_obs = env.step(APAction(action_type="add_note", target="note"))
        if last_obs.done:
            break

    assert last_obs is not None
    assert last_obs.done is True
    assert last_obs.reward == 0.0


# ---------------------------------------------------------------------------
# Property 4: Invalid action type is rejected without termination
# Feature: ap-resolve, Property 4: Invalid action type is rejected without termination
# Note: APAction validates action_type at construction time via Pydantic.
# We test that the environment handles the validation error gracefully.
# ---------------------------------------------------------------------------


@given(st.sampled_from(TASK_IDS))
@settings(max_examples=50)
def test_property_4_invalid_action_type_rejected(task_id: str):
    """
    Invalid action_type strings are rejected by Pydantic before reaching the env.
    The environment should remain alive (not done) after a valid non-terminal action.
    Validates: Requirements 2.5
    """
    from pydantic import ValidationError

    env = APEnvironment()
    env.reset(task_id=task_id)

    # Confirm invalid action_type raises ValidationError (not env crash)
    with pytest.raises(ValidationError):
        APAction(action_type="not_a_valid_action")

    # Confirm env is still alive after a valid non-terminal action
    obs = env.step(APAction(action_type="add_note", target="test"))
    assert obs.done is False
