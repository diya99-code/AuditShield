"""Unit tests for AP-Resolve data models (Task 1.1)."""

import pytest
from pydantic import ValidationError

from envs.ap_resolve_env.models import (
    APAction,
    APObservation,
    APState,
    CaseFixture,
    GradeResult,
    TERMINAL_ACTIONS,
)


# ---------------------------------------------------------------------------
# APAction tests
# ---------------------------------------------------------------------------


class TestAPAction:
    def test_valid_action_types(self):
        valid_types = [
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
        for action_type in valid_types:
            action = APAction(action_type=action_type)
            assert action.action_type == action_type

    def test_invalid_action_type_raises(self):
        with pytest.raises(ValidationError):
            APAction(action_type="invalid_action")

    def test_empty_action_type_raises(self):
        with pytest.raises(ValidationError):
            APAction(action_type="")

    def test_optional_target_defaults_none(self):
        action = APAction(action_type="open_document")
        assert action.target is None

    def test_target_can_be_set(self):
        action = APAction(action_type="open_document", target="invoice")
        assert action.target == "invoice"

    def test_params_defaults_empty_dict(self):
        action = APAction(action_type="add_note")
        assert action.params == {}

    def test_params_can_be_set(self):
        action = APAction(action_type="add_note", params={"text": "some note"})
        assert action.params["text"] == "some note"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            APAction(action_type="open_document", unknown_field="x")

    def test_terminal_actions_set(self):
        assert TERMINAL_ACTIONS == {
            "approve_invoice",
            "hold_invoice",
            "reject_invoice",
            "escalate_case",
        }


# ---------------------------------------------------------------------------
# APObservation tests
# ---------------------------------------------------------------------------


class TestAPObservation:
    def test_required_fields_present(self):
        obs = APObservation(
            case_id="CASE-001",
            task_id="easy_straight_through",
            visible_documents=["invoice", "purchase_order"],
            steps_remaining=10,
            message="Ready",
        )
        assert obs.case_id == "CASE-001"
        assert obs.task_id == "easy_straight_through"
        assert obs.visible_documents == ["invoice", "purchase_order"]
        assert obs.steps_remaining == 10
        assert obs.message == "Ready"

    def test_optional_fields_default(self):
        obs = APObservation()
        assert obs.current_view is None
        assert obs.extracted_facts == {}
        assert obs.pending_issues == []
        assert obs.action_history == []
        assert obs.done is False
        assert obs.reward is None

    def test_all_required_fields_accessible(self):
        obs = APObservation(
            case_id="C1",
            task_id="t1",
            visible_documents=["invoice"],
            extracted_facts={"amount": 100},
            pending_issues=["mismatch"],
            action_history=["open_document:invoice"],
            steps_remaining=5,
            message="ok",
        )
        # All required fields from Requirement 2.3
        assert hasattr(obs, "case_id")
        assert hasattr(obs, "task_id")
        assert hasattr(obs, "visible_documents")
        assert hasattr(obs, "extracted_facts")
        assert hasattr(obs, "pending_issues")
        assert hasattr(obs, "action_history")
        assert hasattr(obs, "steps_remaining")
        assert hasattr(obs, "message")


# ---------------------------------------------------------------------------
# APState tests
# ---------------------------------------------------------------------------


class TestAPState:
    def test_required_fields_present(self):
        state = APState(
            case_id="CASE-001",
            task_id="easy_straight_through",
        )
        assert state.case_id == "CASE-001"
        assert state.task_id == "easy_straight_through"

    def test_optional_fields_default(self):
        state = APState()
        assert state.hidden_ground_truth == {}
        assert state.extracted_facts == {}
        assert state.checks_completed == {}
        assert state.decision is None
        assert state.notes == []
        assert state.vendor_contacted is False
        assert state.reward_so_far == 0.0
        assert state.step_count == 0

    def test_all_required_fields_accessible(self):
        state = APState()
        # All required fields from Requirement 2.4
        for field_name in [
            "case_id",
            "task_id",
            "hidden_ground_truth",
            "extracted_facts",
            "checks_completed",
            "notes",
            "vendor_contacted",
            "reward_so_far",
            "step_count",
        ]:
            assert hasattr(state, field_name), f"Missing field: {field_name}"

    def test_decision_can_be_set(self):
        state = APState()
        state.decision = "approve_invoice"
        assert state.decision == "approve_invoice"


# ---------------------------------------------------------------------------
# GradeResult tests
# ---------------------------------------------------------------------------


class TestGradeResult:
    def test_grade_result_fields(self):
        result = GradeResult(
            composite_score=0.75,
            decision_score=1.0,
            evidence_score=0.8,
            workflow_score=0.6,
            efficiency_score=0.9,
            audit_failure=False,
        )
        assert result.composite_score == 0.75
        assert result.decision_score == 1.0
        assert result.audit_failure is False
        assert result.details == {}
