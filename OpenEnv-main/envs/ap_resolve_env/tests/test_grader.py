"""Unit tests for Grader (Task 5.1)."""

import pytest

from envs.ap_resolve_env.graders import Grader
from envs.ap_resolve_env.models import APState
from envs.ap_resolve_env.tasks_registry import TaskRegistry


def make_state_and_fixture(task_id: str = "easy_straight_through"):
    registry = TaskRegistry()
    fixture = registry.get(task_id)
    state = APState(case_id=fixture.case_id, task_id=fixture.task_id)
    return state, fixture


class TestGrader:
    def setup_method(self):
        self.grader = Grader()

    def test_correct_decision_score_1(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        result = self.grader.grade(state, fixture)
        assert result.decision_score == 1.0

    def test_wrong_decision_score_0(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "reject_invoice"
        result = self.grader.grade(state, fixture)
        assert result.decision_score == 0.0

    def test_no_decision_score_0(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = None
        result = self.grader.grade(state, fixture)
        assert result.decision_score == 0.0

    def test_audit_failure_on_critical_case(self):
        state, fixture = make_state_and_fixture("medium_mismatch")
        state.decision = "approve_invoice"
        result = self.grader.grade(state, fixture)
        assert result.audit_failure is True
        assert result.decision_score == 0.0

    def test_no_audit_failure_on_non_critical(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        result = self.grader.grade(state, fixture)
        assert result.audit_failure is False

    def test_evidence_score_full(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        for field in fixture.required_evidence_fields:
            state.extracted_facts[field] = "some_value"
        result = self.grader.grade(state, fixture)
        assert result.evidence_score == pytest.approx(1.0)

    def test_evidence_score_partial(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        # Only provide first required field
        state.extracted_facts[fixture.required_evidence_fields[0]] = "val"
        result = self.grader.grade(state, fixture)
        expected = 1 / len(fixture.required_evidence_fields)
        assert result.evidence_score == pytest.approx(expected)

    def test_evidence_score_zero(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        result = self.grader.grade(state, fixture)
        assert result.evidence_score == pytest.approx(0.0)

    def test_workflow_score_full(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        for doc in fixture.required_checks:
            state.checks_completed[doc] = True
        result = self.grader.grade(state, fixture)
        assert result.workflow_score == pytest.approx(1.0)

    def test_workflow_score_partial(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        state.checks_completed[fixture.required_checks[0]] = True
        result = self.grader.grade(state, fixture)
        expected = 1 / len(fixture.required_checks)
        assert result.workflow_score == pytest.approx(expected)

    def test_efficiency_score_full(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        state.step_count = 0
        result = self.grader.grade(state, fixture)
        assert result.efficiency_score == pytest.approx(1.0)

    def test_efficiency_score_zero(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        state.step_count = fixture.max_steps
        result = self.grader.grade(state, fixture)
        assert result.efficiency_score == pytest.approx(0.0)

    def test_composite_score_formula(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        for field in fixture.required_evidence_fields:
            state.extracted_facts[field] = "val"
        for doc in fixture.required_checks:
            state.checks_completed[doc] = True
        state.step_count = 5

        result = self.grader.grade(state, fixture)
        expected = (
            0.40 * result.decision_score
            + 0.25 * result.evidence_score
            + 0.20 * result.workflow_score
            + 0.15 * result.efficiency_score
        )
        expected = max(0.0, min(1.0, expected))
        assert result.composite_score == pytest.approx(expected, abs=1e-6)

    def test_composite_score_clipped_to_range(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        result = self.grader.grade(state, fixture)
        assert 0.0 <= result.composite_score <= 1.0

    def test_grade_result_has_details(self):
        state, fixture = make_state_and_fixture("easy_straight_through")
        state.decision = "approve_invoice"
        result = self.grader.grade(state, fixture)
        assert isinstance(result.details, dict)
        assert "decision" in result.details
