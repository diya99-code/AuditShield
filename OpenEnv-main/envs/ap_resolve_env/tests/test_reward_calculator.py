"""Unit tests for RewardCalculator (Task 4.1)."""

import pytest

from envs.ap_resolve_env.models import APAction, APState, ActionResult, CaseFixture
from envs.ap_resolve_env.rewards import RewardCalculator
from envs.ap_resolve_env.tasks_registry import TaskRegistry


def make_state(task_id: str = "easy_straight_through") -> tuple[APState, CaseFixture]:
    registry = TaskRegistry()
    fixture = registry.get(task_id)
    state = APState(case_id=fixture.case_id, task_id=fixture.task_id)
    return state, fixture


class TestRewardCalculator:
    def setup_method(self):
        self.calc = RewardCalculator()

    def test_open_relevant_doc_first_time(self):
        state, fixture = make_state()
        action = APAction(action_type="open_document", target="invoice")
        result = ActionResult(success=True, message="ok", opened_doc="invoice", reopened_doc=False)
        delta = self.calc.compute(action, result, state, fixture)
        assert delta == pytest.approx(RewardCalculator.OPEN_RELEVANT_DOC)

    def test_reopen_doc_penalty(self):
        state, fixture = make_state()
        action = APAction(action_type="open_document", target="invoice")
        result = ActionResult(success=True, message="ok", opened_doc="invoice", reopened_doc=True)
        delta = self.calc.compute(action, result, state, fixture)
        assert delta == pytest.approx(RewardCalculator.REOPEN_DOC)

    def test_extract_critical_field_reward(self):
        state, fixture = make_state()
        action = APAction(action_type="extract_field", target="invoice_amount")
        result = ActionResult(success=True, message="ok", extracted_critical_field=True)
        delta = self.calc.compute(action, result, state, fixture)
        assert delta == pytest.approx(RewardCalculator.EXTRACT_CRITICAL_FIELD)

    def test_complete_comparison_reward(self):
        state, fixture = make_state()
        action = APAction(action_type="compare_fields", target="invoice_amount", params={"field_b": "po_amount"})
        result = ActionResult(success=True, message="ok", completed_comparison=True)
        delta = self.calc.compute(action, result, state, fixture)
        assert delta == pytest.approx(RewardCalculator.COMPLETE_REQUIRED_COMPARISON)

    def test_identify_core_issue_reward(self):
        state, fixture = make_state("medium_mismatch")
        action = APAction(action_type="compare_fields", target="invoice_amount", params={"field_b": "po_amount"})
        result = ActionResult(success=True, message="ok", identified_core_issue=True)
        delta = self.calc.compute(action, result, state, fixture)
        assert delta == pytest.approx(RewardCalculator.IDENTIFY_CORE_ISSUE)

    def test_irrelevant_action_penalty(self):
        state, fixture = make_state()
        action = APAction(action_type="open_document", target="fake")
        result = ActionResult(success=False, message="not found", is_irrelevant=True)
        delta = self.calc.compute(action, result, state, fixture)
        assert delta == pytest.approx(RewardCalculator.IRRELEVANT_ACTION)

    def test_over_soft_threshold_penalty(self):
        state, fixture = make_state()
        state.step_count = fixture.soft_step_threshold + 1
        action = APAction(action_type="add_note", target="note")
        result = ActionResult(success=True, message="ok")
        delta = self.calc.compute(action, result, state, fixture)
        assert delta == pytest.approx(RewardCalculator.OVER_SOFT_THRESHOLD)

    def test_correct_terminal_reward(self):
        state, fixture = make_state("easy_straight_through")
        state.extracted_facts["invoice_amount"] = 2592.0
        action = APAction(action_type="approve_invoice")
        result = ActionResult(success=True, message="ok", is_terminal=True)
        delta = self.calc.compute(action, result, state, fixture)
        assert delta >= RewardCalculator.CORRECT_DISPOSITION_MIN

    def test_audit_risk_penalty_on_critical_case(self):
        state, fixture = make_state("medium_mismatch")
        action = APAction(action_type="approve_invoice")
        result = ActionResult(success=True, message="ok", is_terminal=True)
        delta = self.calc.compute(action, result, state, fixture)
        assert delta == pytest.approx(RewardCalculator.AUDIT_RISK_PENALTY)

    def test_clip_below_zero(self):
        assert RewardCalculator.clip(-0.5) == 0.0

    def test_clip_above_one(self):
        assert RewardCalculator.clip(1.5) == 1.0

    def test_clip_within_range(self):
        assert RewardCalculator.clip(0.5) == pytest.approx(0.5)

    def test_cumulative_reward_stays_clipped(self):
        state, fixture = make_state()
        # Simulate many positive rewards
        cumulative = 0.0
        for _ in range(20):
            action = APAction(action_type="extract_field", target="invoice_amount")
            result = ActionResult(success=True, message="ok", extracted_critical_field=True)
            delta = self.calc.compute(action, result, state, fixture)
            cumulative = RewardCalculator.clip(cumulative + delta)
        assert 0.0 <= cumulative <= 1.0
