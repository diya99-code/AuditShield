"""Unit tests for APEnvironment reset and step (Task 6.1)."""

import pytest

from envs.ap_resolve_env.models import APAction, TERMINAL_ACTIONS
from envs.ap_resolve_env.server.ap_environment import APEnvironment
from envs.ap_resolve_env.tasks_registry import TASK_IDS


class TestAPEnvironmentReset:
    def setup_method(self):
        self.env = APEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="easy_straight_through")
        assert obs is not None
        assert obs.case_id != ""
        assert obs.task_id == "easy_straight_through"

    def test_reset_clears_state(self):
        # Do some steps first
        self.env.reset(task_id="easy_straight_through")
        self.env.step(APAction(action_type="open_document", target="invoice"))
        self.env.step(APAction(action_type="add_note", target="some note"))

        # Reset
        obs = self.env.reset(task_id="easy_straight_through")
        state = self.env.state

        assert state.extracted_facts == {}
        assert state.notes == []
        assert state.decision is None
        assert state.vendor_contacted is False
        assert state.step_count == 0
        assert state.reward_so_far == 0.0

    def test_reset_with_different_task_ids(self):
        for task_id in TASK_IDS:
            obs = self.env.reset(task_id=task_id)
            assert obs.task_id == task_id

    def test_reset_without_task_id_returns_valid_task(self):
        obs = self.env.reset()
        assert obs.task_id in TASK_IDS

    def test_reset_visible_documents_populated(self):
        obs = self.env.reset(task_id="easy_straight_through")
        assert len(obs.visible_documents) > 0

    def test_reset_steps_remaining_equals_max_steps(self):
        obs = self.env.reset(task_id="easy_straight_through")
        fixture = self.env._fixture
        assert obs.steps_remaining == fixture.max_steps


class TestAPEnvironmentStep:
    def setup_method(self):
        self.env = APEnvironment()
        self.env.reset(task_id="easy_straight_through")

    def test_step_open_document(self):
        obs = self.env.step(APAction(action_type="open_document", target="invoice"))
        assert obs.done is False
        assert obs.current_view is not None

    def test_step_increments_step_count(self):
        self.env.step(APAction(action_type="open_document", target="invoice"))
        assert self.env.state.step_count == 1

    def test_step_after_done_returns_done(self):
        self.env.step(APAction(action_type="approve_invoice"))
        obs = self.env.step(APAction(action_type="open_document", target="invoice"))
        assert obs.done is True
        assert obs.reward == 0.0

    def test_terminal_action_sets_done(self):
        for terminal in TERMINAL_ACTIONS:
            self.env.reset(task_id="easy_straight_through")
            obs = self.env.step(APAction(action_type=terminal))
            assert obs.done is True

    def test_terminal_action_records_decision(self):
        self.env.step(APAction(action_type="approve_invoice"))
        assert self.env.state.decision == "approve_invoice"

    def test_step_budget_exhaustion(self):
        fixture = self.env._fixture
        # Fill up all steps with non-terminal actions
        for _ in range(fixture.max_steps):
            obs = self.env.step(APAction(action_type="add_note", target="note"))
            if obs.done:
                break
        assert obs.done is True
        assert obs.reward == 0.0

    def test_state_property_returns_ap_state(self):
        from envs.ap_resolve_env.models import APState
        assert isinstance(self.env.state, APState)

    def test_correct_disposition_gives_positive_reward(self):
        # Open required docs and extract evidence for easy task
        self.env.step(APAction(action_type="open_document", target="invoice"))
        self.env.step(APAction(action_type="extract_field", target="invoice_amount"))
        self.env.step(APAction(action_type="extract_field", target="vendor_id"))
        self.env.step(APAction(action_type="open_document", target="purchase_order"))
        self.env.step(APAction(action_type="extract_field", target="po_amount"))
        self.env.step(APAction(action_type="open_document", target="goods_receipt"))
        obs = self.env.step(APAction(action_type="approve_invoice"))
        assert obs.done is True
        assert obs.reward > 0.0
