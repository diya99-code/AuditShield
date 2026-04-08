"""
RewardCalculator — computes per-step reward delta.
Cumulative reward must be clipped to (0,1) using clip().
"""

from __future__ import annotations
from .models import APAction, APState, ActionResult, CaseFixture


class RewardCalculator:
    """Deterministic per-step reward calculator."""

    # ===== Constraints =====
    EPS = 0.01  # avoid exact 0.0 and 1.0

    # ===== Positive rewards =====
    OPEN_RELEVANT_DOC = 0.05
    EXTRACT_CRITICAL_FIELD = 0.05
    COMPLETE_COMPARISON = 0.10
    DETECT_ISSUE = 0.10
    COMPLETE_CHECK = 0.05
    CORRECT_DECISION = 0.40
    BUDGET_BONUS = 0.05

    # ===== Penalties =====
    REOPEN_DOC_PENALTY = -0.03
    IRRELEVANT_PENALTY = -0.05
    LOOPING_PENALTY = -0.10
    SKIP_CHECKS_PENALTY = -0.20
    WRONG_DECISION_PENALTY = -0.30
    UNSAFE_APPROVAL_PENALTY = -0.40
    STEP_OVERFLOW_PENALTY = -0.05  # applied once

    def compute(
        self,
        action: APAction,
        result: ActionResult,
        state: APState,
        fixture: CaseFixture,
    ) -> float:
        """Compute reward delta for one step."""

        delta = 0.0

        # ===== Invalid / irrelevant =====
        if result.is_invalid:
            delta += self.IRRELEVANT_PENALTY

        if result.is_irrelevant and not result.success:
            delta += self.IRRELEVANT_PENALTY

        # ===== Document interaction =====
        if result.opened_doc is not None:
            if result.reopened_doc:
                delta += self.REOPEN_DOC_PENALTY
            else:
                delta += self.OPEN_RELEVANT_DOC

        # ===== Extraction =====
        if result.extracted_critical_field:
            delta += self.EXTRACT_CRITICAL_FIELD

        # ===== Comparison =====
        if result.completed_comparison:
            delta += self.COMPLETE_COMPARISON

        # ===== Issue detection =====
        if result.identified_core_issue:
            delta += self.DETECT_ISSUE

        # ===== Required check completion =====
        if getattr(result, "completed_check", False):
            delta += self.COMPLETE_CHECK

        # ===== Loop penalty (only once per detection) =====
        if state.loop_detected:
            delta += self.LOOPING_PENALTY
            state.loop_detected = False

        # ===== Step overflow penalty (only once) =====
        if state.step_count == fixture.soft_step_threshold + 1:
            delta += self.STEP_OVERFLOW_PENALTY

        # ===== Terminal step =====
        if result.is_terminal:
            delta += self._terminal_reward(action, state, fixture)

        return delta

    def _terminal_reward(
        self,
        action: APAction,
        state: APState,
        fixture: CaseFixture,
    ) -> float:
        """Final decision reward logic."""

        disposition = action.action_type
        correct = disposition == fixture.ground_truth_disposition

        # ===== Unsafe approval (highest penalty) =====
        if disposition == "approve_invoice" and fixture.has_critical_issues:
            return self.UNSAFE_APPROVAL_PENALTY

        # ===== Wrong decision =====
        if not correct:
            safe_actions = {"hold_invoice", "reject_invoice", "escalate_case"}

            # Safer mistake (not approving bad invoice)
            if disposition in safe_actions:
                return -0.10

            return self.WRONG_DECISION_PENALTY

        # ===== Correct decision =====
        reward = self.CORRECT_DECISION

        # ===== Check completion penalty =====
        all_checks_done = all(
            state.checks_completed.get(c, False)
            for c in fixture.required_checks
        )

        if not all_checks_done:
            reward += self.SKIP_CHECKS_PENALTY

        # ===== Efficiency bonus =====
        if state.step_count <= fixture.soft_step_threshold:
            reward += self.BUDGET_BONUS

        return reward

    @classmethod
    def clip(cls, value: float) -> float:
        """
        Clip cumulative reward into (0,1) strictly.
        """
        if value <= cls.EPS:
            return cls.EPS
        if value >= 1.0 - cls.EPS:
            return 1.0 - cls.EPS
        return round(value, 4)