"""
RewardCalculator — computes per-step reward delta.

All cumulative rewards are clipped to [0.0, 1.0] in APState.reward_so_far.
"""

from __future__ import annotations

from .models import APAction, APState, ActionResult, CaseFixture


class RewardCalculator:
    """Computes the reward delta for a single step."""

    # Positive rewards
    OPEN_RELEVANT_DOC = 0.05
    EXTRACT_CRITICAL_FIELD = 0.05
    COMPLETE_COMPARISON = 0.10
    DETECT_ISSUE = 0.10
    COMPLETE_CHECK = 0.05
    CORRECT_DECISION = 0.40
    BUDGET_BONUS = 0.05

    # Penalties
    REOPEN_DOC_PENALTY = -0.03
    IRRELEVANT_PENALTY = -0.05
    LOOPING_PENALTY = -0.10
    SKIP_CHECKS_PENALTY = -0.20
    WRONG_DECISION_PENALTY = -0.30
    UNSAFE_APPROVAL_PENALTY = -0.40
    OVER_SOFT_THRESHOLD = -0.02

    def compute(
        self,
        action: APAction,
        result: ActionResult,
        state: APState,
        fixture: CaseFixture,
    ) -> float:
        """
        Compute the reward delta for this step.

        Returns a float delta (may be negative). The caller is responsible
        for accumulating into state.reward_so_far and clipping.
        """
        delta = 0.0

        if result.is_invalid:
            delta += self.IRRELEVANT_PENALTY
            return delta

        if result.is_irrelevant and not result.success:
            delta += self.IRRELEVANT_PENALTY
            return delta

        # Relevant Doc
        if result.opened_doc is not None:
            if result.reopened_doc:
                delta += self.REOPEN_DOC_PENALTY
            else:
                delta += self.OPEN_RELEVANT_DOC

        # Extract Field
        if result.extracted_critical_field:
            delta += self.EXTRACT_CRITICAL_FIELD

        # Comparison
        if result.completed_comparison:
            delta += self.COMPLETE_COMPARISON

        # Issue detection
        if result.identified_core_issue:
            delta += self.DETECT_ISSUE

        # Loop detected penalty
        if state.loop_detected:
            delta += self.LOOPING_PENALTY
            state.loop_detected = False # Reset flag after penalty applied

        # Step count penalty
        if state.step_count > fixture.soft_step_threshold:
            delta += self.OVER_SOFT_THRESHOLD

        # Terminal rewards
        if result.is_terminal:
            delta += self._terminal_reward(action, state, fixture)

        return delta

    def _terminal_reward(
        self,
        action: APAction,
        state: APState,
        fixture: CaseFixture,
    ) -> float:
        """Compute the terminal reward component."""
        disposition = action.action_type
        correct = (disposition == fixture.ground_truth_disposition)
        
        # Penalties
        if disposition == "approve_invoice" and fixture.has_critical_issues:
            return self.UNSAFE_APPROVAL_PENALTY
        
        if not correct:
            return self.WRONG_DECISION_PENALTY

        # Correct Decision
        reward = self.CORRECT_DECISION

        # Check for required checks
        all_checks_done = all(state.checks_completed.get(c, False) for c in fixture.required_checks)
        if not all_checks_done:
            reward += self.SKIP_CHECKS_PENALTY
        
        # Budget bonus
        if state.step_count <= fixture.soft_step_threshold:
            reward += self.BUDGET_BONUS

        return reward

    @staticmethod
    def clip(value: float) -> float:
        """Clip a cumulative reward to (0.01, 0.99)."""
        return max(0.01, min(0.99, value))
