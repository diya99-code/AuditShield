"""
RewardCalculator — computes per-step reward delta.

All cumulative rewards are clipped to [0.0, 1.0] in APState.reward_so_far.
"""

from __future__ import annotations

from .models import APAction, APState, ActionResult, CaseFixture


class RewardCalculator:
    """Computes the reward delta for a single step."""

    # Reward deltas (from design document)
    OPEN_RELEVANT_DOC = 0.05
    EXTRACT_CRITICAL_FIELD = 0.10
    COMPLETE_REQUIRED_COMPARISON = 0.10
    IDENTIFY_CORE_ISSUE = 0.15
    REOPEN_DOC = -0.03
    IRRELEVANT_ACTION = -0.05
    INVALID_ACTION = -0.05
    OVER_SOFT_THRESHOLD = -0.02
    CORRECT_DISPOSITION_MAX = 0.40
    CORRECT_DISPOSITION_MIN = 0.30
    EVIDENCE_BACKED_NOTE_BONUS = 0.10
    AUDIT_RISK_PENALTY = -0.40

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

        # Invalid action penalty
        if result.is_invalid:
            delta += self.INVALID_ACTION
            return delta

        # Irrelevant action penalty
        if result.is_irrelevant and not result.success:
            delta += self.IRRELEVANT_ACTION
            return delta

        # Open relevant document (first time)
        if result.opened_doc is not None:
            if result.reopened_doc:
                delta += self.REOPEN_DOC
            else:
                delta += self.OPEN_RELEVANT_DOC

        # Extract critical field
        if result.extracted_critical_field:
            delta += self.EXTRACT_CRITICAL_FIELD

        # Complete required field comparison
        if result.completed_comparison:
            delta += self.COMPLETE_REQUIRED_COMPARISON

        # Identify core issue
        if result.identified_core_issue:
            delta += self.IDENTIFY_CORE_ISSUE

        # Over soft step threshold penalty
        if state.step_count > fixture.soft_step_threshold:
            delta += self.OVER_SOFT_THRESHOLD

        # Terminal action rewards
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
        correct = disposition == fixture.ground_truth_disposition

        # Audit-risk: approve on critical case
        if disposition == "approve_invoice" and fixture.has_critical_issues:
            return self.AUDIT_RISK_PENALTY

        if correct:
            # Bonus if evidence-backed notes present
            evidence_present = any(
                field in state.extracted_facts or
                any(field in note for note in state.notes)
                for field in fixture.required_evidence_fields
            )
            base = self.CORRECT_DISPOSITION_MAX
            if evidence_present:
                base = min(1.0, base + self.EVIDENCE_BACKED_NOTE_BONUS)
            return base
        else:
            return 0.0

    @staticmethod
    def clip(value: float) -> float:
        """Clip a cumulative reward to [0.0, 1.0]."""
        return max(0.0, min(1.0, value))
