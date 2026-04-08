"""
Grader — deterministic, LLM-free scoring component.
Invoked at episode end to compute a structured GradeResult.
"""

from __future__ import annotations

from .models import APState, CaseFixture, GradeResult


class Grader:
    """Deterministic, task-specific grader for AuditShield."""

    WEIGHTS = {
        "decision": 0.40,
        "checks": 0.25,
        "issues": 0.20,
        "efficiency": 0.15,
    }

    MIN_SCORE = 0.01
    MAX_SCORE = 0.99

    def grade(self, state: APState, fixture: CaseFixture) -> GradeResult:
        decision_score = self._decision_score(state, fixture)
        checks_score = self._checks_score(state, fixture)
        issues_score = self._issues_score(state, fixture)
        efficiency_score = self._efficiency_score(state, fixture)

        composite = (
            self.WEIGHTS["decision"] * decision_score
            + self.WEIGHTS["checks"] * checks_score
            + self.WEIGHTS["issues"] * issues_score
            + self.WEIGHTS["efficiency"] * efficiency_score
        )

        composite = self._clamp(composite)

        return GradeResult(
            composite_score=composite,
            decision_score=decision_score,
            evidence_score=issues_score,
            workflow_score=checks_score,
            efficiency_score=efficiency_score,
            audit_failure=(
                state.decision == "approve_invoice" and fixture.has_critical_issues
            ),
            details={
                "task_id": fixture.task_id,
                "difficulty": fixture.difficulty,
                "ground_truth_disposition": fixture.ground_truth_disposition,
                "agent_decision": state.decision,
                "checks_completed": list(state.checks_completed.keys()),
                "required_checks": fixture.required_checks,
                "issues_identified": state.identified_issues,
                "required_issues": fixture.required_issues,
                "loop_detected": state.loop_detected,
                "step_count": state.step_count,
                "soft_step_threshold": fixture.soft_step_threshold,
            },
        )

    def _decision_score(self, state: APState, fixture: CaseFixture) -> float:
        """
        Scores whether the final decision matches the expected disposition.
        Strongly penalizes unsafe approval on critical-issue cases.
        """
        if state.decision is None:
            return self.MIN_SCORE

        if state.decision == "approve_invoice" and fixture.has_critical_issues:
            return self.MIN_SCORE

        if state.decision == fixture.ground_truth_disposition:
            return self.MAX_SCORE

        # Partial credit for "safe but not perfect" handling
        safe_non_approval_actions = {"hold_invoice", "reject_invoice", "escalate_case"}
        if (
            fixture.ground_truth_disposition in safe_non_approval_actions
            and state.decision in safe_non_approval_actions
        ):
            return 0.40

        return 0.10

    def _checks_score(self, state: APState, fixture: CaseFixture) -> float:
        """
        Scores completion of required checks.
        """
        required = fixture.required_checks or []
        if not required:
            return self.MAX_SCORE

        completed = sum(
            1 for check in required if state.checks_completed.get(check, False)
        )
        ratio = completed / len(required)
        return self._scaled_ratio(ratio)

    def _issues_score(self, state: APState, fixture: CaseFixture) -> float:
        """
        Scores issue identification quality.
        - If no issues are expected, rewards clean handling.
        - If issues are expected, rewards finding the right ones.
        """
        required = fixture.required_issues or []
        identified = set(state.identified_issues or [])

        if not required:
            # Clean case: best if no false issues are raised
            if not identified:
                return self.MAX_SCORE
            if len(identified) == 1:
                return 0.50
            return 0.20

        required_set = set(required)
        found = len(required_set.intersection(identified))
        missed = len(required_set - identified)
        extra = len(identified - required_set)

        ratio = found / len(required_set)

        # Penalize noise / incorrect issue flags a bit
        penalty = 0.0
        if missed > 0:
            penalty += 0.10 * missed / len(required_set)
        if extra > 0:
            penalty += 0.05 * extra

        score = ratio - penalty
        return self._clamp(score)

    def _efficiency_score(self, state: APState, fixture: CaseFixture) -> float:
        """
        Scores whether the agent behaved efficiently and avoided loops.
        """
        score = self.MAX_SCORE
        step_count = state.step_count or 0
        threshold = fixture.soft_step_threshold or 10

        if step_count > threshold:
            overage = step_count - threshold
            score -= min(0.40, 0.05 * overage)

        if state.loop_detected:
            score -= 0.30

        # Mild penalty if no decision was made
        if state.decision is None:
            score -= 0.20

        return self._clamp(score)

    def _scaled_ratio(self, ratio: float) -> float:
        """
        Maps [0,1] ratio into (0,1), avoiding exact 0.0 and 1.0.
        """
        ratio = max(0.0, min(1.0, ratio))
        scaled = self.MIN_SCORE + ratio * (self.MAX_SCORE - self.MIN_SCORE)
        return self._clamp(scaled)

    def _clamp(self, value: float) -> float:
        """
        Ensures score is strictly between 0 and 1.
        """
        if value <= self.MIN_SCORE:
            return self.MIN_SCORE
        if value >= self.MAX_SCORE:
            return self.MAX_SCORE
        return round(value, 4)