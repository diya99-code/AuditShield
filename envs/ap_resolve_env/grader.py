"""
Grader — deterministic, LLM-free scoring component.

Invoked at episode end to compute a structured GradeResult.
"""

from __future__ import annotations

from .models import APState, CaseFixture, GradeResult


class Grader:
    """Deterministic, task-specific grader for AP-Resolve."""

    WEIGHTS = {
        "decision": 0.40,
        "checks": 0.25,
        "issues": 0.20,
        "efficiency": 0.15,
    }

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
        
        return GradeResult(
            composite_score=max(0.0, min(1.0, composite)),
            decision_score=decision_score,
            evidence_score=issues_score, # Mapping issues to evidence field
            workflow_score=checks_score,  # Mapping checks to workflow field
            efficiency_score=efficiency_score,
            audit_failure=(state.decision == "approve_invoice" and fixture.has_critical_issues),
            details={
                "task_id": fixture.task_id,
                "difficulty": fixture.difficulty,
                "checks_completed": list(state.checks_completed.keys()),
                "issues_identified": state.identified_issues,
                "loop_detected": state.loop_detected,
                "step_count": state.step_count,
                "budget": fixture.soft_step_threshold
            }
        )

    def _decision_score(self, state: APState, fixture: CaseFixture) -> float:
        if state.decision is None: return 0.0
        if state.decision == "approve_invoice" and fixture.has_critical_issues: return 0.0
        return 1.0 if state.decision == fixture.ground_truth_disposition else 0.0

    def _checks_score(self, state: APState, fixture: CaseFixture) -> float:
        required = fixture.required_checks
        if not required: return 1.0
        completed = sum(1 for c in required if state.checks_completed.get(c, False))
        return completed / len(required)

    def _issues_score(self, state: APState, fixture: CaseFixture) -> float:
        required = fixture.required_issues
        if not required:
            # If no issues expected, penalty if any were raised wrongly
            return 1.0 if not state.identified_issues else 0.5
        
        found = sum(1 for i in required if i in state.identified_issues)
        return found / len(required)

    def _efficiency_score(self, state: APState, fixture: CaseFixture) -> float:
        score = 1.0
        # Budget penalty
        if state.step_count > fixture.soft_step_threshold:
            score -= 0.5
        # Redundant behavior penalty
        if state.loop_detected:
            score -= 0.5
        return max(0.0, score)
