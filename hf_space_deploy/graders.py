"""
Grader — deterministic, LLM-free scoring component.

Invoked at episode end to compute a structured GradeResult.
"""

from __future__ import annotations

from .models import APState, CaseFixture, GradeResult


class Grader:
    """Deterministic grader for AP-Resolve episodes."""

    WEIGHTS = {
        "decision": 0.40,
        "evidence": 0.25,
        "workflow": 0.20,
        "efficiency": 0.15,
    }

    def grade(self, state: APState, fixture: CaseFixture) -> GradeResult:
        """Compute a GradeResult from the final episode state."""
        decision_score = self._decision_score(state, fixture)
        evidence_score = self._evidence_score(state, fixture)
        workflow_score = self._workflow_score(state, fixture)
        efficiency_score = self._efficiency_score(state, fixture)
        audit_failure = self._audit_failure(state, fixture)

        composite = (
            self.WEIGHTS["decision"] * decision_score
            + self.WEIGHTS["evidence"] * evidence_score
            + self.WEIGHTS["workflow"] * workflow_score
            + self.WEIGHTS["efficiency"] * efficiency_score
        )
        composite = max(0.0, min(1.0, composite))

        return GradeResult(
            composite_score=composite,
            decision_score=decision_score,
            evidence_score=evidence_score,
            workflow_score=workflow_score,
            efficiency_score=efficiency_score,
            audit_failure=audit_failure,
            details={
                "decision": state.decision,
                "ground_truth": fixture.ground_truth_disposition,
                "required_evidence": fixture.required_evidence_fields,
                "extracted_facts_keys": list(state.extracted_facts.keys()),
                "notes_count": len(state.notes),
                "checks_completed": dict(state.checks_completed),
                "required_checks": fixture.required_checks,
                "step_count": state.step_count,
                "max_steps": fixture.max_steps,
            },
        )

    # ------------------------------------------------------------------
    # Sub-score computations
    # ------------------------------------------------------------------

    def _decision_score(self, state: APState, fixture: CaseFixture) -> float:
        """1.0 if disposition matches ground truth, 0.0 otherwise."""
        if state.decision is None:
            return 0.0
        # Audit failure: wrong approval on critical case → 0.0
        if state.decision == "approve_invoice" and fixture.has_critical_issues:
            return 0.0
        return 1.0 if state.decision == fixture.ground_truth_disposition else 0.0

    def _evidence_score(self, state: APState, fixture: CaseFixture) -> float:
        """Fraction of required_evidence_fields present in extracted_facts or notes."""
        required = fixture.required_evidence_fields
        if not required:
            return 1.0

        notes_text = " ".join(state.notes).lower()
        present = 0
        for field in required:
            if field in state.extracted_facts:
                present += 1
            elif field.lower() in notes_text:
                present += 1

        return present / len(required)

    def _workflow_score(self, state: APState, fixture: CaseFixture) -> float:
        """Fraction of required_checks completed before terminal action."""
        required = fixture.required_checks
        if not required:
            return 1.0

        completed = sum(
            1 for doc in required if state.checks_completed.get(doc, False)
        )
        return completed / len(required)

    def _efficiency_score(self, state: APState, fixture: CaseFixture) -> float:
        """Ratio of steps remaining to budget (higher = more efficient)."""
        steps_used = state.step_count
        max_steps = fixture.max_steps
        if max_steps <= 0:
            return 1.0
        steps_remaining = max(0, max_steps - steps_used)
        return steps_remaining / max_steps

    def _audit_failure(self, state: APState, fixture: CaseFixture) -> bool:
        """True if agent approved an invoice with critical unresolved issues."""
        return (
            state.decision == "approve_invoice"
            and fixture.has_critical_issues
        )
