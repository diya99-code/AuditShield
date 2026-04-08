"""
ActionHandler — dispatches APAction to workspace/state mutations.

Returns an ActionResult for each action.
"""

from __future__ import annotations

from typing import Any, Dict

from .models import APAction, APState, ActionResult, CaseFixture, TERMINAL_ACTIONS
from .workspace import DocumentWorkspace


class ActionHandler:
    """Dispatches each APAction to the appropriate handler."""

    def handle(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        """Execute the action and return an ActionResult."""
        dispatch = {
            "open_document": self._open_document,
            "extract_field": self._extract_field,
            "compare_fields": self._compare_fields,
            "calculate_total": self._calculate_total,
            "check_policy": self._check_policy,
            "search_history": self._search_history,
            "request_vendor_info": self._request_vendor_info,
            "add_note": self._add_note,
            "approve_invoice": self._terminal,
            "hold_invoice": self._terminal,
            "reject_invoice": self._terminal,
            "escalate_case": self._terminal,
        }
        handler = dispatch.get(action.action_type)
        if handler is None:
            # Should not happen since APAction validates action_type
            return ActionResult(
                success=False,
                message=f"Unknown action type: {action.action_type}",
                is_invalid=True,
            )
        return handler(action, workspace, state, fixture)

    # ------------------------------------------------------------------
    # Document actions
    # ------------------------------------------------------------------

    def _open_document(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        doc_name = action.target or ""
        if not doc_name:
            return ActionResult(
                success=False,
                message="open_document requires a 'target' specifying the document name.",
                is_irrelevant=True,
            )

        already = workspace.already_opened(doc_name)
        content = workspace.open(doc_name)

        if content is None:
            available = workspace.list_available()
            return ActionResult(
                success=False,
                message=(
                    f"Document '{doc_name}' not found. "
                    f"Available documents: {available}"
                ),
                is_irrelevant=True,
            )

        state.checks_completed[doc_name] = True

        return ActionResult(
            success=True,
            message=f"Opened document: {doc_name}",
            opened_doc=doc_name,
            reopened_doc=already,
        )

    def _extract_field(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        field = action.target or action.params.get("field", "")
        if not field:
            return ActionResult(
                success=False,
                message="extract_field requires a 'target' specifying the field name.",
            )

        # Check a document is open
        if not workspace.get_opened_names():
            return ActionResult(
                success=False,
                message="No document is open. Use open_document first.",
            )

        value = workspace.extract_field(field)
        if value is None:
            return ActionResult(
                success=False,
                message=f"Field '{field}' not found in the current document.",
                is_irrelevant=True,
            )

        state.extracted_facts[field] = value
        is_critical = field in fixture.critical_fields

        return ActionResult(
            success=True,
            message=f"Extracted '{field}': {value}",
            new_facts={field: value},
            extracted_critical_field=is_critical,
        )

    def _compare_fields(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        params = action.params or {}
        field_a = action.target or params.get("field_a", "")
        field_b = params.get("field_b", "")

        if not field_a or not field_b:
            return ActionResult(
                success=False,
                message=(
                    "compare_fields requires 'target' (field_a) and "
                    "params.field_b to be set."
                ),
            )

        facts = state.extracted_facts
        missing = [f for f in [field_a, field_b] if f not in facts]
        if missing:
            return ActionResult(
                success=False,
                message=f"Cannot compare: fields not yet extracted: {missing}",
            )

        val_a = facts[field_a]
        val_b = facts[field_b]

        try:
            a_num = float(val_a)
            b_num = float(val_b)
            delta = round(a_num - b_num, 4)
            if delta == 0:
                result_str = f"MATCH: {field_a}={val_a} == {field_b}={val_b}"
                matched = True
            else:
                pct = abs(delta / b_num * 100) if b_num != 0 else float("inf")
                result_str = (
                    f"MISMATCH: {field_a}={val_a}, {field_b}={val_b}, "
                    f"delta={delta} ({pct:.1f}%)"
                )
                matched = False
        except (TypeError, ValueError):
            matched = str(val_a).strip().lower() == str(val_b).strip().lower()
            result_str = (
                f"MATCH: {field_a}='{val_a}' == {field_b}='{val_b}'"
                if matched
                else f"MISMATCH: {field_a}='{val_a}' != {field_b}='{val_b}'"
            )

        # Check if this comparison is a required check
        is_required = (
            field_a in fixture.critical_fields and field_b in fixture.critical_fields
        )

        # Detect core issue identification
        identified_issue = (not matched) and is_required

        return ActionResult(
            success=True,
            message=result_str,
            completed_comparison=is_required,
            identified_core_issue=identified_issue,
        )

    def _calculate_total(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        facts = state.extracted_facts
        # Try to sum line item amounts from extracted facts
        total = 0.0
        found_any = False
        for key, val in facts.items():
            if "amount" in key.lower() or "total" in key.lower():
                try:
                    total += float(val)
                    found_any = True
                except (TypeError, ValueError):
                    pass

        if not found_any:
            return ActionResult(
                success=False,
                message=(
                    "No numeric amount fields found in extracted facts. "
                    "Extract invoice_amount or line item amounts first."
                ),
                is_irrelevant=True,
            )

        return ActionResult(
            success=True,
            message=f"Calculated total from extracted facts: ${total:,.2f}",
            new_facts={"calculated_total": total},
        )

    def _check_policy(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        rule = action.target or action.params.get("rule", "")
        policy_content = fixture.documents.get("ap_policy", "")

        if not policy_content:
            return ActionResult(
                success=False,
                message="AP policy document not available.",
                is_irrelevant=True,
            )

        if rule:
            # Return lines containing the rule keyword
            lines = [
                line.strip()
                for line in policy_content.splitlines()
                if rule.lower() in line.lower()
            ]
            if lines:
                return ActionResult(
                    success=True,
                    message=f"Policy rule(s) matching '{rule}':\n" + "\n".join(lines),
                )
            return ActionResult(
                success=True,
                message=f"No policy rule found matching '{rule}'. Full policy:\n{policy_content}",
            )

        return ActionResult(
            success=True,
            message=f"AP Policy:\n{policy_content}",
        )

    def _search_history(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        history_content = fixture.documents.get("invoice_history", "")
        if not history_content:
            return ActionResult(
                success=False,
                message="Invoice history not available.",
                is_irrelevant=True,
            )

        duplicate_detected = "duplicate" in history_content.lower()
        return ActionResult(
            success=True,
            message=history_content,
            new_facts={"duplicate_flag": duplicate_detected},
            identified_core_issue=duplicate_detected,
        )

    def _request_vendor_info(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        state.vendor_contacted = True
        vendor_response = fixture.vendor_response or (
            "Thank you for your inquiry. We will respond within 1 business day."
        )
        return ActionResult(
            success=True,
            message=f"Vendor contacted. Response: {vendor_response}",
        )

    def _add_note(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        note_text = action.target or action.params.get("text", "")
        if not note_text:
            return ActionResult(
                success=False,
                message="add_note requires a 'target' or params.text with the note content.",
                is_irrelevant=True,
            )
        state.notes.append(note_text)
        return ActionResult(
            success=True,
            message=f"Note added: {note_text}",
        )

    # ------------------------------------------------------------------
    # Terminal actions
    # ------------------------------------------------------------------

    def _terminal(
        self,
        action: APAction,
        workspace: DocumentWorkspace,
        state: APState,
        fixture: CaseFixture,
    ) -> ActionResult:
        state.decision = action.action_type
        return ActionResult(
            success=True,
            message=f"Terminal action submitted: {action.action_type}",
            is_terminal=True,
        )
