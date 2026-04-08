"""Unit tests for DocumentWorkspace and ActionHandler (Task 3.1)."""

import pytest

from envs.ap_resolve_env.models import APAction, APState, CaseFixture
from envs.ap_resolve_env.workspace import DocumentWorkspace
from envs.ap_resolve_env.action_handler import ActionHandler
from envs.ap_resolve_env.tasks_registry import TaskRegistry


def make_workspace(task_id: str = "easy_straight_through") -> tuple[DocumentWorkspace, CaseFixture]:
    registry = TaskRegistry()
    fixture = registry.get(task_id)
    return DocumentWorkspace(fixture), fixture


class TestDocumentWorkspace:
    def test_list_available_returns_all_docs(self):
        ws, fixture = make_workspace()
        available = ws.list_available()
        assert set(available) == set(fixture.documents.keys())

    def test_open_valid_document_returns_content(self):
        ws, _ = make_workspace()
        content = ws.open("invoice")
        assert content is not None
        assert len(content) > 0

    def test_open_invalid_document_returns_none(self):
        ws, _ = make_workspace()
        result = ws.open("nonexistent_doc")
        assert result is None

    def test_already_opened_false_before_open(self):
        ws, _ = make_workspace()
        assert ws.already_opened("invoice") is False

    def test_already_opened_true_after_open(self):
        ws, _ = make_workspace()
        ws.open("invoice")
        assert ws.already_opened("invoice") is True

    def test_already_opened_false_for_other_doc(self):
        ws, _ = make_workspace()
        ws.open("invoice")
        assert ws.already_opened("purchase_order") is False

    def test_extract_field_before_open_returns_none(self):
        ws, _ = make_workspace()
        result = ws.extract_field("invoice_amount")
        assert result is None

    def test_extract_field_after_open_returns_value(self):
        ws, _ = make_workspace()
        ws.open("invoice")
        value = ws.extract_field("invoice_amount")
        assert value is not None
        assert isinstance(value, float)

    def test_extract_vendor_id(self):
        ws, _ = make_workspace()
        ws.open("invoice")
        vendor_id = ws.extract_field("vendor_id")
        assert vendor_id == "VENDOR-001"

    def test_extract_field_cached_in_extracted(self):
        ws, _ = make_workspace()
        ws.open("invoice")
        ws.extract_field("invoice_amount")
        assert "invoice_amount" in ws.get_extracted()

    def test_get_opened_names_empty_initially(self):
        ws, _ = make_workspace()
        assert ws.get_opened_names() == []

    def test_get_opened_names_after_open(self):
        ws, _ = make_workspace()
        ws.open("invoice")
        ws.open("purchase_order")
        assert set(ws.get_opened_names()) == {"invoice", "purchase_order"}


class TestActionHandler:
    def setup_method(self):
        self.registry = TaskRegistry()
        self.handler = ActionHandler()

    def _make_env(self, task_id: str = "easy_straight_through"):
        fixture = self.registry.get(task_id)
        workspace = DocumentWorkspace(fixture)
        state = APState(case_id=fixture.case_id, task_id=fixture.task_id)
        return workspace, state, fixture

    def test_open_document_valid(self):
        ws, state, fixture = self._make_env()
        action = APAction(action_type="open_document", target="invoice")
        result = self.handler.handle(action, ws, state, fixture)
        assert result.success is True
        assert result.opened_doc == "invoice"

    def test_open_document_invalid(self):
        ws, state, fixture = self._make_env()
        action = APAction(action_type="open_document", target="fake_doc")
        result = self.handler.handle(action, ws, state, fixture)
        assert result.success is False

    def test_extract_field_before_open(self):
        ws, state, fixture = self._make_env()
        action = APAction(action_type="extract_field", target="invoice_amount")
        result = self.handler.handle(action, ws, state, fixture)
        assert result.success is False

    def test_extract_field_after_open(self):
        ws, state, fixture = self._make_env()
        self.handler.handle(APAction(action_type="open_document", target="invoice"), ws, state, fixture)
        action = APAction(action_type="extract_field", target="invoice_amount")
        result = self.handler.handle(action, ws, state, fixture)
        assert result.success is True
        assert "invoice_amount" in state.extracted_facts

    def test_add_note_appends(self):
        ws, state, fixture = self._make_env()
        action = APAction(action_type="add_note", target="Invoice looks correct")
        result = self.handler.handle(action, ws, state, fixture)
        assert result.success is True
        assert "Invoice looks correct" in state.notes

    def test_terminal_action_sets_decision(self):
        ws, state, fixture = self._make_env()
        action = APAction(action_type="approve_invoice")
        result = self.handler.handle(action, ws, state, fixture)
        assert result.is_terminal is True
        assert state.decision == "approve_invoice"

    def test_request_vendor_info_sets_contacted(self):
        ws, state, fixture = self._make_env()
        action = APAction(action_type="request_vendor_info", target="Please confirm quantities")
        result = self.handler.handle(action, ws, state, fixture)
        assert result.success is True
        assert state.vendor_contacted is True

    def test_compare_fields_match(self):
        ws, state, fixture = self._make_env()
        # Extract same field twice under different names to test match
        state.extracted_facts["invoice_amount"] = 2592.0
        state.extracted_facts["po_amount"] = 2592.0
        action = APAction(
            action_type="compare_fields",
            target="invoice_amount",
            params={"field_b": "po_amount"},
        )
        result = self.handler.handle(action, ws, state, fixture)
        assert result.success is True
        assert "MATCH" in result.message

    def test_compare_fields_mismatch(self):
        ws, state, fixture = self._make_env()
        state.extracted_facts["invoice_amount"] = 3391.20
        state.extracted_facts["po_amount"] = 3002.40
        action = APAction(
            action_type="compare_fields",
            target="invoice_amount",
            params={"field_b": "po_amount"},
        )
        result = self.handler.handle(action, ws, state, fixture)
        assert result.success is True
        assert "MISMATCH" in result.message
