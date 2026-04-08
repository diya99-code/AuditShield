"""Unit tests for TaskRegistry (Task 2.1)."""

import pytest

from envs.ap_resolve_env.tasks_registry import TaskRegistry, TASK_IDS
from envs.ap_resolve_env.models import CaseFixture

REQUIRED_DOC_KEYS = {
    "invoice",
    "purchase_order",
    "goods_receipt",
    "vendor_master",
    "ap_policy",
    "invoice_history",
}


class TestTaskRegistry:
    def setup_method(self):
        self.registry = TaskRegistry()

    def test_get_valid_task_ids(self):
        for task_id in TASK_IDS:
            fixture = self.registry.get(task_id)
            assert isinstance(fixture, CaseFixture)
            assert fixture.task_id == task_id

    def test_get_invalid_task_id_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            self.registry.get("nonexistent_task")

    def test_get_empty_task_id_raises(self):
        with pytest.raises(ValueError):
            self.registry.get("")

    def test_random_returns_valid_task(self):
        for _ in range(20):
            fixture = self.registry.random()
            assert isinstance(fixture, CaseFixture)
            assert fixture.task_id in TASK_IDS

    def test_list_ids_returns_all_tasks(self):
        ids = self.registry.list_ids()
        assert set(ids) == set(TASK_IDS)
        assert len(ids) == 3

    def test_all_fixtures_load_without_error(self):
        # Just constructing the registry loads all fixtures
        registry = TaskRegistry()
        for task_id in TASK_IDS:
            fixture = registry.get(task_id)
            assert fixture is not None

    def test_easy_fixture_has_correct_disposition(self):
        fixture = self.registry.get("easy_straight_through")
        assert fixture.ground_truth_disposition == "approve_invoice"
        assert fixture.has_critical_issues is False

    def test_medium_fixture_has_critical_issues(self):
        fixture = self.registry.get("medium_mismatch")
        assert fixture.has_critical_issues is True
        assert fixture.ground_truth_disposition in {"hold_invoice", "reject_invoice"}

    def test_hard_fixture_has_critical_issues(self):
        fixture = self.registry.get("hard_duplicate_partial")
        assert fixture.has_critical_issues is True

    def test_all_fixtures_have_required_document_keys(self):
        for task_id in TASK_IDS:
            fixture = self.registry.get(task_id)
            missing = REQUIRED_DOC_KEYS - set(fixture.documents.keys())
            assert not missing, f"{task_id} missing docs: {missing}"

    def test_fixtures_have_non_empty_documents(self):
        for task_id in TASK_IDS:
            fixture = self.registry.get(task_id)
            for doc_name, content in fixture.documents.items():
                assert content.strip(), f"{task_id}/{doc_name} is empty"

    def test_fixtures_have_max_steps_gt_soft_threshold(self):
        for task_id in TASK_IDS:
            fixture = self.registry.get(task_id)
            assert fixture.max_steps > fixture.soft_step_threshold
