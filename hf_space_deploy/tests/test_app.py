"""Unit tests for the FastAPI app (Task 8.1)."""

import pytest
from fastapi.testclient import TestClient

from envs.ap_resolve_env.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert data.get("status") in ("healthy", "HEALTHY")


class TestSchemaEndpoint:
    def test_schema_endpoint_exists(self, client):
        response = client.get("/schema")
        assert response.status_code == 200

    def test_schema_has_action_and_observation(self, client):
        response = client.get("/schema")
        data = response.json()
        assert "action" in data
        assert "observation" in data
