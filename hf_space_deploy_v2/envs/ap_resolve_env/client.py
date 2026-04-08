"""
APClient — client for the AP-Resolve environment.

Extends EnvClient[APAction, APObservation, APState] for use in RL training loops.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import APAction, APObservation, APState


class APClient(EnvClient[APAction, APObservation, APState]):
    """
    Client for the AP-Resolve environment.

    Example (async):
        >>> async with APClient(base_url="ws://localhost:8000") as env:
        ...     result = await env.reset(task_id="easy_straight_through")
        ...     while not result.done:
        ...         action = APAction(action_type="open_document", target="invoice")
        ...         result = await env.step(action)

    Example (sync):
        >>> client = APClient(base_url="ws://localhost:8000").sync()
        >>> with client:
        ...     result = client.reset()
        ...     result = client.step(APAction(action_type="approve_invoice"))
    """

    def _step_payload(self, action: APAction) -> Dict[str, Any]:
        """Convert APAction to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[APObservation]:
        """Parse server response into StepResult[APObservation]."""
        obs_data = payload.get("observation", payload)
        done = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward", None))

        obs = APObservation(**obs_data) if isinstance(obs_data, dict) else obs_data
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> APState:
        """Parse server state response into APState."""
        return APState(**payload)
