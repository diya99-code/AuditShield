"""
APEnvironment — standalone version for HF Space deployment.

Imports from the ap_resolve_env package (parent directory is on sys.path
via the Dockerfile's PYTHONPATH setting).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

# When running as `uvicorn server.app:app --app-dir /app/env`, the cwd is
# /app/env and the parent of server/ is already importable as a package.
# We just need to ensure the env root is on sys.path.
_env_root = str(Path(__file__).parent.parent)
if _env_root not in sys.path:
    sys.path.insert(0, _env_root)

# Now import using the package name (ap_resolve_env is a package in /app/env)
from openenv.core.env_server.interfaces import Environment

# Use relative-style imports via the package — works because /app/env is on PYTHONPATH
# and ap_resolve_env/ has an __init__.py
import importlib

_pkg = importlib.import_module("ap_resolve_env.action_handler")
ActionHandler = _pkg.ActionHandler

_pkg = importlib.import_module("ap_resolve_env.graders")
Grader = _pkg.Grader

_pkg = importlib.import_module("ap_resolve_env.models")
APAction = _pkg.APAction
APObservation = _pkg.APObservation
APState = _pkg.APState
TERMINAL_ACTIONS = _pkg.TERMINAL_ACTIONS

_pkg = importlib.import_module("ap_resolve_env.rewards")
RewardCalculator = _pkg.RewardCalculator

_pkg = importlib.import_module("ap_resolve_env.tasks_registry")
TaskRegistry = _pkg.TaskRegistry

_pkg = importlib.import_module("ap_resolve_env.workspace")
DocumentWorkspace = _pkg.DocumentWorkspace


class APEnvironment(Environment):
    """AP-Resolve OpenEnv environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task_registry = TaskRegistry()
        self._state: APState = APState()
        self._workspace: Optional[DocumentWorkspace] = None
        self._fixture = None
        self._done: bool = False
        self._action_handler = ActionHandler()
        self._reward_calc = RewardCalculator()
        self._grader = Grader()
        self._default_task_id: Optional[str] = os.environ.get("TASK_ID")

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> APObservation:
        resolved_task_id = task_id or self._default_task_id
        if resolved_task_id:
            fixture = self._task_registry.get(resolved_task_id)
        else:
            fixture = self._task_registry.random()

        self._fixture = fixture
        self._workspace = DocumentWorkspace(fixture)
        self._done = False
        self._state = APState(
            case_id=fixture.case_id,
            task_id=fixture.task_id,
            hidden_ground_truth={
                "disposition": fixture.ground_truth_disposition,
                "has_critical_issues": fixture.has_critical_issues,
            },
        )
        return self._build_observation(
            message=f"Episode started. Case: {fixture.case_id}. Task: {fixture.task_id}. "
                    f"Available documents: {list(fixture.documents.keys())}",
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: APAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> APObservation:
        if self._done:
            return self._build_observation(
                message="Episode is already done. Call reset() to start a new episode.",
                done=True,
                reward=0.0,
            )

        assert self._fixture is not None, "Call reset() before step()"
        assert self._workspace is not None

        self._state.step_count += 1
        result = self._action_handler.handle(
            action, self._workspace, self._state, self._fixture
        )
        delta = self._reward_calc.compute(action, result, self._state, self._fixture)
        self._state.reward_so_far = RewardCalculator.clip(
            self._state.reward_so_far + delta
        )

        if result.is_terminal:
            self._done = True
            grade = self._grader.grade(self._state, self._fixture)
            return self._build_observation(
                message=result.message,
                done=True,
                reward=grade.composite_score,
                current_view=self._workspace.get_current_content(),
            )

        if self._state.step_count >= self._fixture.max_steps:
            self._done = True
            return self._build_observation(
                message="Step budget exhausted. Episode terminated.",
                done=True,
                reward=0.0,
                current_view=self._workspace.get_current_content(),
            )

        return self._build_observation(
            message=result.message,
            done=False,
            reward=delta,
            current_view=self._workspace.get_current_content(),
        )

    @property
    def state(self) -> APState:
        return self._state

    def _build_observation(
        self,
        message: str,
        done: bool,
        reward: float,
        current_view: Optional[str] = None,
    ) -> APObservation:
        fixture = self._fixture
        visible_docs = list(fixture.documents.keys()) if fixture else []
        steps_remaining = (
            max(0, fixture.max_steps - self._state.step_count) if fixture else 0
        )
        return APObservation(
            case_id=self._state.case_id,
            task_id=self._state.task_id,
            visible_documents=visible_docs,
            current_view=current_view,
            extracted_facts=dict(self._state.extracted_facts),
            pending_issues=list(self._state.notes),
            action_history=[],
            steps_remaining=steps_remaining,
            message=message,
            done=done,
            reward=reward,
        )
