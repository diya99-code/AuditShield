"""
APEnvironment — core OpenEnv environment for AP-Resolve.

Extends openenv.core.env_server.interfaces.Environment and wires together
TaskRegistry, DocumentWorkspace, ActionHandler, RewardCalculator, and Grader.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from .action_handler import ActionHandler
from .grader import Grader
from .models import APAction, APObservation, APState, TERMINAL_ACTIONS
from .reward import RewardCalculator
from .tasks_registry import TaskRegistry
from .workspace import DocumentWorkspace


class APEnvironment(Environment):
    """AP-Resolve OpenEnv environment."""

    SUPPORTS_CONCURRENT_SESSIONS = False

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
        # Support TASK_ID env var for default task selection
        self._default_task_id: Optional[str] = os.environ.get("TASK_ID")

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[Literal["easy_straight_through", "medium_mismatch", "hard_duplicate_partial"]] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> APObservation:
        """Reset the environment and return the initial observation."""
        # Determine which task to load
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
                "required_checks": fixture.required_checks,
                "required_issues": fixture.required_issues,
                "has_critical_issues": fixture.has_critical_issues,
            },
        )

        return self._build_observation(
            message=f"Episode started. Case: {fixture.case_id}. Task: {fixture.task_id}. "
                    f"Available documents: {fixture.documents.keys()}",
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: APAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> APObservation:
        """Execute one agent action and return the resulting observation."""
        # Post-terminal guard
        if self._done:
            return self._build_observation(
                message="Episode is already done. Call reset() to start a new episode.",
                done=True,
                reward=0.0,
            )

        assert self._fixture is not None, "Call reset() before step()"
        assert self._workspace is not None

        # Increment step count
        self._state.step_count += 1

        # Execute action
        result = self._action_handler.handle(
            action, self._workspace, self._state, self._fixture
        )

        # Compute reward delta
        delta = self._reward_calc.compute(action, result, self._state, self._fixture)
        self._state.reward_so_far = RewardCalculator.clip(
            self._state.reward_so_far + delta
        )

        # Check terminal conditions
        if result.is_terminal:
            self._done = True
            # Grader computes final composite score
            grade = self._grader.grade(self._state, self._fixture)
            final_reward = grade.composite_score
            return self._build_observation(
                message=result.message,
                done=True,
                reward=final_reward,
                current_view=self._workspace.get_current_content(),
            )

        # Step budget exhaustion
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
        """Return the full internal simulator state."""
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        message: str,
        done: bool,
        reward: float,
        current_view: Optional[str] = None,
    ) -> APObservation:
        fixture = self._fixture
        workspace = self._workspace

        visible_docs = list(fixture.documents.keys()) if fixture else []
        steps_remaining = (
            max(0, fixture.max_steps - self._state.step_count)
            if fixture
            else 0
        )

        return APObservation(
            case_id=self._state.case_id,
            task_id=self._state.task_id,
            visible_documents=visible_docs,
            current_view=current_view,
            extracted_facts=dict(self._state.extracted_facts),
            pending_issues=list(self._state.notes),
            action_history=[],  # kept lightweight; full history in state
            steps_remaining=steps_remaining,
            message=message,
            done=done,
            reward=reward,
        )
