"""
AP-Resolve data models.

Typed Pydantic v2 models for all data exchanged between agent and environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from openenv.core.env_server.types import Action, Observation, State

# ---------------------------------------------------------------------------
# Action type literal
# ---------------------------------------------------------------------------

ActionType = Literal[
    "open_document",
    "extract_field",
    "compare_fields",
    "calculate_total",
    "check_policy",
    "search_history",
    "request_vendor_info",
    "add_note",
    "approve_invoice",
    "hold_invoice",
    "reject_invoice",
    "escalate_case",
]

TERMINAL_ACTIONS: frozenset[str] = frozenset(
    {"approve_invoice", "hold_invoice", "reject_invoice", "escalate_case"}
)

# ---------------------------------------------------------------------------
# APAction
# ---------------------------------------------------------------------------


class APAction(Action):
    """A single agent action in the AP-Resolve environment."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    action_type: ActionType
    target: Optional[str] = None
    params: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# APObservation
# ---------------------------------------------------------------------------


class APObservation(Observation):
    """What the agent perceives after each step."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    case_id: str = ""
    task_id: str = ""
    visible_documents: List[str] = []
    current_view: Optional[str] = None
    extracted_facts: Dict[str, Any] = {}
    pending_issues: List[str] = []
    action_history: List[str] = []
    steps_remaining: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# APState
# ---------------------------------------------------------------------------


class APState(State):
    """Full internal simulator state (used for debugging and grading)."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )

    case_id: str = ""
    task_id: str = ""
    hidden_ground_truth: Dict[str, Any] = {}
    extracted_facts: Dict[str, Any] = {}
    checks_completed: Dict[str, bool] = {}
    decision: Optional[str] = None
    notes: List[str] = []
    vendor_contacted: bool = False
    reward_so_far: float = 0.0
    step_count: int = 0


# ---------------------------------------------------------------------------
# CaseFixture
# ---------------------------------------------------------------------------


@dataclass
class CaseFixture:
    """A hand-authored JSON case file with hidden ground truth."""

    task_id: str
    case_id: str
    documents: Dict[str, str]            # doc_name -> content string
    critical_fields: List[str]           # fields that must be extracted
    required_checks: List[str]           # document names that must be opened
    ground_truth_disposition: str        # correct final action_type
    required_evidence_fields: List[str]  # fields that must appear in notes/facts
    has_critical_issues: bool            # True for medium/hard cases
    soft_step_threshold: int
    max_steps: int

    # Optional vendor response template for request_vendor_info action
    vendor_response: str = ""


# ---------------------------------------------------------------------------
# ActionResult (internal)
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """Result of executing a single action."""

    success: bool
    message: str
    new_facts: Dict[str, Any] = field(default_factory=dict)
    issues_resolved: List[str] = field(default_factory=list)
    is_terminal: bool = False
    is_irrelevant: bool = False
    is_invalid: bool = False
    opened_doc: Optional[str] = None
    reopened_doc: bool = False
    extracted_critical_field: bool = False
    completed_comparison: bool = False
    identified_core_issue: bool = False


# ---------------------------------------------------------------------------
# GradeResult
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    """Structured result from the Grader."""

    composite_score: float       # weighted sum, clipped [0, 1]
    decision_score: float        # 0.0 or 1.0
    evidence_score: float        # [0, 1] fraction of required evidence present
    workflow_score: float        # [0, 1] fraction of required checks completed
    efficiency_score: float      # [0, 1] steps remaining / budget
    audit_failure: bool          # True if wrong approval on critical case
    details: Dict[str, Any] = field(default_factory=dict)
