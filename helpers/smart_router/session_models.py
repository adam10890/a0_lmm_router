"""Session and workflow data models for a0_lmm_router.

v1.7 replacement for the v0.9.7 helpers/session_models.py that was never completed.
Minimal implementation to unblock _20_smart_router.py.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class WorkflowStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    id: str
    name: str
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    result: Any = None
    error: Optional[str] = None


@dataclass
class Session:
    id: str
    workflow_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    steps: list[WorkflowStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING

    def is_complete(self) -> bool:
        return self.status in (WorkflowStepStatus.DONE, WorkflowStepStatus.FAILED)
