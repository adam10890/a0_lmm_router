"""WorkflowRegistry — v1.7 reimplementation for a0_lmm_router plugin.

The v0.9.7 version was never completed. This is a minimal working implementation
that loads workflow definitions from conf/routing_config.yaml and routes requests.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from helpers import files, yaml as yaml_helper

logger = logging.getLogger(__name__)

ROUTING_CONFIG_PATH = "conf/routing_config.yaml"


@dataclass
class WorkflowDefinition:
    id: str
    name: str
    patterns: list[str] = field(default_factory=list)
    description: str = ""
    enabled: bool = True
    _compiled: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def matches(self, text: str) -> bool:
        if not self.enabled:
            return False
        return any(rx.search(text) for rx in self._compiled)


class WorkflowRegistry:
    """Registry of available workflows and routing logic."""

    def __init__(self):
        self._workflows: dict[str, WorkflowDefinition] = {}
        self._load()

    def _load(self):
        try:
            path = files.get_abs_path(ROUTING_CONFIG_PATH)
            if files.exists(path):
                data = yaml_helper.loads(files.read_file(path)) or {}
                for wf in data.get("workflows", []):
                    wf_id = wf.get("id", "")
                    if wf_id:
                        self._workflows[wf_id] = WorkflowDefinition(
                            id=wf_id,
                            name=wf.get("name", wf_id),
                            patterns=wf.get("patterns", []),
                            description=wf.get("description", ""),
                            enabled=wf.get("enabled", True),
                        )
                logger.info(f"WorkflowRegistry: loaded {len(self._workflows)} workflows")
            else:
                logger.debug("WorkflowRegistry: no routing_config.yaml found, running with empty registry")
        except Exception as e:
            logger.warning(f"WorkflowRegistry: failed to load config: {e}")

    def route_request(self, text: str) -> Optional[WorkflowDefinition]:
        """Return the first matching workflow or None."""
        for wf in self._workflows.values():
            if wf.matches(text):
                logger.debug(f"WorkflowRegistry: routed to '{wf.id}'")
                return wf
        return None

    def get(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        return self._workflows.get(workflow_id)

    def all(self) -> list[WorkflowDefinition]:
        return list(self._workflows.values())

    def reload(self):
        self._workflows.clear()
        self._load()


_registry: Optional[WorkflowRegistry] = None


def get_workflow_registry() -> WorkflowRegistry:
    global _registry
    if _registry is None:
        _registry = WorkflowRegistry()
    return _registry
