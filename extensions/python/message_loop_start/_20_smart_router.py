"""Smart Router Extension — currently DISABLED.

NOTE: Workflow routing is not currently used. The key 'smart_router_workflow'
was never read by any other component. This file retained only for potential
future reactivation; will be deleted in a future release if not reactivated.
"""
from helpers.extension import Extension


class SmartRouterExtension(Extension):
    """Smart Router: currently disabled."""

    async def execute(self, loop_data=None, **kwargs):
        # DISABLED — no-op to avoid computational overhead per message.
        return
