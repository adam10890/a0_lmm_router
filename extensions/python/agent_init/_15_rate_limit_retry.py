"""Rate Limit Retry Extension — adds exponential backoff for RateLimitError.

Patches litellm.acompletion and Model.unified_call with retry logic for
handling upstream provider rate limits from OpenRouter and other providers.

Placement: extensions/python/agent_init/_15_rate_limit_retry.py
Load order: _15_ (after _10_init_servers, before message_loop_start)
"""
from __future__ import annotations

import logging

from helpers.extension import Extension

logger = logging.getLogger(__name__)


class RateLimitRetryExtension(Extension):
    """Patches LLM calls to add retry logic for rate limits."""

    def execute(self, **kwargs) -> None:
        """Apply rate limit retry patches."""
        try:
            # Import and apply patches from the helper module
            from usr.plugins.a0_lmm_router.helpers.rate_limit_retry import patch_litellm

            success = patch_litellm()
            if success:
                logger.info("RateLimitRetryExtension: applied rate limit retry patches")
            else:
                logger.debug("RateLimitRetryExtension: no patches applied (modules not available)")
        except Exception as e:
            logger.warning(f"RateLimitRetryExtension: failed to apply patches: {e}")
