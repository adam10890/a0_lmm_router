"""Rate Limit Retry Helper — standalone retry logic for litellm calls.

This module provides retry logic for RateLimitError that can be applied
to litellm's acompletion function or Model.unified_call.

Usage:
    # Option 1: Patch at startup (in an extension)
    from usr.plugins.a0_lmm_router.helpers.rate_limit_retry import patch_litellm
    patch_litellm()

    # Option 2: Use as decorator on specific calls
    from usr.plugins.a0_lmm_router.helpers.rate_limit_retry import with_retry

    @with_retry(max_retries=5)
    async def my_llm_call():
        return await acompletion(...)
"""
from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0


def is_rate_limit_error(exc: Exception) -> bool:
    """
    Check if an exception is a rate limit error.

    Detects:
    - litellm.exceptions.RateLimitError
    - HTTP 429 status codes
    - OpenRouter-specific rate limit messages
    - Any exception with 'rate limit' in the message
    """
    if exc is None:
        return False

    exc_class = exc.__class__.__name__
    exc_module = getattr(exc.__class__, "__module__", "")
    exc_str = str(exc).lower()

    # litellm RateLimitError
    if exc_class == "RateLimitError" and "litellm" in exc_module:
        return True

    # HTTP 429
    if "429" in str(exc):
        return True

    # Rate limit keywords
    rate_limit_keywords = [
        "rate limit",
        "too many requests",
        "temporarily rate-limited",
        "ratelimit",
        "throttled",
    ]
    if any(keyword in exc_str for keyword in rate_limit_keywords):
        return True

    # Provider-specific patterns
    if "provider returned error" in exc_str and "429" in str(exc):
        return True

    return False


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    retryable_exceptions: Optional[tuple[type[Exception], ...]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds exponential backoff retry for rate limit errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: Tuple of exception types to retry on.
                           Defaults to auto-detecting rate limit errors.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if this exception should trigger a retry
                    should_retry = False
                    if retryable_exceptions:
                        should_retry = isinstance(e, retryable_exceptions)
                    else:
                        should_retry = is_rate_limit_error(e)

                    if not should_retry:
                        raise  # Not retryable, propagate immediately

                    last_exception = e

                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        # Add jitter (+-10%)
                        jitter = delay * 0.1 * (2 * random.random() - 1)
                        total_delay = delay + jitter

                        logger.warning(
                            f"Rate limit hit on attempt {attempt + 1}/{max_retries} "
                            f"in {func.__name__}: {e}. "
                            f"Retrying in {total_delay:.1f}s..."
                        )
                        await asyncio.sleep(total_delay)
                    else:
                        logger.error(
                            f"Rate limit persisted after {max_retries} attempts "
                            f"in {func.__name__}: {e}"
                        )

            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected: no exception captured after retries")

        return async_wrapper  # type: ignore[return-value]

    return decorator


def _record_litellm_usage(kwargs: dict, response: Any) -> None:
    """Best-effort per-provider accounting for a completed litellm call.

    Maps the call to a provider via api_base first, model prefix second
    (budget_engine.match_provider); unmatched calls are not recorded.
    """
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return

    def _get(obj: Any, key: str) -> int:
        val = getattr(obj, key, None)
        if val is None and isinstance(obj, dict):
            val = obj.get(key)
        return int(val or 0)

    tokens_in = _get(usage, "prompt_tokens")
    tokens_out = _get(usage, "completion_tokens")
    if not tokens_in and not tokens_out:
        return

    try:
        from . import budget_engine, usage_ledger
    except ImportError:
        from helpers import budget_engine, usage_ledger

    provider_id = budget_engine.match_provider(
        str(kwargs.get("model", "")), kwargs.get("api_base")
    )
    if not provider_id:
        return
    usage_ledger.record_usage(
        provider_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        model=str(kwargs.get("model", "")),
        source="litellm",
    )


def patch_acompletion() -> bool:
    """
    Monkey-patch litellm.acompletion to add retry logic.

    Returns:
        True if patching succeeded, False otherwise.
    """
    try:
        import litellm

        # Check if already patched
        if hasattr(litellm, "_original_acompletion"):
            logger.debug("litellm.acompletion already patched, skipping")
            return True

        original = litellm.acompletion
        litellm._original_acompletion = original

        @functools.wraps(original)
        async def retrying_acompletion(*args: Any, **kwargs: Any) -> Any:
            retry_wrapper = with_retry()
            result = await retry_wrapper(original)(*args, **kwargs)
            try:  # ponytail: best-effort accounting, never break the call path
                _record_litellm_usage(kwargs, result)
            except Exception:
                pass
            return result

        litellm.acompletion = retrying_acompletion
        logger.info("Patched litellm.acompletion with rate limit retry logic")
        return True

    except ImportError:
        logger.debug("litellm not available for patching")
        return False
    except Exception as e:
        logger.warning(f"Failed to patch litellm.acompletion: {e}")
        return False


def patch_model_unified_call() -> bool:
    """
    Monkey-patch models.Model.unified_call to add retry logic.

    This is the method that calls litellm.acompletion in Agent Zero.

    Returns:
        True if patching succeeded, False otherwise.
    """
    try:
        from models import Model

        # Check if already patched
        if hasattr(Model, "_original_unified_call"):
            logger.debug("Model.unified_call already patched, skipping")
            return True

        original = Model.unified_call
        Model._original_unified_call = original

        # NOTE: no usage recording here — unified_call goes through
        # litellm.acompletion (also patched); recording in both layers
        # would double-count every agent call.
        @functools.wraps(original)
        async def retrying_unified_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            retry_wrapper = with_retry()
            return await retry_wrapper(original)(self, *args, **kwargs)

        Model.unified_call = retrying_unified_call  # type: ignore[method-assign]
        logger.info("Patched Model.unified_call with rate limit retry logic")
        return True

    except ImportError:
        logger.debug("models.Model not available for patching (expected outside container)")
        return False
    except Exception as e:
        logger.warning(f"Failed to patch Model.unified_call: {e}")
        return False


def patch_litellm() -> bool:
    """
    Apply all available patches for rate limit retry logic.

    This patches both litellm.acompletion and models.Model.unified_call
    if they are available.

    Returns:
        True if at least one patch succeeded, False otherwise.
    """
    results = [
        patch_acompletion(),
        patch_model_unified_call(),
    ]
    return any(results)
