"""Hermes pip-plugin entry point for the DeepSeek Context Engine."""

from __future__ import annotations

import logging

from . import (
    HERMES_CONTEXT_AVAILABLE,
    HERMES_CONTEXT_IMPORT_ERROR,
    DeepSeekContextEngine,
)

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Register the packaged ContextEngine through a real Hermes host.

    Graceful degradation (UNAVAILABLE mode): when neither DEEPSEEK_API_KEY
    nor OPENROUTER_API_KEY is configured, the plugin still loads cleanly but
    registers NO context-engine hooks — instead of registering an engine
    whose every summarization call would fail.  The degraded state is logged
    as a hint naming only the environment variables, never values.
    """
    if not HERMES_CONTEXT_AVAILABLE:
        raise RuntimeError(
            "deepseek-context requires the Hermes Agent host at registration time"
        ) from HERMES_CONTEXT_IMPORT_ERROR
    engine = DeepSeekContextEngine()
    if not engine.is_configured():
        logger.warning(
            "deepseek-context loaded but UNAVAILABLE: no provider API key set. "
            "Export DEEPSEEK_API_KEY (direct) or OPENROUTER_API_KEY (routed) "
            "and reload the plugin to enable context compression; "
            "no context-engine hooks were registered."
        )
        return
    ctx.register_context_engine(engine)
