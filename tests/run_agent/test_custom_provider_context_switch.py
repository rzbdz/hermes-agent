"""Tests that custom_providers per-model context_length is correctly resolved.

Covers:
- __init__ stores custom_providers per-model context_length in _config_context_length
- switch_model re-resolves context_length from custom_providers for the new model
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


_CUSTOM_PROVIDERS = [
    {
        "name": "smt-claude",
        "base_url": "https://api.example.com:60000/",
        "api_key": "sk-test-claude",
        "api_mode": "anthropic_messages",
        "model": "claude-opus-4.6",
        "models": {
            "claude-opus-4.6": {"context_length": 1000000},
        },
    },
    {
        "name": "smt-codex",
        "base_url": "https://api.example.com:60000/v1",
        "api_key": "sk-test-codex",
        "api_mode": "chat_completions",
        "model": "gemini-3.1-pro-preview",
        "models": {
            "gemini-3.1-pro-preview": {"context_length": 1000000},
            "gpt-5.4": {"context_length": 1000000},
        },
    },
]


def _build_init_agent(model, base_url, custom_providers=None):
    """Build an AIAgent via __init__ with custom_providers config."""
    cfg = {
        "model": {"default": model, "provider": "custom", "base_url": base_url},
    }
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=128_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model=model,
            api_key="sk-test",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


def test_init_stores_custom_provider_context_length():
    """__init__ should store per-model context_length from custom_providers."""
    agent = _build_init_agent(
        model="gemini-3.1-pro-preview",
        base_url="https://api.example.com:60000/v1",
        custom_providers=_CUSTOM_PROVIDERS,
    )
    assert agent._config_context_length == 1000000


def test_init_no_custom_provider_match():
    """When no custom_provider matches, _config_context_length stays None."""
    agent = _build_init_agent(
        model="some-unknown-model",
        base_url="https://other-api.example.com/v1",
        custom_providers=_CUSTOM_PROVIDERS,
    )
    assert agent._config_context_length is None


# ── switch_model tests ──


def _make_agent_for_switch(initial_model, initial_base_url, config_context_length=None):
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)
    agent.model = initial_model
    agent.provider = "custom"
    agent.base_url = initial_base_url
    agent.api_key = "sk-test"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._config_context_length = config_context_length
    agent._primary_runtime = {}

    compressor = ContextCompressor(
        model=initial_model,
        threshold_percent=0.50,
        base_url=initial_base_url,
        api_key="sk-test",
        provider="custom",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor
    return agent


def test_switch_model_resolves_new_custom_provider_context():
    """switch_model should re-resolve context_length from custom_providers for the new model."""
    agent = _make_agent_for_switch(
        initial_model="claude-opus-4.6",
        initial_base_url="https://api.example.com:60000/",
        config_context_length=1000000,
    )

    cfg = {"custom_providers": _CUSTOM_PROVIDERS}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=1000000) as mock_ctx,
    ):
        agent.switch_model(
            "gemini-3.1-pro-preview",
            "custom",
            api_key="sk-test-codex",
            base_url="https://api.example.com:60000/v1",
        )

    # _config_context_length should be updated to the new model's value
    assert agent._config_context_length == 1000000
    # get_model_context_length should have been called with the new config override
    mock_ctx.assert_called_once()
    assert mock_ctx.call_args.kwargs["config_context_length"] == 1000000


def test_switch_model_clears_context_when_no_match():
    """switch_model to a model not in custom_providers should fall back to None."""
    agent = _make_agent_for_switch(
        initial_model="gemini-3.1-pro-preview",
        initial_base_url="https://api.example.com:60000/v1",
        config_context_length=1000000,
    )

    cfg = {"custom_providers": _CUSTOM_PROVIDERS}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=128_000) as mock_ctx,
    ):
        agent.switch_model(
            "unknown-model",
            "custom",
            api_key="sk-test",
            base_url="https://other-api.example.com/v1",
        )

    # No custom_provider match — should be None (the initial _config_context_length
    # of 1000000 should NOT persist for a different model)
    assert agent._config_context_length is None
    mock_ctx.assert_called_once()
    assert mock_ctx.call_args.kwargs["config_context_length"] is None
