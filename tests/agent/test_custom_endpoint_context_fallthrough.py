"""Tests that custom endpoints without context_length fall through to downstream lookups.

When a custom endpoint's /models response omits context_length, the resolver
should continue to models.dev / OpenRouter / hardcoded defaults instead of
returning the 128K fallback immediately.
"""

from unittest.mock import patch, MagicMock

from agent.model_metadata import get_model_context_length


def test_custom_endpoint_no_context_falls_through_to_models_dev():
    """Custom endpoint without context_length should fall through to models.dev."""
    # Simulate /models returning a model entry without context_length
    endpoint_metadata = {
        "gemini-3.1-pro-preview": {"name": "gemini-3.1-pro-preview"},
    }

    with (
        patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value=endpoint_metadata,
        ),
        patch("agent.model_metadata._is_custom_endpoint", return_value=True),
        patch("agent.model_metadata._is_known_provider_base_url", return_value=False),
        patch("agent.model_metadata.is_local_endpoint", return_value=False),
        # models.dev should be consulted and return the correct value
        patch(
            "agent.models_dev.lookup_models_dev_context",
            return_value=1048576,
        ) as mock_models_dev,
    ):
        result = get_model_context_length(
            "gemini-3.1-pro-preview",
            base_url="https://proxy.example.com/v1",
            api_key="sk-test",
            provider="custom",
        )

    # Should have fallen through to models.dev instead of returning 128K
    assert result == 1048576
    mock_models_dev.assert_called()


def test_custom_endpoint_with_context_returns_immediately():
    """Custom endpoint that reports context_length should return it directly."""
    endpoint_metadata = {
        "my-model": {"name": "my-model", "context_length": 32768},
    }

    with (
        patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value=endpoint_metadata,
        ),
        patch("agent.model_metadata._is_custom_endpoint", return_value=True),
        patch("agent.model_metadata._is_known_provider_base_url", return_value=False),
        patch("agent.model_metadata.is_local_endpoint", return_value=False),
    ):
        result = get_model_context_length(
            "my-model",
            base_url="https://proxy.example.com/v1",
            api_key="sk-test",
            provider="custom",
        )

    assert result == 32768


def test_custom_endpoint_local_server_still_queried():
    """Local custom endpoints should still try _query_local_context_length."""
    endpoint_metadata = {
        "my-model": {"name": "my-model"},  # no context_length
    }

    with (
        patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value=endpoint_metadata,
        ),
        patch("agent.model_metadata._is_custom_endpoint", return_value=True),
        patch("agent.model_metadata._is_known_provider_base_url", return_value=False),
        patch("agent.model_metadata.is_local_endpoint", return_value=True),
        patch(
            "agent.model_metadata._query_local_context_length",
            return_value=65536,
        ) as mock_local,
        patch("agent.model_metadata.save_context_length"),
    ):
        result = get_model_context_length(
            "my-model",
            base_url="http://localhost:8080/v1",
            api_key="",
            provider="custom",
        )

    assert result == 65536
    mock_local.assert_called_once()
