"""Contract tests for the skill marketplace stub interfaces.

These tests validate:
  - The typed signatures of all public functions and dataclasses.
  - The schema described in spec/skill_marketplace.md.
  - That stub functions raise NotImplementedError with the correct message.

Tests do NOT suppress NotImplementedError or mock around it. The contract is
that stubs raise — any test that passes by catching that exception is
explicitly checking the stub is in place, not bypassing the requirement.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest
from mind_nerve.marketplace import (
    ProviderDelta,
    SkillProvider,
    apply_delta,
    list_providers,
    query_route_delta,
    register_skill_provider,
)

# ---------------------------------------------------------------------------
# SkillProvider dataclass contract
# ---------------------------------------------------------------------------


class TestSkillProviderContract:
    def test_has_required_fields(self) -> None:
        fields = {f.name for f in SkillProvider.__dataclass_fields__.values()}
        assert "name" in fields
        assert "url" in fields
        assert "signature" in fields
        assert "license" in fields

    def test_name_field_is_str(self) -> None:
        hints = get_type_hints(SkillProvider)
        assert hints["name"] is str

    def test_url_field_is_str(self) -> None:
        hints = get_type_hints(SkillProvider)
        assert hints["url"] is str

    def test_signature_field_is_bytes(self) -> None:
        hints = get_type_hints(SkillProvider)
        assert hints["signature"] is bytes

    def test_license_field_is_str(self) -> None:
        hints = get_type_hints(SkillProvider)
        assert hints["license"] is str

    def test_is_frozen_dataclass(self) -> None:
        provider = SkillProvider(
            name="test",
            url="https://example.com",
            signature=b"\x00" * 64,
            license="apache-2.0",
        )
        with pytest.raises((AttributeError, TypeError)):
            provider.name = "mutated"  # type: ignore[misc]

    def test_instantiation_with_valid_values(self) -> None:
        provider = SkillProvider(
            name="example-skills",
            url="https://example.com/skills",
            signature=bytes(64),  # 64 zero bytes, ed25519 signature size
            license="apache-2.0",
        )
        assert provider.name == "example-skills"
        assert provider.url == "https://example.com/skills"
        assert len(provider.signature) == 64
        assert provider.license == "apache-2.0"


# ---------------------------------------------------------------------------
# ProviderDelta dataclass contract
# ---------------------------------------------------------------------------


class TestProviderDeltaContract:
    def test_has_required_fields(self) -> None:
        fields = {f.name for f in ProviderDelta.__dataclass_fields__.values()}
        assert "provider" in fields
        assert "added_count" in fields
        assert "removed_count" in fields
        assert "as_of_version" in fields

    def test_is_frozen_dataclass(self) -> None:
        provider = SkillProvider(
            name="p", url="https://example.com", signature=bytes(64), license="mit"
        )
        delta = ProviderDelta(
            provider=provider, added_count=5, removed_count=1, as_of_version="1.2.0"
        )
        with pytest.raises((AttributeError, TypeError)):
            delta.added_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# register_skill_provider signature contract
# ---------------------------------------------------------------------------


class TestRegisterSkillProviderSignature:
    def test_accepts_url_parameter(self) -> None:
        sig = inspect.signature(register_skill_provider)
        assert "url" in sig.parameters

    def test_url_parameter_is_str_annotated(self) -> None:
        hints = get_type_hints(register_skill_provider)
        assert hints.get("url") is str

    def test_return_annotation_is_skill_provider(self) -> None:
        hints = get_type_hints(register_skill_provider)
        assert hints.get("return") is SkillProvider

    def test_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            register_skill_provider("https://example.com")
        assert "Phase 2" in str(exc_info.value)

    def test_stub_message_references_phase2(self) -> None:
        try:
            register_skill_provider("https://example.com")
        except NotImplementedError as err:
            assert "Phase 2" in str(err)


# ---------------------------------------------------------------------------
# list_providers signature contract
# ---------------------------------------------------------------------------


class TestListProvidersSignature:
    def test_takes_no_required_parameters(self) -> None:
        sig = inspect.signature(list_providers)
        required = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
        assert len(required) == 0

    def test_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            list_providers()
        assert "Phase 2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# query_route_delta signature contract
# ---------------------------------------------------------------------------


class TestQueryRouteDeltaSignature:
    def test_accepts_provider_parameter(self) -> None:
        sig = inspect.signature(query_route_delta)
        assert "provider" in sig.parameters

    def test_provider_annotation_is_skill_provider(self) -> None:
        hints = get_type_hints(query_route_delta)
        assert hints.get("provider") is SkillProvider

    def test_return_annotation_is_bytes(self) -> None:
        hints = get_type_hints(query_route_delta)
        assert hints.get("return") is bytes

    def test_raises_not_implemented(self) -> None:
        provider = SkillProvider(
            name="p", url="https://example.com", signature=bytes(64), license="mit"
        )
        with pytest.raises(NotImplementedError) as exc_info:
            query_route_delta(provider)
        assert "Phase 2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# apply_delta signature contract
# ---------------------------------------------------------------------------


class TestApplyDeltaSignature:
    def test_accepts_provider_and_delta_bytes(self) -> None:
        sig = inspect.signature(apply_delta)
        assert "provider" in sig.parameters
        assert "delta_bytes" in sig.parameters

    def test_return_annotation_is_provider_delta(self) -> None:
        hints = get_type_hints(apply_delta)
        assert hints.get("return") is ProviderDelta

    def test_raises_not_implemented(self) -> None:
        provider = SkillProvider(
            name="p", url="https://example.com", signature=bytes(64), license="mit"
        )
        with pytest.raises(NotImplementedError) as exc_info:
            apply_delta(provider, b"{}")
        assert "Phase 2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Schema alignment: spec/skill_marketplace.md field names
# ---------------------------------------------------------------------------


class TestSpecSchemaAlignment:
    """Verify that the Python types mirror the JSON-RPC schema in the spec."""

    def test_skill_provider_fields_match_describe_response_schema(self) -> None:
        # spec/skill_marketplace.md §"marketplace.describe" maps to SkillProvider
        expected_fields = {"name", "url", "signature", "license"}
        actual_fields = set(SkillProvider.__dataclass_fields__)
        assert expected_fields == actual_fields

    def test_provider_delta_fields_match_delta_response_schema(self) -> None:
        # spec/skill_marketplace.md §"marketplace.delta" result maps to ProviderDelta
        expected_fields = {"provider", "added_count", "removed_count", "as_of_version"}
        actual_fields = set(ProviderDelta.__dataclass_fields__)
        assert expected_fields == actual_fields
