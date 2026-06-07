"""License classification for skill discovery.

Locks in the rule that a declared *public* license in the frontmatter is
authoritative and must win over commercial markers that appear in the body
as subject matter (security / forensics skills legitimately discuss words
like "proprietary", "confidential", "do not distribute").
"""

from mind_nerve.discovery import _classify, _parse_frontmatter


def _bucket(text: str) -> str:
    return _classify(text, _parse_frontmatter(text))[0]


def test_public_license_wins_over_body_marker():
    text = (
        "---\n"
        "name: analyze-c2\n"
        "license: Apache-2.0\n"
        "---\n"
        "This skill inspects proprietary C2 protocols and confidential beacons.\n"
        "Do not distribute the captured samples.\n"
    )
    assert _bucket(text) == "public_ok"


def test_mit_license_wins_over_body_marker():
    text = (
        "---\nname: bec\nlicense: MIT\n---\n"
        "Detect business email compromise; flags 'confidential' urgency language.\n"
    )
    assert _bucket(text) == "public_ok"


def test_internal_visibility_always_excluded():
    text = "---\nname: x\nlicense: MIT\nvisibility: internal\n---\nbody\n"
    assert _bucket(text) == "commercial_risk"


def test_commercial_license_excluded():
    text = "---\nname: x\nlicense: STARGA Commercial\n---\nbody\n"
    assert _bucket(text) == "commercial_risk"


def test_no_license_with_body_marker_is_commercial_risk():
    text = "---\nname: x\n---\nThis tool is STARGA Commercial License, source by agreement.\n"
    assert _bucket(text) == "commercial_risk"


def test_no_license_no_marker_is_unknown():
    text = "---\nname: x\n---\nA perfectly ordinary skill with no license field.\n"
    assert _bucket(text) == "unknown"
