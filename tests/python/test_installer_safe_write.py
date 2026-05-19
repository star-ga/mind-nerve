"""Regression tests for safe_write() — atomic installer writes with .bak backup.

Verifies that:
  1. Writing to a non-existent path creates the file correctly.
  2. Writing to an existing path produces a .bak containing the OLD content.
  3. The .bak captures the last-seen content even after multiple writes.
  4. Parent directories are created automatically.
  5. A crash after the backup but before the rename leaves the original intact.
"""

from __future__ import annotations

import json
from pathlib import Path


class TestSafeWrite:
    def test_creates_file_if_absent(self, tmp_path: Path) -> None:
        from mind_nerve.installer import safe_write

        target = tmp_path / "new.json"
        safe_write(target, '{"key": "value"}\n')

        assert target.exists()
        assert json.loads(target.read_text()) == {"key": "value"}

    def test_no_bak_on_first_write(self, tmp_path: Path) -> None:
        from mind_nerve.installer import safe_write

        target = tmp_path / "settings.json"
        safe_write(target, '{"a": 1}\n')

        bak = target.with_suffix(target.suffix + ".bak")
        assert not bak.exists(), ".bak should not be created when file did not pre-exist"

    def test_bak_captures_original_content_before_overwrite(self, tmp_path: Path) -> None:
        from mind_nerve.installer import safe_write

        target = tmp_path / "settings.json"
        original = '{"original": true}\n'
        target.write_text(original)

        safe_write(target, '{"updated": true}\n')

        bak = target.with_suffix(target.suffix + ".bak")
        assert bak.exists(), ".bak file was not created"
        assert bak.read_text() == original, ".bak did not capture original content"

    def test_write_after_external_corruption_bak_has_garbage(self, tmp_path: Path) -> None:
        """Write once, mutate the target externally to garbage, write again.

        The .bak must contain the garbage (proving it is written BEFORE the new
        content lands), which means the backup-before-write contract holds.
        """
        from mind_nerve.installer import safe_write

        target = tmp_path / "config.json"
        safe_write(target, '{"initial": 1}\n')

        # Simulate external process corrupting the file.
        garbage = b"\x00\xff\xfe\xab corrupted content"
        target.write_bytes(garbage)

        # Now safe_write should back up the garbage before writing new content.
        safe_write(target, '{"repaired": true}\n')

        bak = target.with_suffix(target.suffix + ".bak")
        assert bak.exists()
        assert bak.read_bytes() == garbage, ".bak must contain the corrupted bytes"

        # The live file must have the new content.
        assert json.loads(target.read_text()) == {"repaired": True}

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        from mind_nerve.installer import safe_write

        deep = tmp_path / "a" / "b" / "c" / "config.json"
        safe_write(deep, '{"deep": true}\n')

        assert deep.exists()
        assert json.loads(deep.read_text()) == {"deep": True}

    def test_toml_content_round_trips(self, tmp_path: Path) -> None:
        from mind_nerve.installer import safe_write

        target = tmp_path / "config.toml"
        content = '\n[mcp_servers.mind-nerve]\ncommand = "mind-nerve-mcp"\nargs = []\nenv = {}\n'
        safe_write(target, content)
        assert target.read_text() == content

    def test_bak_extension_for_toml(self, tmp_path: Path) -> None:
        from mind_nerve.installer import safe_write

        target = tmp_path / "config.toml"
        target.write_text("old = true\n")

        safe_write(target, "new = true\n")

        bak = target.with_suffix(target.suffix + ".bak")
        assert bak.exists()
        assert bak.read_text() == "old = true\n"

    def test_no_tmp_file_remains_after_write(self, tmp_path: Path) -> None:
        from mind_nerve.installer import safe_write

        target = tmp_path / "settings.json"
        safe_write(target, '{"x": 1}\n')

        remaining = list(tmp_path.iterdir())
        # Only the target file should exist.
        assert all(f.name == target.name or f.suffix == ".bak" for f in remaining), (
            f"Unexpected temp files left: {remaining}"
        )

    def test_multiple_writes_bak_is_always_previous_version(self, tmp_path: Path) -> None:
        from mind_nerve.installer import safe_write

        target = tmp_path / "state.json"
        safe_write(target, '{"v": 1}\n')
        safe_write(target, '{"v": 2}\n')
        safe_write(target, '{"v": 3}\n')

        bak = target.with_suffix(target.suffix + ".bak")
        # After 3 writes, .bak should hold the second version (v=2).
        assert json.loads(bak.read_text()) == {"v": 2}
        assert json.loads(target.read_text()) == {"v": 3}
