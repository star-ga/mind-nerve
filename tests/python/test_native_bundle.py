"""U4b — the native binary-catalog PRODUCER.

Covers:
  * ``tools/quantize_encoder_to_q16.py``'s MNW1 placeholder builder + CLI flag.
  * ``mind_nerve.native_bundle``'s seed-wiring glue (route_id/external_id
    derivation, bundle writing, coexistence with the Python-path files).
  * ``mind_nerve.inference._seed_from_hf`` calling the native-bundle glue.
  * The end-to-end proof: a bundle written by this producer is ACCEPTED by
    the real native loader (``src/loader.mind``) and native ``tools/call``
    (``src/mcp.mind``) serves routes from it — built + run via ``mindc``
    when the toolchain is available on PATH, skipped cleanly otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "catalog-builder"))
from format.cat_v2 import EMBEDDING_DIM, decode_mnc1  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "python"))
from mind_nerve import native_bundle  # noqa: E402

QUANTIZE_TOOL = REPO_ROOT / "tools" / "quantize_encoder_to_q16.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qmod() -> Any:
    return _load_module(QUANTIZE_TOOL, "quantize_encoder_to_q16_mnw1_tests")


# ---------------------------------------------------------------------------
# MNW1 placeholder builder
# ---------------------------------------------------------------------------


class TestMnw1PlaceholderBundle:
    def test_magic_version_reserved(self, qmod: Any) -> None:
        blob = qmod.build_mnw1_placeholder_bundle()
        assert blob[:4] == b"MNW1"
        assert int.from_bytes(blob[4:6], "little") == 1
        assert int.from_bytes(blob[6:8], "little") == 0

    def test_shape_matches_native_loader(self, qmod: Any) -> None:
        blob = qmod.build_mnw1_placeholder_bundle()
        layers = int.from_bytes(blob[72:76], "little")
        hidden = int.from_bytes(blob[76:80], "little")
        assert layers == qmod.NATIVE_ENCODER_LAYERS == 2
        assert hidden == qmod.NATIVE_ENCODER_HIDDEN == 256

    def test_total_size(self, qmod: Any) -> None:
        blob = qmod.build_mnw1_placeholder_bundle()
        h = qmod.NATIVE_ENCODER_HIDDEN
        layers = qmod.NATIVE_ENCODER_LAYERS
        per_layer = 4 * h * h + 4 * (h * 4) + 3 * h * 4
        expected = 80 + layers * per_layer + 2 * h * 4 + qmod.NATIVE_VOCAB_SIZE * h * 4
        assert len(blob) == expected

    def test_deterministic(self, qmod: Any) -> None:
        a = qmod.build_mnw1_placeholder_bundle()
        b = qmod.build_mnw1_placeholder_bundle()
        assert a == b

    def test_different_labels_change_hash_fields_only(self, qmod: Any) -> None:
        a = qmod.build_mnw1_placeholder_bundle(model_hash_label="x")
        b = qmod.build_mnw1_placeholder_bundle(model_hash_label="y")
        assert a[8:40] != b[8:40]
        assert a[80:] == b[80:]  # everything after the header is unaffected

    def test_cli_emit_placeholder(self, tmp_path: Path) -> None:
        out = tmp_path / "runtime"
        env = dict(os.environ)
        env.pop("MIND_NERVE_RUNTIME_DIR", None)
        result = subprocess.run(
            [
                sys.executable,
                str(QUANTIZE_TOOL),
                "--emit-mnw1-placeholder",
                "--output",
                str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        weights_path = out / "encoder_weights.mnw"
        assert weights_path.is_file()
        assert payload["byte_size"] == weights_path.stat().st_size
        assert (
            payload["sha256"] == __import__("hashlib").sha256(weights_path.read_bytes()).hexdigest()
        )

    def test_cli_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "runtime"
        result = subprocess.run(
            [
                sys.executable,
                str(QUANTIZE_TOOL),
                "--emit-mnw1-placeholder",
                "--dry-run",
                "--output",
                str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert not (out / "encoder_weights.mnw").exists()

    def test_cli_requires_checkpoint_or_placeholder_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, str(QUANTIZE_TOOL)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "checkpoint" in result.stderr


# ---------------------------------------------------------------------------
# native_bundle — route id / external id derivation
# ---------------------------------------------------------------------------


class TestRouteIdDerivation:
    def test_prefers_sha256_field(self) -> None:
        meta = [{"sha256": "abc123", "id": "should-not-use-this"}]
        route_ids, external_ids = native_bundle.route_ids_and_external_ids(meta)
        assert external_ids[0] == b"abc123"

    def test_falls_back_to_id_field(self) -> None:
        meta = [{"id": "r0"}]
        route_ids, external_ids = native_bundle.route_ids_and_external_ids(meta)
        assert external_ids[0] == b"r0"

    def test_falls_back_to_synthetic_when_neither_present(self) -> None:
        meta = [{"name": "no identifiers here"}]
        route_ids, external_ids = native_bundle.route_ids_and_external_ids(meta)
        assert external_ids[0] == b"route_000000"

    def test_route_id_is_sha256_of_external_id(self) -> None:
        import hashlib

        meta = [{"sha256": "deadbeef"}]
        route_ids, external_ids = native_bundle.route_ids_and_external_ids(meta)
        assert route_ids[0] == hashlib.sha256(b"deadbeef").digest()

    def test_deterministic_across_calls(self) -> None:
        meta = [{"sha256": "abc"}, {"id": "r1"}, {}]
        a = native_bundle.route_ids_and_external_ids(meta)
        b = native_bundle.route_ids_and_external_ids(meta)
        assert a == b


# ---------------------------------------------------------------------------
# native_bundle — MNC1 catalog construction
# ---------------------------------------------------------------------------


class TestBuildRouteTableCat:
    def test_round_trips_through_decode_mnc1(self) -> None:
        meta = [{"sha256": f"r{i:04x}", "id": f"id{i}"} for i in range(5)]
        blob = native_bundle.build_route_table_cat_bytes(meta)
        result = decode_mnc1(blob)
        assert result["route_count"] == 5
        for route in result["routes"]:
            assert route["embedding"] == [0] * EMBEDDING_DIM

    def test_empty_meta_produces_empty_catalog(self) -> None:
        blob = native_bundle.build_route_table_cat_bytes([])
        result = decode_mnc1(blob)
        assert result["route_count"] == 0


# ---------------------------------------------------------------------------
# native_bundle — write_native_bundle / write_native_bundle_from_route_table
# ---------------------------------------------------------------------------


class TestWriteNativeBundle:
    def _sample_meta(self, n: int) -> list[dict[str, Any]]:
        return [
            {"sha256": f"cafef00d{i:04x}", "id": f"id{i}", "name": f"route{i}"} for i in range(n)
        ]

    def test_writes_both_files(self, tmp_path: Path) -> None:
        result = native_bundle.write_native_bundle(tmp_path, self._sample_meta(3))
        cat_path = tmp_path / "route_table.cat"
        weights_path = tmp_path / "encoder_weights.mnw"
        assert cat_path.is_file()
        assert weights_path.is_file()
        assert result["route_count"] == 3
        assert result["route_table_cat_bytes"] == cat_path.stat().st_size
        assert result["encoder_weights_mnw_bytes"] == weights_path.stat().st_size

    def test_written_catalog_decodes(self, tmp_path: Path) -> None:
        native_bundle.write_native_bundle(tmp_path, self._sample_meta(4))
        blob = (tmp_path / "route_table.cat").read_bytes()
        result = decode_mnc1(blob)
        assert result["route_count"] == 4

    def test_coexists_with_python_path_files(self, tmp_path: Path) -> None:
        """Writing the native bundle must never touch route_table.jsonl/.npy
        or encoder_weights.q16.bin — both wire formats coexist (no flag-day)."""
        jsonl = tmp_path / "route_table.jsonl"
        jsonl.write_text('{"id":"r0"}\n', encoding="utf-8")
        q16_bin = tmp_path / "encoder_weights.q16.bin"
        q16_bin.write_bytes(b"\x00" * 16)

        native_bundle.write_native_bundle(tmp_path, self._sample_meta(1))

        assert jsonl.read_text(encoding="utf-8") == '{"id":"r0"}\n'
        assert q16_bin.read_bytes() == b"\x00" * 16
        assert (tmp_path / "route_table.cat").is_file()
        assert (tmp_path / "encoder_weights.mnw").is_file()

    def test_from_route_table_reads_existing_jsonl(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "route_table.jsonl"
        rows = [{"sha256": f"r{i:04x}", "id": f"id{i}"} for i in range(3)]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = native_bundle.write_native_bundle_from_route_table(tmp_path)
        assert result is not None
        assert result["route_count"] == 3

    def test_from_route_table_none_when_jsonl_absent(self, tmp_path: Path) -> None:
        assert native_bundle.write_native_bundle_from_route_table(tmp_path) is None

    def test_from_route_table_skips_blank_lines(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "route_table.jsonl"
        jsonl.write_text('{"sha256":"a"}\n\n{"sha256":"b"}\n', encoding="utf-8")
        result = native_bundle.write_native_bundle_from_route_table(tmp_path)
        assert result is not None
        assert result["route_count"] == 2


# ---------------------------------------------------------------------------
# _seed_from_hf wiring (inference.py) — mocked download, no network
# ---------------------------------------------------------------------------


class TestSeedFromHfWiring:
    def test_seed_from_hf_does_not_auto_write_native_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Native tools/call is fail-closed by default: ``_seed_from_hf`` seeds
        the Python routing path but must NOT auto-write the native .cat/.mnw
        bundle. The only bundle producible today uses placeholder zero
        embeddings (no 256-dim checkpoint), which would return misleading
        zero-ranked routes — the placeholder producer is reachable only via the
        explicit ``mind-nerve seed-native-bundle`` dev command.
        """
        import mind_nerve.inference as inf_mod

        cached = tmp_path / "hf_cache"
        cached.mkdir()
        rows = [{"sha256": f"r{i:04x}", "id": f"id{i}"} for i in range(2)]
        (cached / "route_table.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        (cached / "manifest.json").write_text("{}", encoding="utf-8")

        def _fake_snapshot_download(**kwargs: Any) -> str:
            return str(cached)

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

        target = tmp_path / "seeded_runtime"
        inf_mod._seed_from_hf(target)

        # Python routing path is seeded ...
        assert (target / "route_table.jsonl").is_file()
        # ... but the native bundle is NOT auto-seeded (fail-closed by default).
        assert not (target / "route_table.cat").exists()
        assert not (target / "encoder_weights.mnw").exists()

    def test_explicit_dev_command_writes_placeholder_bundle(self, tmp_path: Path) -> None:
        """The explicit producer still works (dev/plumbing path): given a
        route_table.jsonl, it writes a loader-shaped placeholder bundle.
        """
        rows = [{"sha256": f"r{i:04x}", "id": f"id{i}"} for i in range(2)]
        (tmp_path / "route_table.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        result = native_bundle.write_native_bundle_from_route_table(tmp_path)

        assert result is not None
        assert (tmp_path / "route_table.cat").is_file()
        assert (tmp_path / "encoder_weights.mnw").is_file()
        assert decode_mnc1((tmp_path / "route_table.cat").read_bytes())["route_count"] == 2


# ---------------------------------------------------------------------------
# End-to-end proof: the native loader ACCEPTS a producer-written bundle and
# native tools/call serves routes from it. Requires mindc + the LLVM-20
# toolchain on PATH; skips cleanly when unavailable (never a false pass).
# ---------------------------------------------------------------------------


def _native_toolchain_available() -> bool:
    return shutil.which("mindc") is not None and Path("/usr/lib/llvm-20/bin").is_dir()


@pytest.mark.skipif(
    not _native_toolchain_available(), reason="mindc / llvm-20 toolchain not on PATH"
)
class TestNativeLoaderRoundTrip:
    @pytest.fixture(scope="class")
    def native_binary(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        out_dir = tmp_path_factory.mktemp("mind-nerve-bin")
        binary = out_dir / "mn"
        env = dict(os.environ)
        env["PATH"] = f"/usr/lib/llvm-20/bin:{env.get('PATH', '')}"
        result = subprocess.run(
            ["mindc", "build", "--target", "cpu", "--out", str(binary)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"mindc build failed:\n{result.stdout}\n{result.stderr}"
        assert binary.is_file(), "mindc build reported success but produced no binary"
        return binary

    def _tools_call(
        self, binary: Path, runtime_dir: Path | None, query: str, top_k: int = 3
    ) -> dict:
        env = dict(os.environ)
        if runtime_dir is not None:
            env["MIND_NERVE_RUNTIME_DIR"] = str(runtime_dir)
        else:
            env.pop("MIND_NERVE_RUNTIME_DIR", None)
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "mind_nerve_route",
                    "arguments": {"query": query, "top_k": top_k},
                },
            }
        )
        result = subprocess.run(
            [str(binary)],
            input=msg + "\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        line = result.stdout.strip().splitlines()[-1]
        envelope = json.loads(line)
        return json.loads(envelope["result"]["content"][0]["text"])

    def test_unseeded_dir_fails_closed(self, native_binary: Path, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_runtime"
        empty_dir.mkdir()
        payload = self._tools_call(native_binary, empty_dir, "hello")
        assert payload["served_by"] == "unavailable"
        assert payload["routes"] == []

    def test_seeded_dir_routes(self, native_binary: Path, tmp_path: Path) -> None:
        rt_dir = tmp_path / "seeded_runtime"
        meta = [{"sha256": f"cafef00d{i:04x}", "id": f"id{i}"} for i in range(5)]
        native_bundle.write_native_bundle(rt_dir, meta)

        payload = self._tools_call(native_binary, rt_dir, "deploy the app", top_k=3)

        assert payload["served_by"] == "in-process (native)"
        assert payload["catalog_size"] == 5
        assert len(payload["routes"]) == 3
        for route in payload["routes"]:
            assert len(route["id"]) == 64  # 32-byte SHA-256, hex-encoded
            int(route["id"], 16)  # valid hex

    def test_seeded_routes_are_the_expected_route_ids(
        self, native_binary: Path, tmp_path: Path
    ) -> None:
        import hashlib

        rt_dir = tmp_path / "seeded_runtime_2"
        meta = [{"sha256": f"route-alpha-{i}"} for i in range(3)]
        native_bundle.write_native_bundle(rt_dir, meta)
        expected_ids = {hashlib.sha256(f"route-alpha-{i}".encode()).hexdigest() for i in range(3)}

        payload = self._tools_call(native_binary, rt_dir, "anything", top_k=3)

        returned_ids = {route["id"] for route in payload["routes"]}
        assert returned_ids <= expected_ids
        assert len(returned_ids) == 3
