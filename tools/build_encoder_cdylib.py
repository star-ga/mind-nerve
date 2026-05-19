#!/usr/bin/env python3
"""Build libmind_nerve_encoder.so from the mind/exports/c_abi.mind surface.

Strategy: compile each .mind source to MLIR individually using mindc
--emit-mlir, then merge all MLIR function bodies into a single combined
module and invoke the mlir-build pipeline (mlir-opt -> mlir-translate ->
clang) to produce a self-contained .so.

Cross-module symbols (encode, matmul_score, topk_q16) become concrete
definitions in the merged module rather than undefined external references.

Usage:
    python3 tools/build_encoder_cdylib.py [--mind-checkout /path/to/mind]
                                           [--output /path/to/libmind_nerve_encoder.so]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# Ordered list of .mind sources. Order matters: later files may reference
# symbols defined in earlier files (cross-module privates need to be defined
# before they're referenced in the merged MLIR, since MLIR is order-sensitive
# for func.func private declarations).
def _source_list(nerve_root: Path) -> list[Path]:
    return [
        nerve_root / "mind/luts/exp_q16.mind",
        nerve_root / "mind/luts/recip_q32.mind",
        nerve_root / "mind/luts/sqrt_q16.mind",
        nerve_root / "mind/luts/tanh_q16.mind",
        nerve_root / "mind/luts/softmax_q16.mind",
        nerve_root / "mind/kernels/matmul_q16.mind",
        nerve_root / "mind/kernels/matmul_blas.mind",
        nerve_root / "mind/kernels/batched_matmul_q16.mind",
        nerve_root / "mind/kernels/layernorm_q16.mind",
        nerve_root / "mind/kernels/gelu_q16.mind",
        nerve_root / "mind/kernels/l2_norm_q16.mind",
        nerve_root / "mind/kernels/embedding_q16.mind",
        nerve_root / "mind/kernels/sliding_window.mind",
        nerve_root / "mind/kernels/topk_q16.mind",
        nerve_root / "mind/kernels/encode.mind",
        nerve_root / "mind/exports/c_abi.mind",
    ]


def _mindc_path(mind_checkout: Path) -> Path:
    return mind_checkout / "target/release/mindc"


def _emit_mlir(mindc: Path, source: Path) -> str:
    """Run mindc --emit-mlir on a single source file and return the MLIR text."""
    result = subprocess.run(
        [str(mindc), str(source), "--emit-mlir"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mindc --emit-mlir failed for {source.name}:\n{result.stderr}"
        )
    return result.stdout


def _extract_functions(mlir_text: str) -> tuple[list[str], list[str]]:
    """Parse MLIR text and return (private_decls, func_defs).

    private_decls: list of `func.func private @name(...)` declaration lines
                   (external references only — no body).
    func_defs: list of complete function definition blocks (with body).
    """
    private_decls: list[str] = []
    func_defs: list[str] = []

    lines = mlir_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Private declaration (no body — just a declaration line)
        if re.match(r"\s*func\.func private @\w+\(", line) and "{" not in line:
            private_decls.append(stripped)
            i += 1
            continue

        # Function definition with a body
        if re.match(r"\s*func\.func\b", line) and "{" in line:
            # Collect the complete function block
            depth = line.count("{") - line.count("}")
            block = [line]
            i += 1
            while i < len(lines) and depth > 0:
                block.append(lines[i])
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
            func_defs.append("\n".join(block))
            continue

        i += 1

    return private_decls, func_defs


def _merge_mlir(per_file_mlir: list[str]) -> str:
    """Merge MLIR from multiple files into a single combined module.

    Deduplicates private declarations and concatenates all function
    definitions. In the merged module, all cross-module private symbols
    that have a concrete definition in some later file will be defined
    rather than declared external.
    """
    all_privates: dict[str, str] = {}  # name -> declaration line
    all_defs: dict[str, str] = {}      # name -> full function block

    for mlir_text in per_file_mlir:
        privates, defs = _extract_functions(mlir_text)
        for decl in privates:
            m = re.search(r"@(\w+)\(", decl)
            if m:
                name = m.group(1)
                all_privates.setdefault(name, decl)
        for block in defs:
            m = re.search(r"func\.func (?:@|\w+ @)(\w+)\(", block)
            if not m:
                # Try alternative: func.func @name(
                m = re.search(r"func\.func @(\w+)\(", block)
            if m:
                name = m.group(1)
                all_defs[name] = block

    # Any private declaration that has a concrete definition → remove from privates.
    remaining_privates = {
        name: decl
        for name, decl in all_privates.items()
        if name not in all_defs
    }

    lines: list[str] = ["module {"]

    # Emit remaining private declarations (truly external: __mind_*, libc)
    for decl in remaining_privates.values():
        lines.append(f"  {decl}")

    # Emit all function definitions
    for block in all_defs.values():
        # Indent the block by 2 spaces (it's already at module top level)
        indented = "\n".join(f"  {l}" if l.strip() else "" for l in block.splitlines())
        lines.append(indented)

    lines.append("}")
    return "\n".join(lines)


def _run_mlir_opt(mlir_text: str, mlir_opt: str) -> str:
    """Run mlir-opt to lower and validate the combined MLIR."""
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [
                mlir_opt,
                tmp_path,
                "--convert-func-to-llvm",
                "--convert-index-to-llvm",
                "--convert-arith-to-llvm",
                "--reconcile-unrealized-casts",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mlir-opt failed:\n{result.stderr}\n\nInput MLIR (first 50 lines):\n"
                + "\n".join(mlir_text.splitlines()[:50])
            )
        return result.stdout
    finally:
        os.unlink(tmp_path)


def _run_mlir_translate(llvm_mlir: str, mlir_translate: str) -> str:
    """Translate LLVM MLIR dialect to LLVM IR text."""
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(llvm_mlir)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [mlir_translate, "--mlir-to-llvmir", tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"mlir-translate failed:\n{result.stderr}")
        return result.stdout
    finally:
        os.unlink(tmp_path)


def _compile_runtime_support(
    clang: str,
    mind_checkout: Path,
    nerve_root: Path,
) -> list[str]:
    """Compile runtime support C files to temp .o files, return list of paths."""
    temp_objects: list[str] = []

    # Primary: mind_intrinsics.c from mindc checkout
    c_src = mind_checkout / "runtime-support/mind_intrinsics.c"
    if not c_src.exists():
        print("[build] runtime-support/mind_intrinsics.c not found; using inline stub")
        stub = _minimal_runtime_stub()
        tmp_c = tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False)
        tmp_c.write(stub)
        tmp_c.close()
        c_path = tmp_c.name
        cleanup_c = True
    else:
        c_path = str(c_src)
        cleanup_c = False

    tmp_o = tempfile.NamedTemporaryFile(suffix=".o", delete=False)
    tmp_o.close()
    result = subprocess.run(
        [clang, "-c", "-fPIC", "-O2", c_path, "-o", tmp_o.name],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if cleanup_c:
        os.unlink(c_path)
    if result.returncode != 0:
        raise RuntimeError(f"Compiling runtime support failed:\n{result.stderr}")
    temp_objects.append(tmp_o.name)

    # A1.5 LUT shims: tanh_q16, rsqrt_q16, softmax_q16 C implementations
    lut_shim = nerve_root / "mind/runtime/lut_shims.c"
    if lut_shim.exists():
        tmp_shim_o = tempfile.NamedTemporaryFile(suffix=".o", delete=False)
        tmp_shim_o.close()
        result = subprocess.run(
            [clang, "-c", "-fPIC", "-O2", str(lut_shim), "-o", tmp_shim_o.name, "-lm"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compiling lut_shims.c failed:\n{result.stderr}")
        temp_objects.append(tmp_shim_o.name)
        print("[build] Compiled A1.5 LUT shims (tanh_q16, rsqrt_q16, softmax_q16)")

    # A1.5 BLAS shim: i64-layout Q16.16 dot + score matmul with AVX2
    # dispatcher. Closes the 40x gap on the score path.
    blas_shim = nerve_root / "mind/runtime/blas_shims_i64.c"
    if blas_shim.exists():
        tmp_blas_o = tempfile.NamedTemporaryFile(suffix=".o", delete=False)
        tmp_blas_o.close()
        result = subprocess.run(
            [
                clang, "-c", "-fPIC", "-O2", "-mavx2", "-mfma",
                str(blas_shim), "-o", tmp_blas_o.name,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compiling blas_shims_i64.c failed:\n{result.stderr}")
        temp_objects.append(tmp_blas_o.name)
        print("[build] Compiled A1.5 BLAS shim (i64 Q16.16 dot + score matmul)")

    return temp_objects


def _minimal_runtime_stub() -> str:
    """Minimal C stub providing __mind_alloc/free/load/store for the .so."""
    return r"""
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int64_t __mind_alloc(int64_t n_bytes) {
    void *p = calloc(1, (size_t)n_bytes);
    return (int64_t)(uintptr_t)p;
}

int64_t __mind_free(int64_t addr) {
    if (addr) free((void *)(uintptr_t)addr);
    return 0;
}

int64_t __mind_load_i64(int64_t addr) {
    int64_t v;
    memcpy(&v, (void *)(uintptr_t)addr, 8);
    return v;
}

int64_t __mind_store_i64(int64_t addr, int64_t val) {
    memcpy((void *)(uintptr_t)addr, &val, 8);
    return 0;
}
"""


def _compile_shared(
    llvm_ir: str,
    runtime_objs: list[str],
    output: Path,
    clang: str,
) -> None:
    """Compile LLVM IR + runtime support objects to a shared library."""
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(llvm_ir)
        ll_path = f.name
    try:
        result = subprocess.run(
            [
                clang,
                "-shared",
                "-fPIC",
                "-O2",
                ll_path,
                *runtime_objs,
                "-o",
                str(output),
                "-lm",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"clang shared link failed:\n{result.stderr}"
            )
    finally:
        os.unlink(ll_path)


def build(
    nerve_root: Path,
    mind_checkout: Path,
    output: Path,
    *,
    verbose: bool = True,
) -> None:
    mindc = _mindc_path(mind_checkout)
    if not mindc.exists():
        raise FileNotFoundError(
            f"mindc not found at {mindc}. "
            f"Run: cargo build --release --bin mindc "
            f"--features 'std-surface cross-module-imports mlir-build' "
            f"inside {mind_checkout}"
        )

    # Resolve external tools
    mlir_opt = os.environ.get("MLIR_OPT") or "mlir-opt"
    mlir_translate = os.environ.get("MLIR_TRANSLATE") or "mlir-translate"
    clang = os.environ.get("CLANG") or "clang"

    sources = _source_list(nerve_root)
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

    # Step 1: generate MLIR for each source
    per_file_mlir: list[str] = []
    for src in sources:
        if verbose:
            print(f"[build] emit-mlir: {src.name}")
        mlir = _emit_mlir(mindc, src)
        per_file_mlir.append(mlir)

    # Step 2: merge into one combined module
    if verbose:
        print("[build] Merging MLIR modules...")
    combined = _merge_mlir(per_file_mlir)

    if verbose:
        fn_count = combined.count("func.func @")
        print(f"[build] Combined module: {fn_count} function definitions, "
              f"{len(combined)} chars")

    # Step 3: lower combined MLIR via mlir-opt
    if verbose:
        print("[build] Running mlir-opt (LLVM lowering)...")
    try:
        llvm_mlir = _run_mlir_opt(combined, mlir_opt)
    except RuntimeError as e:
        raise RuntimeError(f"mlir-opt step failed:\n{e}") from e

    # Step 4: translate to LLVM IR
    if verbose:
        print("[build] Running mlir-translate (LLVM IR)...")
    llvm_ir = _run_mlir_translate(llvm_mlir, mlir_translate)

    # Step 5: compile runtime support stubs
    if verbose:
        print("[build] Compiling runtime support stubs...")
    runtime_objs = _compile_runtime_support(clang, mind_checkout, nerve_root)

    # Step 6: link shared library
    output.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[build] Linking shared library -> {output}")
    try:
        _compile_shared(llvm_ir, runtime_objs, output, clang)
    finally:
        for obj in runtime_objs:
            if os.path.exists(obj):
                os.unlink(obj)

    so_size = output.stat().st_size
    if verbose:
        print(f"[build] Built: {output} ({so_size:,} bytes)")

    # Step 7: verify required symbols
    required = [
        "mn_encoder_init",
        "mn_encoder_encode",
        "mn_encoder_score",
        "mn_encoder_topk",
        "mn_encoder_free",
        "mn_encoder_version",
    ]
    result = subprocess.run(
        ["nm", "-D", str(output)],
        capture_output=True,
        text=True,
    )
    defined_t = set(re.findall(r" T (\w+)", result.stdout))
    missing = [sym for sym in required if sym not in defined_t]
    if missing:
        raise RuntimeError(
            f"Required symbols missing from {output}: {missing}"
        )
    if verbose:
        print("[build] Symbol check: all 6 mn_encoder_* symbols present as T")
        undefined = re.findall(r" U (\w+)", result.stdout)
        # Filter expected libc/libm symbols (versioned or bare)
        expected_undef = {
            "malloc", "free", "calloc", "realloc", "memcpy",
            "read", "write", "pread", "pwrite",
            "exp", "sqrt", "tanh",
            "getenv", "strcmp",
        }
        unexpected_undef = [
            u for u in undefined
            if u.split("@")[0] not in expected_undef
        ]
        if unexpected_undef:
            print(f"[build] WARNING: unexpected undefined symbols: {unexpected_undef}")
        else:
            print("[build] Undefined symbols: only libc (expected)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mind-checkout",
        default=os.environ.get("MIND_CHECKOUT", "/home/n/mind"),
        help="Path to the star-ga/mind checkout with built mindc",
    )
    parser.add_argument(
        "--nerve-root",
        default=str(Path(__file__).parent.parent),
        help="Path to mind-nerve repository root",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for libmind_nerve_encoder.so",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    nerve_root = Path(args.nerve_root).resolve()
    mind_checkout = Path(args.mind_checkout).resolve()
    output = Path(args.output) if args.output else (
        nerve_root / "python/mind_nerve/_native/libmind_nerve_encoder.so"
    )

    print(f"[build] mind-nerve root: {nerve_root}")
    print(f"[build] mindc checkout: {mind_checkout}")
    print(f"[build] output: {output}")

    try:
        build(nerve_root, mind_checkout, output, verbose=args.verbose)
    except Exception as e:
        print(f"[build] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("[build] Done.")


if __name__ == "__main__":
    main()
