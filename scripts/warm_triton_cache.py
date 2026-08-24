#!/usr/bin/env python3
"""Compile representative Gemma kernels into the persistent Triton cache."""

from __future__ import annotations

import json
import os
import pathlib

from gpu_smoke import CASES, run_triton_case, validate_compute_capability


def cache_files(cache_dir: pathlib.Path) -> set[str]:
    return {
        str(item.relative_to(cache_dir))
        for item in cache_dir.rglob("*")
        if item.is_file()
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; Triton cache warmup requires an NVIDIA GPU.")

    capability = torch.cuda.get_device_capability()
    validate_compute_capability(capability)

    raw_cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if not raw_cache_dir:
        raise SystemExit("TRITON_CACHE_DIR is unset")
    cache_dir = pathlib.Path(raw_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = cache_dir / ".write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"Triton cache is not writable at {cache_dir}: {exc}. "
            "Mount persistent storage at /workspace or this exact directory."
        ) from exc

    files_before = cache_files(cache_dir)
    warmed = []
    for case in CASES:
        output, gradients, _, _ = run_triton_case(case)
        if not output.isfinite().all() or not all(gradient.isfinite().all() for gradient in gradients):
            raise RuntimeError(f"{case.name}: warmup produced non-finite output or gradients")
        warmed.append(case.name)
    files_after = cache_files(cache_dir)

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "compute_capability": f"sm_{capability[0]}{capability[1]}",
                "triton_cache_dir": str(cache_dir),
                "warmed_cases": warmed,
                "files_before": len(files_before),
                "files_after": len(files_after),
                "new_files": len(files_after - files_before),
                "cache_reuse_hint": (
                    "Run this command again with the same image, GPU architecture, and "
                    "persistent mount. A zero new_files count is a useful cache-reuse signal."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
