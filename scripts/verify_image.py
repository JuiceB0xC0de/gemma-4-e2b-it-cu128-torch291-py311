#!/usr/bin/env python3
"""Build-time, no-network receipt for the finished image."""

from __future__ import annotations

import importlib.metadata
import json
import os
import pathlib
import shutil
import subprocess


EXPECTED = {
    "torch": "2.9.1",
    "torchvision": "0.24.1",
    "torchaudio": "2.9.1",
    "transformers": "5.5.4",
}
SOURCE_DIRS = {
    "gemma_attention": pathlib.Path("/opt/src/gemma-triton-flash-attn"),
    "sae_trainer": pathlib.Path("/opt/src/event-aware-SAE-trainer"),
}
SOURCE_ENV = {
    "gemma_attention": "GEMMA_ATTN_SHA",
    "sae_trainer": "SAE_TRAINER_SHA",
}
COMMANDS = {
    "nvcc": ["nvcc", "--version"],
    "cuda-gdb": ["cuda-gdb", "--version"],
    "compute-sanitizer": ["compute-sanitizer", "--version"],
    "nsys": ["nsys", "--version"],
    "hf": ["hf", "--version"],
    "wandb": ["wandb", "--version"],
    "btop": ["btop", "--version"],
    "nvitop": ["nvitop", "--version"],
}


def distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def source_revision(source_dir: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()


def command_version(argv: list[str]) -> str:
    if shutil.which(argv[0]) is None:
        raise RuntimeError(f"required command not found: {argv[0]}")
    completed = subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else "available"


def main() -> None:
    import torch
    import torchaudio
    import torchvision
    import transformers

    for package, expected in EXPECTED.items():
        actual = distribution_version(package)
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {actual}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Torch expected CUDA 12.8, found {torch.version.cuda}")

    from gemma_triton_flash_attn import (
        patch_transformers_5_5_4_flash_attn_key,
        register_triton_attention,
    )

    patch_transformers_5_5_4_flash_attn_key()
    register_triton_attention()
    from transformers import Gemma4ForConditionalGeneration  # noqa: F401
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config  # noqa: F401

    if "triton_gqa" not in ALL_ATTENTION_FUNCTIONS:
        raise RuntimeError("triton_gqa was not registered with Transformers")

    import sae_diagnostics  # noqa: F401
    import sae_scheduler  # noqa: F401
    import sae_trainer_rolling  # noqa: F401

    revisions: dict[str, str] = {}
    for name, source_dir in SOURCE_DIRS.items():
        actual = source_revision(source_dir)
        expected = os.environ[SOURCE_ENV[name]]
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {actual}")
        revisions[name] = actual

    receipt = {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "triton": distribution_version("triton"),
        "sources": revisions,
        "tools": {name: command_version(argv) for name, argv in COMMANDS.items()},
        "weights_included": False,
    }
    print("IMAGE_BUILD_RECEIPT=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
