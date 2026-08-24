#!/usr/bin/env python3
"""No-weights A100/H100 correctness smoke for Gemma Triton attention."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys


SUPPORTED_CAPABILITIES = {(8, 0), (9, 0)}
MIN_COSINE = 0.999
MAX_RELATIVE_ERROR = 0.05
MIN_GRAD_COSINE = 0.99
MAX_GRAD_RELATIVE_ERROR = 0.20


@dataclasses.dataclass(frozen=True)
class KernelCase:
    name: str
    batch: int
    query_heads: int
    kv_heads: int
    sequence: int
    head_dim: int
    causal: bool
    slide_size: int


CASES = (
    KernelCase(
        name="full_d512",
        batch=1,
        query_heads=32,
        kv_heads=4,
        sequence=512,
        head_dim=512,
        causal=True,
        slide_size=0,
    ),
    KernelCase(
        name="swa_d256",
        batch=1,
        query_heads=32,
        kv_heads=16,
        sequence=768,
        head_dim=256,
        causal=True,
        slide_size=512,
    ),
)


def validate_compute_capability(capability: tuple[int, int]) -> None:
    if capability not in SUPPORTED_CAPABILITIES:
        supported = ", ".join(f"sm_{major}{minor}" for major, minor in sorted(SUPPORTED_CAPABILITIES))
        raise RuntimeError(
            f"unsupported GPU compute capability sm_{capability[0]}{capability[1]}; "
            f"this image targets {supported}"
        )


def driver_version() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()[0].strip()


def similarity(actual, expected) -> tuple[float, float]:
    import torch

    actual_f = actual.detach().float().flatten()
    expected_f = expected.detach().float().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual_f, expected_f, dim=0).item()
    relative = ((actual_f - expected_f).norm() / expected_f.norm().clamp_min(1e-12)).item()
    return cosine, relative


def make_inputs(case: KernelCase, *, seed: int):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape_q = (case.batch, case.query_heads, case.sequence, case.head_dim)
    shape_kv = (case.batch, case.kv_heads, case.sequence, case.head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, generator=generator)
    return q, k, v


def run_triton_case(case: KernelCase, *, seed: int = 20260824):
    import torch
    from gemma_triton_flash_attn import flash_attn_gqa_train

    raw = make_inputs(case, seed=seed)
    tensors = tuple(item.detach().clone().requires_grad_(True) for item in raw)
    output = flash_attn_gqa_train(
        *tensors, causal=case.causal, slide_size=case.slide_size
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 1)
    grad_output = torch.randn(
        output.shape,
        device=output.device,
        dtype=output.dtype,
        generator=generator,
    )
    output.backward(grad_output)
    torch.cuda.synchronize()
    gradients = tuple(item.grad.detach().clone() for item in tensors)
    return output.detach(), gradients, raw, grad_output


def run_reference_case(case: KernelCase, raw, grad_output):
    import torch
    from gemma_triton_flash_attn import attention_gqa_ref, attention_swa_ref

    tensors = tuple(item.detach().clone().requires_grad_(True) for item in raw)
    if case.slide_size:
        output = attention_swa_ref(*tensors, slide_size=case.slide_size)
    else:
        output = attention_gqa_ref(*tensors, causal=case.causal)
    output.backward(grad_output)
    torch.cuda.synchronize()
    gradients = tuple(item.grad.detach().clone() for item in tensors)
    return output.detach(), gradients


def check_case(case: KernelCase) -> dict[str, object]:
    triton_output, triton_gradients, raw, grad_output = run_triton_case(case)
    reference_output, reference_gradients = run_reference_case(case, raw, grad_output)
    output_cosine, output_relative = similarity(triton_output, reference_output)

    if output_cosine <= MIN_COSINE or output_relative >= MAX_RELATIVE_ERROR:
        raise RuntimeError(
            f"{case.name}: output mismatch cosine={output_cosine:.6f}, "
            f"relative_error={output_relative:.6f}"
        )

    gradient_metrics = []
    for label, actual, expected in zip("qkv", triton_gradients, reference_gradients):
        if not actual.isfinite().all() or not expected.isfinite().all():
            raise RuntimeError(f"{case.name}: non-finite {label} gradient")
        cosine, relative = similarity(actual, expected)
        if cosine <= MIN_GRAD_COSINE or relative >= MAX_GRAD_RELATIVE_ERROR:
            raise RuntimeError(
                f"{case.name}: {label} gradient mismatch cosine={cosine:.6f}, "
                f"relative_error={relative:.6f}"
            )
        gradient_metrics.append({"tensor": label, "cosine": cosine, "relative_error": relative})

    return {
        "case": dataclasses.asdict(case),
        "output_cosine": output_cosine,
        "output_relative_error": output_relative,
        "gradients": gradient_metrics,
    }


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch import failed: {exc}") from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Start the container with an NVIDIA GPU and a "
            "host driver that supports CUDA 12.8."
        )

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    validate_compute_capability(capability)

    from gemma_triton_flash_attn import (
        patch_transformers_5_5_4_flash_attn_key,
        register_triton_attention,
    )

    patch_transformers_5_5_4_flash_attn_key()
    register_triton_attention()
    import sae_scheduler  # noqa: F401
    import sae_trainer_rolling  # noqa: F401

    report = {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"sm_{capability[0]}{capability[1]}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": driver_version(),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "cases": [check_case(case) for case in CASES],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GPU smoke failed: {exc}", file=sys.stderr)
        raise
