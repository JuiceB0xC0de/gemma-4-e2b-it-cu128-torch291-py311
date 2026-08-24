# Gemma-4-E2B-it CUDA 12.8 development image

One public `linux/amd64` image for Gemma-4-E2B-it attention-kernel and event-aware SAE work:

- CUDA 12.8.1 + cuDNN development base
- Python 3.11
- PyTorch 2.9.1 / TorchVision 0.24.1 / TorchAudio 2.9.1 from the official cu128 index
- Transformers 5.5.4
- [`zzhhjjj/gemma-triton-flash-attn`](https://github.com/zzhhjjj/gemma-triton-flash-attn) at `c6a0cf860612792597382e7745d1514a52d6ca58`
- [`JuiceB0xC0de/event-aware-SAE-trainer`](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer) at `319ba616e4562d98cd3603fd6551dfcf86393308`

The image contains source and developer tools. It contains **no model weights, checkpoints, datasets, credentials, automatic downloads, or automatic jobs**.

## Image tags

```text
juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311
juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311-<image-repo-short-sha>
juiceboxdocks/gemma-4-e2b-it-base:latest
```

Use the stable stack tag for normal launches and the SHA tag when exact image provenance matters.

## Run it

Docker:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v gemma-workspace:/workspace \
  juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311
```

## Persistent Triton cache

`TRITON_CACHE_DIR` is set to `/workspace/.cache/triton`. With persistent storage mounted at `/workspace`, kernels compiled during the first forward/backward pass survive container restarts.

Warm both representative Gemma attention families without model weights:

```bash
python /opt/image-tools/warm_triton_cache.py
```

Run it a second time with the same image, mount, and GPU architecture. `new_files: 0` is a useful reuse signal. A cache hit removes Triton compilation work; it does not remove every source of startup latency.

A100 (`sm_80`) and H100 (`sm_90`) compile different binaries, and both can coexist under the same cache root. Changing GPU architecture, tensor shape, kernel source, compiler options, Torch, or Triton can create new entries.

## GPU smoke test

```bash
python /opt/image-tools/gpu_smoke.py
```

The smoke test downloads no weights. It reports the GPU/driver/stack, imports the SAE trainer, registers the Triton attention adapter, and compares forward plus backward behavior against the package's PyTorch references for:

- Gemma full causal attention: D=512, 32 query heads, 4 KV heads
- Gemma sliding-window attention: D=256, 32 query heads, 16 KV heads, window 512

## Included development tools

The image includes `tmux`, Git, `nvcc`, `cuda-gdb`, `compute-sanitizer`, Nsight Systems (`nsys`), `btop`, `nvitop`, W&B, the modern `hf` CLI, Prometheus client, NVTX, `libaio-dev`, Ninja, `numactl`, and common shell/build diagnostics.

Pinned source trees are editable and inspectable at:

```text
/opt/src/gemma-triton-flash-attn
/opt/src/event-aware-SAE-trainer
```

## Verification boundary

GitHub Actions verifies the complete image build, package versions, source revisions, Transformers registration, trainer imports, and installed CLI tools. A CPU GitHub runner cannot prove CUDA execution or precompile A100/H100 Triton binaries. GPU and persistent-cache claims are considered verified only after the shipped image is exercised on the named architecture with the scripts above.

## Local repository checks

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py tests/*.py
uv pip compile requirements.txt --constraint constraints-cu128.txt \
  --python-version 3.11 --python-platform x86_64-unknown-linux-gnu \
  --output-file /tmp/gemma-image-resolved.txt
git diff --check
```
