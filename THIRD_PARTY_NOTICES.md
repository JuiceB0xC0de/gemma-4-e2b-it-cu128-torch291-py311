# Third-party notices

This repository's Dockerfile, validation scripts, and documentation are MIT licensed. The built image also contains independently licensed third-party software. The repository license does not replace those terms.

| Component | Source | Pin / version | Declared license |
|---|---|---|---|
| Gemma Triton Flash Attention | <https://github.com/zzhhjjj/gemma-triton-flash-attn> | `c6a0cf860612792597382e7745d1514a52d6ca58` | Apache-2.0 |
| Event-aware SAE trainer | <https://github.com/JuiceB0xC0de/event-aware-SAE-trainer> | `319ba616e4562d98cd3603fd6551dfcf86393308` | MIT |
| PyTorch | <https://pytorch.org/> | 2.9.1 | BSD-style |
| NVIDIA CUDA container and tools | <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda> | CUDA 12.8.1 family | NVIDIA container/tool license terms |
| Hugging Face Transformers | <https://github.com/huggingface/transformers> | 5.5.4 | Apache-2.0 |

The image package database and each installed source tree retain their own metadata and notices. Redistributors are responsible for reviewing all transitive package licenses and the NVIDIA container license for their use case.

`google/gemma-4-E2B-it` weights are not included in this repository or image. Any separately downloaded model is governed by its own model card and license terms.
