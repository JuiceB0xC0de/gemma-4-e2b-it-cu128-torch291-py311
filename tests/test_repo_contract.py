from __future__ import annotations

import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class DockerfileContractTests(unittest.TestCase):
    def test_stack_and_source_pins_are_exact(self) -> None:
        dockerfile = read("Dockerfile")
        requirements = read("requirements.txt")
        constraints = read("constraints-cu128.txt")

        self.assertIn("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", dockerfile)
        self.assertIn("ARG PYTHON_VERSION=3.11", dockerfile)
        self.assertIn("torch==2.9.1", dockerfile)
        self.assertIn("torchvision==0.24.1", dockerfile)
        self.assertIn("torchaudio==2.9.1", dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cu128", dockerfile)
        self.assertIn(
            "ARG GEMMA_ATTN_SHA=c6a0cf860612792597382e7745d1514a52d6ca58",
            dockerfile,
        )
        self.assertIn(
            "ARG SAE_TRAINER_SHA=319ba616e4562d98cd3603fd6551dfcf86393308",
            dockerfile,
        )
        self.assertIn("transformers==5.5.4", requirements)
        self.assertIn("torch==2.9.1", constraints)
        self.assertIn("-c /tmp/constraints-cu128.txt", dockerfile)
        self.assertIn("--no-deps -e /opt/src/event-aware-SAE-trainer", dockerfile)
        self.assertIn("PYTHONPATH=/opt/src/event-aware-SAE-trainer", dockerfile)

    def test_image_has_no_weights_or_automatic_download(self) -> None:
        dockerfile = read("Dockerfile").lower()

        self.assertNotIn("from_pretrained", dockerfile)
        self.assertNotIn("hf download", dockerfile)
        self.assertNotIn("huggingface-cli download", dockerfile)
        self.assertNotIn("git lfs", dockerfile)
        self.assertNotIn("volume ", dockerfile)

    def test_runtime_cache_and_process_contract(self) -> None:
        dockerfile = read("Dockerfile")

        self.assertIn("TRITON_CACHE_DIR=/workspace/.cache/triton", dockerfile)
        self.assertIn("HF_HOME=/workspace/.cache/huggingface", dockerfile)
        self.assertIn("TORCH_HOME=/workspace/.cache/torch", dockerfile)
        self.assertIn("WANDB_DIR=/workspace/.wandb", dockerfile)
        self.assertIn("WORKDIR /workspace", dockerfile)
        self.assertIn('ENTRYPOINT ["/usr/bin/tini", "--"]', dockerfile)
        self.assertIn('CMD ["/bin/bash", "-l"]', dockerfile)

    def test_required_developer_tools_are_installed_and_checked(self) -> None:
        dockerfile = read("Dockerfile")
        verifier = read("scripts/verify_image.py")

        for package in (
            "tmux",
            "git",
            "btop",
            "libaio-dev",
            "ninja-build",
            "numactl",
            "cuda-command-line-tools-12-8",
            "nsight-systems-2025.6.3",
        ):
            self.assertIn(package, dockerfile)
        for command in (
            "nvcc",
            "cuda-gdb",
            "compute-sanitizer",
            "nsys",
            "hf",
            "wandb",
            "btop",
            "nvitop",
        ):
            self.assertIn(command, verifier)


class RuntimeScriptContractTests(unittest.TestCase):
    def test_gpu_smoke_has_exact_architectures_and_kernel_cases(self) -> None:
        source = read("scripts/gpu_smoke.py")
        ast.parse(source)

        self.assertIn("SUPPORTED_CAPABILITIES = {(8, 0), (9, 0)}", source)
        self.assertIn('name="full_d512"', source)
        self.assertIn("sequence=512", source)
        self.assertIn("head_dim=512", source)
        self.assertIn('name="swa_d256"', source)
        self.assertIn("head_dim=256", source)
        self.assertIn("slide_size=512", source)
        self.assertIn("MIN_COSINE = 0.999", source)
        self.assertIn("MAX_RELATIVE_ERROR = 0.05", source)
        self.assertIn("output.backward(grad_output)", source)
        self.assertNotIn("output.float().square().mean()", source)

    def test_gpu_smoke_reads_driver_from_nvidia_smi(self) -> None:
        source = read("scripts/gpu_smoke.py")

        self.assertIn('"nvidia-smi"', source)
        self.assertNotIn("_cuda_getDriverVersion", source)

    def test_cache_warmup_reuses_smoke_case_definitions(self) -> None:
        source = read("scripts/warm_triton_cache.py")
        ast.parse(source)

        self.assertIn("from gpu_smoke import CASES", source)
        self.assertIn("TRITON_CACHE_DIR", source)
        self.assertIn("files_before", source)
        self.assertIn("files_after", source)


class WorkflowContractTests(unittest.TestCase):
    def test_workflow_builds_amd64_and_publishes_three_tags(self) -> None:
        workflow = read(".github/workflows/build.yml")

        self.assertIn("linux/amd64", workflow)
        self.assertIn("juiceboxdocks/gemma-4-e2b-it-base:latest", workflow)
        self.assertIn(
            "juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311", workflow
        )
        self.assertIn("cu128-torch291-py311-${{ steps.meta.outputs.short_sha }}", workflow)
        self.assertIn("Derive Docker Hub username", workflow)
        self.assertIn("${IMAGE%%/*}", workflow)
        self.assertIn("DOCKERHUB_TOKEN", workflow)
        self.assertIn("provenance: false", workflow)
        self.assertIn("sbom: false", workflow)

    def test_workflow_frees_disk_and_does_not_publish_pull_requests(self) -> None:
        workflow = read(".github/workflows/build.yml")

        self.assertIn("Free runner disk space", workflow)
        self.assertIn("docker/login-action", workflow)
        self.assertIn(
            "github.event_name == 'push' || (github.event_name == 'workflow_dispatch' && inputs.push_image)",
            workflow,
        )
        self.assertIn("push: ${{", workflow)


class ContextContractTests(unittest.TestCase):
    def test_docker_context_is_deny_by_default(self) -> None:
        dockerignore = read(".dockerignore")

        self.assertTrue(dockerignore.startswith("**\n"))
        self.assertIn("!Dockerfile", dockerignore)
        self.assertIn("!requirements.txt", dockerignore)
        self.assertIn("!scripts/", dockerignore)
        self.assertNotIn("!docs/", dockerignore)


if __name__ == "__main__":
    unittest.main()
