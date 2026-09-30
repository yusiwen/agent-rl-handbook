# Compute Tiers and How to Split the Work

## The reference machine: DGX Spark (GB10)

| Dimension | Specification | What it means for studying |
|---|---|---|
| Memory | 128 GB **unified** (shared between CPU and GPU) | ✅ The big win: an 8B full fine-tune (weights + optimiser + gradients) fits, and four models can live in memory at once — impossible on a 24 GB card |
| Bandwidth | ≈ 273 GB/s (LPDDR5X; NVIDIA has not published an official figure) | ⚠️ The real bottleneck. An H100 is 3.35 TB/s and a 4090 is 1 TB/s. RL is rollout-heavy, so throughput lands an order of magnitude lower |
| Architecture | **aarch64 + sm_121** | ⚠️ The awkward zone: official images are mostly x86_64, so you need native aarch64 installs or GB10-specific images |
| OS | DGX OS (Ubuntu 24.04 LTS) | GPU access from Docker goes through CDI (`/var/run/cdi/nvidia.yaml`) |

Reference throughput (NVIDIA's own SFT benchmarks): Llama 3.2 3B full fine-tune 82,739 tok/s; Llama 3.1 8B LoRA 53,657 tok/s; Llama 3.3 70B QLoRA 5,079 tok/s. **For RL, expect one third to one fifth of that.**

## The three tiers

| Tier | Meaning | How to handle it |
|---|---|---|
| ✅ | Comfortable on the Spark | Do it locally, keep the service running |
| ⚠️ | Runs, but must be shrunk | Small models (0.6–3B), QLoRA, a different attention backend, capped concurrency |
| ❌ | Needs multiple GPUs or another machine | Rent cloud GPUs by the hour; the goal is to *understand* industrial scale, not to reproduce it |

## Known platform traps (Spark)

| Component | Problem | Workaround |
|---|---|---|
| vLLM | upstream has no sm_121 / aarch64 support; standard images fail with `SM121 not supported` | use a GB10-patched image, or fall back to Ollama / SGLang |
| FlashAttention | no sm_121 wheel, and building from source does not work either; CUTLASS MoE kernels and CUDA-graph capture break with it | set the attention backend to `sdpa`, or use FlashInfer |
| Building from source | aggressive parallelism exhausts the shared memory | cap `MAX_JOBS` (for example 4) |
| NGC containers | mostly x86_64 | install natively for aarch64, or build your own image |
| The Megatron route | a separate engineering project on aarch64 | stay on the single-GPU FSDP path; move multi-node work off the machine |

> The hardware groundwork (Secure Boot + MOK, DKMS driver, CDI) is already done and Docker can reach the GPU — that is the hardest first door in this stack, and it is behind us.

## How to split the work

**The Spark as the always-on workbench:**

- End-to-end loops at 0.6–3B (Chapters 5, 6, 10)
- Reward infrastructure and sandbox environments (Chapters 5, 8)
- Evaluation services and regression tests (Chapter 11)
- The skill-evolution service, running 24/7 (Chapter 12)

**Rented by the hour:**

- Multi-node / multi-GPU VeRL, Megatron, anything at the 70B scale (the heavy variants of Chapter 9)
- Purely to see what industrial scale looks like — not to reproduce it

**The boundary:** what the Spark cannot run is not the same as what you cannot learn. Chapter 9's architecture and memory ledger read just as well on a single GPU with a small model.
