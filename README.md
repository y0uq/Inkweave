# Inkweave

Inkweave is a Vulkan performance fork of Unsloth llama.cpp for Inkling 2 Mini. It targets Linux on the Radeon RX 9060 XT 16 GB (`gfx1200`) and Ryzen 7 9700X (Zen 5). Inkling reuses its middle blocks across three passes, so efficient reuse of expert weights and intermediate state matters for both prompt processing and generation. [Inkling 2 Mini model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) · [Unsloth starting release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)

## Requirements and scope

| Item | Tested configuration |
| --- | --- |
| OS and driver | CachyOS x86-64, Linux 7.3.4, amdgpu, Mesa RADV 26.2.6 |
| Build tools | GCC 16.2.1, CMake 4.5.0, Ninja 1.13.2, Vulkan headers and loader 1.4.363, and `glslc` from shaderc 2026.4 |
| System RAM | 32 GB on the testing system. The mixed expert preset alone allocates about 11.36 GiB of Vulkan host RAM for target weights. Leave RAM for the CPU projector, other allocations, and the OS. |
| Storage | 13.63 GB for the UD-Q3_K_XL target, 402 MB for the Q8_0 DSpark drafter, and 639 MB for the F16 projector. Allow at least 14.68 GB for the three model files, plus build and download cache space. |
| GPU memory | 16 GB tested, including 5,750 MiB budgeted for the mixed expert cache. The main and draft models each allocate a 262,144-token Q8_0 K/V context with separate sliding-window buffers. |

The Vulkan build files declare a CMake 3.19 minimum and need Vulkan headers, a loader, and `glslc` that can target Vulkan 1.3. The build probes shader features. Use the toolchain in the table as the known working reference. An older Vulkan SDK minimum and older Mesa versions have not been qualified. [Vulkan CMake configuration](ggml/src/ggml-vulkan/CMakeLists.txt)

The single-device Vulkan host buffers are driver allocations. The preset runs with an 8 MiB Linux `RLIMIT_MEMLOCK` and without `--mlock`. It needs enough driver-allocatable host memory for the expert pool. Check for `Vulkan_Host model buffer size` at startup. [Vulkan host allocator](ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp)

RDNA 3, other Radeon cards, NVIDIA, and Windows are untested for these presets. The fused and banded Vulkan kernels check RDNA 4 and device capabilities. Attention graph construction checks that the selected backend supports the operation before using it. Other switches have their own checks and can change graph placement even when a specialized shader is unavailable. On other systems start with the general backend paths and leave the Inkweave switches unset until you can check the log, output, and memory use. [Vulkan operation checks](ggml/src/ggml-vulkan/ggml-vulkan.cpp) · [Inkling graph](src/models/inkling2.cpp)

## Quick start

Install Git, CMake, Ninja, a C/C++ toolchain with OpenMP, and the Vulkan dependencies above. Build from this repository.

```sh
git clone https://github.com/y0uq/Inkweave.git
cd Inkweave

cmake -S . -B build-vulkan -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_OPENMP=ON -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-vulkan --parallel 4 --target llama-server
```

From the repository root, fetch the pinned Unsloth GGUFs with `uvx`. The direct preset needs the target and projector. The DSpark preset also needs the drafter. [GGUF files](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f)

```sh
uvx --from huggingface-hub hf download unsloth/Inkling-2-Mini-GGUF \
    --revision ae9327c5fbc7333116ba32eee78bb5d2bbefae4f \
    Inkling-2-Mini-UD-Q3_K_XL.gguf dspark-Inkling-2-Mini-Q8_0.gguf mmproj-F16.gguf \
    --local-dir models/Inkling
```

Use `vulkaninfo --summary` to check device order. Set `GGML_VK_VISIBLE_DEVICES` to the index of the discrete Radeon if it differs from `0`. The selected device is exposed to this command as `Vulkan0`. Start the OpenAI-compatible server from the repository root.

```sh
GGML_VK_VISIBLE_DEVICES=0 GGML_VK_DIRECT_HOST=1 GGML_VK_INKLING_BANDED=1 \
GGML_VK_INKLING_BANDED_DECODE=1 GGML_VK_INKLING_DEPTH_MIX=1 GGML_VK_INKLING_FUSED=1 \
GGML_VK_INKLING_BANDED_PROMPT_POSITIONS=1 GGML_VK_INKLING_BOUND_ATTN_SCRATCH=1 \
LLAMA_INKLING_POLY_MOE=1 LLAMA_INKLING_POLY_SNAPSHOT=1 LLAMA_INKLING_POLY_BATCH_UPLOAD=1 \
LLAMA_INKLING_DSPARK_DEVICE_FEATURES=1 LLAMA_INKLING_DSPARK_FOLD=1 LLAMA_INKLING_DSPARK_FOLD_Q8=1 \
    ./build-vulkan/bin/llama-server --models-preset examples/inkweave-dspark.ini \
    --models-max 1 --offline --mmproj-device none --no-mmproj-offload \
    --host 127.0.0.1 --port 8080
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"Name three ways to test a function."}],"max_tokens":64}'
```

Use [`examples/inkweave.ini`](examples/inkweave.ini) in the same command for direct decoding with static expert placement. Both examples use temperature 1.0, top-p 1.0, top-k disabled, and min-p 0. The DSpark preset keeps draft proposals greedy while the target samples. The launcher flags keep the vision projector on the CPU. Set target sampling in the preset or per request. [Unsloth sampling guidance](https://unsloth.ai/docs/models/inkling-2-mini) · [Server API](tools/server/README.md)

## Tuning controls

The INI accepts registered server options such as `ctx-size`, `cache-type-k`, `ubatch-size`, `moe-cache-mib`, `moe-cache-cpu-share`, and `spec-draft-n-max`. The switches below are process environment variables read directly by the backend or model code and have no INI equivalent. They are opt-in so the general backend paths stay available on other setups, and so each optimization can be tested separately. Unset a switch to disable it. Several checks treat even `0` as enabled. [Preset parser](common/preset.cpp)

| Environment variable | Effect | When to watch for problems |
| --- | --- | --- |
| `GGML_VK_VISIBLE_DEVICES` | Filters Vulkan devices by index. Use one discrete device for this setup. | The wrong index can select an iGPU or leave no Vulkan device. |
| `GGML_VK_DIRECT_HOST` | Allows eligible Vulkan operations to bind weights in host RAM directly. | Requires a single visible device for this discrete-GPU path. Host reads go over PCIe. |
| `GGML_VK_INKLING_BANDED` | Allows the specialized Flash Attention kernel to compute relative position bias as it visits K/V tiles. | Requires compatible RDNA 4 cooperative matrices and the tested Q8_0 layout. |
| `GGML_VK_INKLING_BANDED_DECODE` | Extends the attention path to generation, including a position map for wrapped sliding-window K/V. | Test at and beyond the window boundary when changing context or cache settings. |
| `GGML_VK_INKLING_DEPTH_MIX` | Fuses the weighted combination of earlier block states. | Subgroup and tensor-layout checks apply. A different reduction order can affect numerical parity. |
| `GGML_VK_INKLING_FUSED` | Fuses operations including normalization, gated residual updates, and SiTU-related graph work. | Device and graph checks apply. Recheck numerical parity on new drivers. |
| `GGML_VK_INKLING_BANDED_PROMPT_POSITIONS` | Supplies compact K/V position metadata during prompt processing. | Uses the banded path and depends on cache layout. Recheck long prefill on other configurations. |
| `GGML_VK_INKLING_BOUND_ATTN_SCRATCH` | Lets long relative-attention calls read quantized K/V tiles without allocating a full F16 copy above a 256 MiB conversion threshold. | Can trade prefill speed for lower peak memory. It caps an individual conversion decision, not total VRAM. |
| `LLAMA_INKLING_POLY_MOE` | Uses the PolyStrata-inspired mixed expert executor with a configured expert cache and CPU miss share. | Costs pinned RAM and depends on the balance of CPU compute, PCIe, and GPU reuse. |
| `LLAMA_INKLING_POLY_SNAPSHOT` | Captures routed expert IDs and activations during the router submission for the mixed executor. | Depends on compatible Vulkan placements and buffer layouts. |
| `LLAMA_INKLING_POLY_BATCH_UPLOAD` | Groups cache-miss weight uploads for the mixed executor. | Throughput depends on batch size and expert reuse. |
| `LLAMA_INKLING_DSPARK_DEVICE_FEATURES` | Keeps eligible target activations on the device for the DSpark drafter. | Applies to eligible device placements. Has no effect on direct decoding. |
| `LLAMA_INKLING_DSPARK_FOLD` | Precomputes the draft head's correction projection instead of repeating it for each proposal. | Spends GPU memory at load and can change draft scores through rounding. Target verification still runs. |
| `LLAMA_INKLING_DSPARK_FOLD_Q8` | Stores the precomputed draft correction matrix in Q8_0, using about 53 MiB in the tested model. | Requires the fold switch and a block-aligned rank. Quantization can change proposed tokens and acceptance. |

**Expert execution.** Inkling is a mixture-of-experts (MoE) model. Its router selects a subset of feed-forward networks for each token. The mixed preset keeps the full expert pool in Vulkan host RAM and caches frequently reused weights in VRAM across repeated visits to the same physical block. The default `moe-cache-cpu-share = 0.4` targets 40% CPU execution for eligible misses. Cache fills account for recency and reuse, while CPU work overlaps with Vulkan expert work. [Expert cache implementation](src/llama-moe-cache.cpp)

**Attention and activations.** Banded attention uses compact relative-position and K/V position data to compute attention bias inside the shader. Sliding-window layers attend to a recent range of tokens. Depth mixing combines activations from earlier blocks. SiTU is Inkling's expert feed-forward activation. It combines tanh soft clipping with a sigmoid gate. [Inkling model graph](src/models/inkling2.cpp) · [SiTU implementation](ggml/src/ggml-cpu/ops.cpp)

**Speculation.** DSpark is a smaller model that reads target activations and proposes tokens for the full target to verify. The folded correction matrix combines the drafter's low-rank, previous-token correction with the output projection ahead of time. These controls affect the drafter and can influence how many proposals the target accepts. [DSpark implementation](src/models/dspark.cpp) · [Fold implementation](src/llama-context.cpp)

## RX 9060 XT measurements

The comparison below uses the same 8,195-token mixed prompt and generates 128 tokens at temperature 0. Every run uses the pinned UD-Q3_K_XL files, 262,144 allocated context tokens, Q8_0 K/V and Flash Attention. The stock and Inkweave setups use their own measured expert placement, batch sizes and draft widths. [Exact profiles and fixture generator](docs/inkweave-benchmarks.md)

| Build and profile | Prefill (tok/s) | Generation (tok/s) | Draft accepted | Minimum free VRAM during generation |
| --- | ---: | ---: | ---: | ---: |
| Original Unsloth release, static offload and 3 guesses | 172.39 | 14.77 | 93/101 | 5,914 MiB |
| First Inkweave iteration, [`6e5256529`](https://github.com/y0uq/Inkweave/commit/6e5256529f8d97b0b54fe9928e0f94d74ac2c065), direct | 840.77 | 27.79 | | 3,422 MiB |
| Clean [`2c5b8c912`](https://github.com/y0uq/Inkweave/commit/2c5b8c912460b5c2c8faf0cf3c913f459bccc1d9), static direct | 1,128.95 | 34.34 | | 3,502 MiB |
| Same commit, mixed expert cache, direct | 955.35 | 32.87 | | 4,273 MiB |
| Same commit, mixed expert cache with 4 DSpark guesses | 898.04 | 57.23 | 99/109 | 2,892 MiB |

The earlier **56.15 tok/s** 8K figure was a staging run of the runtime changes recorded in [`bb3295dd6`](https://github.com/y0uq/Inkweave/commit/bb3295dd664bfb039aacfd66f07fe0b0b6174b0d). On the 49-token code fixture, that run reached 50.16 tok/s with 90/148 drafts accepted. The clean build reproduced 90/148 and reached 49.58 tok/s. On both fixtures its DSpark output matched both direct profiles for all 128 greedy tokens. The stock output also matched all 128 on the 8K fixture. Stock incurred modest drive reads in its logged run, with zero process swap. The measured floor was 2,560 MiB during generation in the staging test and is 2,048 MiB for the current tests. The clean build cleared the generation floor in all the tabled runs. Near-full generation and broader model quality still need validation. [Build provenance and measurement details](docs/inkweave-benchmarks.md)

**Sampled generation.** The same clean commit was tested at temperature 1.0, top-p 1.0, top-k disabled and min-p 0. The table shows generation tok/s over 64 output tokens per fixture and seed. Both profiles use the same 5,750 MiB expert cache.

| Profile | Code median | 8K median | 8K range |
| --- | ---: | ---: | ---: |
| Mixed expert cache, direct | 32.55 | 30.88 | 30.56-32.00 |
| Mixed expert cache, 4 DSpark guesses | 28.37 | 18.02 | 16.85-44.63 |

Choose direct decoding for more consistent sampled throughput on these fixtures. At 8K, DSpark accepted between 13/190 and 48/58 drafts across the three seeds. The greedy match and three sampled seeds establish a small correctness and performance baseline. [Sampling method](docs/inkweave-benchmarks.md#reproducing-the-fixtures)

## Contributing and upstream

For an Inkweave change, open a pull request against this repository's default branch. Include the affected model and quant, backend and driver, build commit, preset and environment switches, and a reproduction with comparable before and after timings or output checks. [Inkweave contributing notes](CONTRIBUTING.md)

Inkweave builds on [Unsloth llama.cpp](https://github.com/unslothai/llama.cpp) and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). See the [MIT license](LICENSE) for engine code and the model card for weight licensing.
