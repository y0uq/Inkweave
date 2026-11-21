# Inkling Vulkan

A performance fork of Unsloth llama.cpp for Inkling 2 Mini on Vulkan. The tuning target is the Radeon RX 9060 XT 16 GiB (gfx1200) and Ryzen 7 9700X (Zen 5). The fork started from [Unsloth b12588-mix-c0c3de8](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8).

## Build on Linux

Install CMake, Ninja, a C/C++ toolchain, Vulkan headers and loader, glslc, and the Vulkan driver for your GPU. This build uses native CPU tuning and the Vulkan backend.

```sh
cmake -S . -B build-vulkan -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_OPENMP=ON -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-vulkan --parallel --target llama-server
```

## Serve Inkling 2 Mini

From the repository root, download the model, Q8_0 DSpark drafter, and projector and launch the OpenAI-compatible API.

```sh
uvx --from huggingface-hub hf download unsloth/Inkling-2-Mini-GGUF \
    Inkling-2-Mini-UD-Q3_K_XL.gguf dspark-Inkling-2-Mini-Q8_0.gguf mmproj-F16.gguf \
    --local-dir models/Inkling

GGML_VK_VISIBLE_DEVICES=0 GGML_VK_DIRECT_HOST=1 GGML_VK_INKLING_BANDED=1 \
GGML_VK_INKLING_BANDED_DECODE=1 GGML_VK_INKLING_DEPTH_MIX=1 GGML_VK_INKLING_FUSED=1 \
GGML_VK_INKLING_BANDED_PROMPT_POSITIONS=1 GGML_VK_INKLING_BOUND_ATTN_SCRATCH=1 \
LLAMA_INKLING_POLY_MOE=1 LLAMA_INKLING_POLY_SNAPSHOT=1 LLAMA_INKLING_POLY_BATCH_UPLOAD=1 \
LLAMA_INKLING_DSPARK_DEVICE_FEATURES=1 LLAMA_INKLING_DSPARK_FOLD=1 LLAMA_INKLING_DSPARK_FOLD_Q8=1 \
    ./build-vulkan/bin/llama-server --models-preset examples/inkling-vulkan-dspark.ini \
    --models-max 1 --offline --mmproj-device none --no-mmproj-offload \
    --host 127.0.0.1 --port 8080
```

The DSpark preset allocates a 262,144-token context with Q8_0 K/V for both models, enables Flash Attention, and uses greedy target decoding with up to four draft guesses. Dense weights, KV, and DSpark run on Vulkan0. The projector runs on the CPU. The full expert pool lives in pinned RAM, with 5,750 MiB budgeted for the VRAM expert cache.

The executor shares expert-cache entries across visits to the same physical block and uses recency and reuse to select evictions. It targets a 60% upload and 40% CPU split for eligible misses, favoring CPU execution for low-reuse experts. Vulkan captures routes and activations in the router submission, groups weight uploads, and overlaps the CPU expert work with Vulkan execution. The folded Q8_0 correction matrix is used in the draft head. The target verifies the proposals.

The fork also fuses SiTU and residual operations and computes relative attention bias inside Vulkan Flash Attention on RDNA 4. The banded path uses FP32 accumulation and reads a compact position map when the sliding-window cache wraps. The scratch cap switches long requests to direct Q8_0 attention tiles. Direct host access supports a single visible Vulkan device.

Use [the direct preset](examples/inkling-vulkan.ini) for higher prompt throughput with DSpark disabled. For the greedy numbers below, send `temperature: 0` with the request. Tune `moe-cache-mib`, `moe-cache-cpu-share`, `ubatch-size`, and `spec-draft-n-max` in the DSpark preset.


## RX 9060 XT measurements

### Stock Unsloth starting point

On Zeus, the original Unsloth release delivered **165.92 tok/s prefill and 8.87 tok/s generation** with static expert offload and three DSpark guesses. These runs used a 262,144-token context, Q8_0 K/V, Flash Attention, and the same UD-Q3_K_XL target. Prefill was measured on a 1,825-token prompt. Generation was measured separately on a 79-token prompt with 96 generated tokens. [Original Unsloth release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)

| Stock Vulkan profile | DSpark guesses | Prefill (tok/s) | Generation (tok/s) |
| --- | ---: | ---: | ---: |
| Static expert offload | 3 | 165.92 | 8.87 |
| Pinned RAM experts, 3,072 MiB Vulkan cache | 3 | 148.63 | 9.45 |
| Same cache, 50% CPU share for eligible misses | 3 | 146.99 | 8.84 |
| Pinned RAM experts, 3,200 MiB Vulkan cache | 2 | 151.58 | 11.33 |
| Same 3,200 MiB cache, direct | 0 | 153.34 | 16.24 |

The first four rows average two measurements per rate. The direct run used one measurement per rate. In a separate diagnostic with the model fully resident in VRAM and the context reduced to 4,096, stock Vulkan reached 400.86 tok/s prefill and 24.33 tok/s generation. Its decode graph still had 194 splits.

### Inkling Vulkan iterations

Measured on the same RX 9060 XT and Ryzen 7 9700X with Inkling 2 Mini UD-Q3_K_XL, an 8,195-token mixed prompt, 128 generated tokens, greedy sampling, and 262,144 allocated context tokens.

| Profile | Prefill (tok/s) | Generation (tok/s) | Minimum free VRAM during generation |
| --- | ---: | ---: | ---: |
| First Inkling Vulkan iteration (`6e5256529`), direct | 840.77 | 27.79 | 3,422 MiB |
| Tuned direct | **1,095.17** | 34.48 | 3,488 MiB |
| Mixed expert execution, direct | 929.53 | 32.56 | 4,259 MiB |
| Mixed expert execution and four DSpark guesses | 898.88 | **56.15** | 2,899 MiB |

Against the first Inkling Vulkan iteration on the 8K fixture, the tuned direct path improved prefill by 30% and generation by 24%. The four-guess DSpark profile reached 2.02× the first iteration's generation rate. On the 49-token code fixture, generation progressed from 30.00 tok/s in the first iteration to 36.18 tok/s with tuned direct and 50.16 tok/s with four guesses. Early fork tests on the 79-token prompt measured 13.66 tok/s with two guesses and 26.87 tok/s with direct decoding.

The four-guess profile matched all 128 greedy tokens from the tuned direct reference on both fixtures. It maintained at least 2,899 MiB free during generation, 339 MiB above the test floor. Acceptance was 99 of 109 proposals at 8K and 90 of 148 on code. Rates vary with prompt, sampling and acceptance. The 262,144-token allocations were verified at load, and near-full generation headroom for the four-guess preset still needs qualification.

<details>
<summary>Upstream llama.cpp README</summary>

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

</details>
