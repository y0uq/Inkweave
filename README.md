# Inkling Vulkan

A performance fork of Unsloth llama.cpp for Inkling 2 Mini on AMD Vulkan. The starting point is the compiled source tree from [Unsloth b12588-mix-c0c3de8](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8).

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

From the repository root, download the model and projector and launch the OpenAI-compatible API.

```sh
uvx --from huggingface-hub hf download unsloth/Inkling-2-Mini-GGUF \
    Inkling-2-Mini-UD-Q3_K_XL.gguf mmproj-F16.gguf --local-dir models/Inkling
GGML_VK_VISIBLE_DEVICES=0 GGML_VK_DIRECT_HOST=1 GGML_VK_INKLING_BANDED=1 \
    ./build-vulkan/bin/llama-server --models-preset examples/inkling-vulkan.ini \
    --models-max 1 --offline --mmproj-device none --no-mmproj-offload \
    --host 127.0.0.1 --port 8080
```

The preset uses a 262,144-token context with Q8_0 K/V and Flash Attention. The RX 9060 XT configuration uses Vulkan0 for language inference and the CPU for the projector. The most frequently visited blocks keep their experts in VRAM. Vulkan buffers expose the other experts in pinned RAM to the GPU so routing and compute can run without per-layer CPU readbacks. Direct host access is opt-in and supports one visible Vulkan device.

The fork also fuses the SiTU expert activation on Vulkan, shares expert-cache entries across visits to the same physical block, reuses normalized residual states, and computes relative attention bias inside the Vulkan Flash Attention kernel on RDNA 4. This banded path uses Q8_0 K/V, FP32 accumulation, and position-contiguous cache ranges in a single sequence. The engine tiles the dense mask when it needs the fallback path. DSpark is configurable in the preset.

Use `--list-devices` to identify the Vulkan device on other systems. Tune `n-cpu-moe`, `override-tensor`, and `ubatch-size` for the available VRAM. The dense attention fallback can require more scratch space.

To compare dynamic caching, set `GGML_VK_DIRECT_HOST=0`, replace `n-cpu-moe` with `cpu-moe = true`, leave only the token-embedding placement in `override-tensor`, and set `moe-cache-mib` for the available VRAM.


## RX 9060 XT measurements

Measured with the included preset, Inkling 2 Mini UD-Q3_K_XL, 262,144 allocated context tokens, and greedy generation.

| Prompt workload | Prefill (tok/s) | Generation (tok/s) | Minimum free VRAM |
| --- | ---: | ---: | ---: |
| 1,825-token prefill and separate short-prompt decode | 763.7 | 29.7 | 3.01 GiB |
| 8,195-token prompt followed by 64 generated tokens | 632.7 | 27.3 | 3.01 GiB |
| 32,835-token prompt followed by 64 generated tokens | 569.9 | 24.9 | 2.80 GiB |

Free VRAM was sampled across the requests, including generation. The 32,835-token prompt was the longest measured. The preset keeps DSpark disabled for speed. The Q8_0 drafter is available from the same Hugging Face repository as `dspark-Inkling-2-Mini-Q8_0.gguf`.

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
