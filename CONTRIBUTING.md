# Contributing to Inkweave

Inkweave focuses on Inkling 2 Mini on Linux with a Radeon RX 9060 XT and a Zen 5 CPU. Open changes against [Inkweave's default branch](https://github.com/y0uq/Inkweave). Report general upstream issues to the repository where you reproduced them. [Unsloth llama.cpp](https://github.com/unslothai/llama.cpp) · [ggml-org llama.cpp](https://github.com/ggml-org/llama.cpp)

For a bug report or performance change, include the build commit, GPU and driver, exact model files, command or preset, environment switches, and the shortest input that reproduces the result. Compare performance with the same fixture, allocation, output length, and sampling settings on both builds. For speculative decoding include draft acceptance, and for changes to placement or buffers include minimum free VRAM during generation. Use greedy token comparisons for numerical checks and sampled runs for ordinary generation throughput. [Setup and performance notes](README.md)
