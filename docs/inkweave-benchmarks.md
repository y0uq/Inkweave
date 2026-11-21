# Inkweave benchmark method

Tests use one request at a time, with the Radeon RX 9060 XT as Vulkan0, the CPU projector, Flash Attention, 262,144 context tokens, and Q8_0 K and V. Speculative profiles also give the drafter a 262,144-token Q8_0 context. The testing system ran in a Linux text console with about 65 MiB of idle GPU use and the existing CPU power-saving policy.

## Build provenance

| Build | Source |
| --- | --- |
| Starting Unsloth build | [`b12588-mix-c0c3de8`](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8), built locally from the release source. The binary reports `883286076`. The corresponding import is [`80ae8ff50`](https://github.com/y0uq/Inkweave/commit/80ae8ff506fda9489552a5fb9b6181308aefc357). |
| First Inkweave iteration | [`6e5256529`](https://github.com/y0uq/Inkweave/commit/6e5256529f8d97b0b54fe9928e0f94d74ac2c065) |
| Mixed expert execution and Q8_0 draft correction | Runtime code recorded in [`bb3295dd6`](https://github.com/y0uq/Inkweave/commit/bb3295dd664bfb039aacfd66f07fe0b0b6174b0d). The earlier 56.15 tok/s result came from a staging build before the changes were committed. That binary reported its base `6e5256529`. |
| Clean public build in the current comparison | [`2c5b8c912`](https://github.com/y0uq/Inkweave/commit/2c5b8c912460b5c2c8faf0cf3c913f459bccc1d9), with the same runtime code as `bb3295dd6`. |

## Profiles

Build the stock release with the Vulkan, native CPU, LTO and OpenMP flags from the [Inkweave quick start](../README.md#quick-start). Run its server with the same GGUF files and CPU projector flags. Start from the DSpark preset and remove `moe-cache-mib`, `moe-cache-cpu-share`, and `spec-draft-sampling` to use the stock drafter default. Set these values:

```ini
n-cpu-moe = 12
override-tensor = ^blk[.](18|19|20|21|22|23)[.]ffn_(up|down|gate|gate_up)_(ch|)exps=CPU
batch-size = 512
ubatch-size = 128
spec-draft-n-max = 3
```

Use `GGML_VK_VISIBLE_DEVICES=0` and leave the Inkweave switches unset for stock. The clean Inkweave build uses the [DSpark preset](../examples/inkweave-dspark.ini) and fast-path switches from the quick start. The direct comparison with the same cache uses the DSpark preset with `model-draft` removed and `spec-type = none`. The static direct fixture uses [the static preset](../examples/inkweave.ini). These measurements compare whole profiles. The stock placement, batch sizes and speculation width differ.

## Reproducing the fixtures

Start one server at a time on port 8080. Warm each server with a short, unrelated completion before timing. This Python 3 snippet builds the same synthetic fixtures through the server tokenizer and sends requests to `/completion`. Choose `code` or `mixed` in the request. Greedy tests use 128 generated tokens and seed 42. Sampled tests use 64 tokens per seed at temperature 1.0, top-p 1.0, top-k disabled and min-p 0.

```python
import json
from urllib.request import Request, urlopen

model = "Inkling-2-Mini"

def post(path, body):
    body = json.dumps({"model": model, **body}).encode()
    request = Request("http://127.0.0.1:8080" + path, body, {"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        return json.load(response)

def tokenize(text, special=True):
    return post("/tokenize", {"content": text, "add_special": special, "parse_special": False})["tokens"]

topics = ("warehouse inventory", "school attendance", "water filtration", "weather records",
          "staff scheduling", "software maintenance", "public transport", "orchard irrigation",
          "municipal finance", "archive conservation", "power consumption", "clinic records")
entries = []
for i in range(1700):
    topic = topics[(i * 17 + i // 5) % len(topics)]
    entries.append(
        f"Entry {i + 1}. The {topic} team reviewed {71 + (i * 23) % 943} records across "
        f"{2 + i % 17} sites. They compared those records with the previous report and "
        f"found {5 + (i * 11) % 91} discrepancies. Explain which of the discrepancies would "
        "justify an independent audit. Consider missing records, duplicate entries, and the "
        "effect of seasonality. Draft the finding as a note to a new coordinator.\n"
    )
notes = tokenize("Independent audit notes.\n" + "".join(entries))
tail = tokenize("\nUsing these notes, explain three concrete review priorities.\nAnswer:", False)
mixed = notes[:8195 - len(tail)] + tail
code = tokenize(
    "Write a Python function to enumerate the strongly connected components of a directed "
    "graph iteratively. Include an example, an explanation of the low-link invariant, and "
    "the time and memory bounds. Use the following format:\n\nImplementation:\n```python\n"
)
assert len(mixed) == 8195 and len(code) == 49

for temperature, n_predict, seeds in [(0, 128, [42]), (1, 64, [1729, 42, 1730])]:
    for seed in seeds:
        result = post("/completion", {
            "prompt": mixed, "n_predict": n_predict, "temperature": temperature,
            "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": seed,
            "ignore_eos": True, "cache_prompt": False, "return_tokens": True,
        })
        print(temperature, seed, result["timings"])
```

Prefill and generation rates use the server's prompt and predicted tokens per second. The benchmark logged free VRAM from the Radeon PCI device's `mem_info_vram_total` minus `mem_info_vram_used` about every 25 ms. The lowest sample during generation is reported separately from the lowest sample across the whole request. The historical pass threshold was 2,560 MiB of free VRAM during generation. The current local target is 2,048 MiB. Allocation of a 262,144-token context does not imply qualification at that prompt length.

## Additional controls

The stock and frozen four-guess Inkweave binaries also ran the same 8K fixture sequentially with a fresh prompt at each temperature. The frozen binary used the runtime code recorded in `bb3295dd6` and its own library directory. This shows the sampled comparison on one fixed seed. The README includes three-seed measurements from a clean Inkweave build.

| Profile | Temperature | Seed | Output tokens | Prefill (tok/s) | Generation (tok/s) | Draft accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock | 0 | 42 | 128 | 172.39 | 14.77 | 93/101 |
| Frozen Inkweave | 0 | 42 | 128 | 892.48 | 55.25 | 99/109 |
| Stock | 1 | 1729 | 64 | 140.49 | 5.98 | 25/111 |
| Frozen Inkweave | 1 | 1729 | 64 | 898.75 | 16.69 | 14/187 |

The stock and frozen runs logged small physical reads in the 8K greedy request. Stock also read roughly 1.1 MiB during its sampled request. All four runs had zero process swap and passed the generation headroom threshold. All timed 8K runs of the clean Inkweave build recorded zero physical reads and process swap.

## Historical stock diagnostics

The original short runs used a 1,825-token document for prefill and a separate 79-token prompt with 96 generated tokens. They establish the project's starting point. Use the 8K table in the README for matched workloads.

| Starting Unsloth release profile | Draft guesses | Prefill (tok/s) | Generation (tok/s) |
| --- | ---: | ---: | ---: |
| Static expert offload | 3 | 165.92 | 8.87 |
| Host experts with 3,072 MiB Vulkan cache | 3 | 148.63 | 9.45 |
| Same cache, 50% CPU share for eligible misses | 3 | 146.99 | 8.84 |
| Host experts with 3,200 MiB Vulkan cache | 2 | 151.58 | 11.33 |
| Same 3,200 MiB cache, direct | 0 | 153.34 | 16.24 |

The first four rows average two measurements per rate. The direct row uses one measurement per rate. A separate fully resident 4,096-context diagnostic reached 400.86 tok/s prefill and 24.33 tok/s generation with 194 decode graph splits.
