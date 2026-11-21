#pragma once

#include "ggml-cpp.h"

#include <memory>
#include <vector>

struct llama_model;

class llama_inkling_cache {
public:
    static constexpr uint32_t max_rows = 8;

    explicit llama_inkling_cache(const llama_model & model);
    ~llama_inkling_cache();

    bool enabled() const { return updates != nullptr; }
    void collect(ggml_backend_t backend, const std::vector<ggml_tensor *> & routes, uint32_t rows);
    void commit(uint32_t rows);

private:
    struct layer {
        const ggml_tensor * gate_up;
        const ggml_tensor * down;
        std::vector<float> recent;
        std::vector<uint8_t> resident;
        std::vector<uint8_t> initial;
    };

    struct trade {
        uint32_t layer;
        uint32_t incoming;
        uint32_t outgoing;
        double gain;
    };

    const llama_model & model;
    std::vector<layer> layers;
    std::vector<trade> pending;
    ggml_backend_buffer_ptr host;
    int32_t * routes = nullptr;
    uint32_t captured_rows = 0;
    uint32_t width = 0;
    uint64_t tokens = 0;
    uint64_t next_trade = 0;
    uint64_t next_report = 32;
    uint64_t hits = 0;
    uint64_t static_hits = 0;
    uint64_t bad_rows = 0;
    uint64_t selections = 0;
    uint64_t trades = 0;
    uint64_t late = 0;
    uint64_t nominal_upload_bytes = 0;
    void * updates = nullptr;

    void (*destroy)(void *) = nullptr;
    int (*poll)(void *) = nullptr;
    bool (*submit)(void *, const ggml_tensor * const *, const uint32_t *, const uint32_t *, size_t) = nullptr;
    bool (*read)(ggml_backend_t, ggml_tensor * const *, size_t, size_t, size_t, int32_t *) = nullptr;
};
