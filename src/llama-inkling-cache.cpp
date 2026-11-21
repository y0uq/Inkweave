#include "llama-inkling-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <limits>

llama_inkling_cache::llama_inkling_cache(const llama_model & model) : model(model) {
    ggml_backend_dev_t dev = model.dev_layer(0);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    using create_fn = void * (*)(ggml_backend_dev_t);
    const auto create = reinterpret_cast<create_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_vk_expert_updates_create"));
    destroy = reinterpret_cast<decltype(destroy)>(ggml_backend_reg_get_proc_address(reg, "ggml_vk_expert_updates_destroy"));
    poll = reinterpret_cast<decltype(poll)>(ggml_backend_reg_get_proc_address(reg, "ggml_vk_expert_updates_poll"));
    submit = reinterpret_cast<decltype(submit)>(ggml_backend_reg_get_proc_address(reg, "ggml_vk_expert_updates_submit"));
    read = reinterpret_cast<decltype(read)>(ggml_backend_reg_get_proc_address(reg, "ggml_vk_expert_routes_async"));
    const auto host_type = ggml_backend_dev_host_buffer_type(dev);
    if (!host_type || !create || !destroy || !poll || !submit || !read) {
        return;
    }

    const auto & hp = model.hparams;
    for (uint32_t visit = 0; visit < hp.n_layer(); ++visit) {
        width = std::max(width, hp.n_expert_used(visit));
    }
    host.reset(ggml_backend_buft_alloc_buffer(host_type, hp.n_layer() * max_rows * width * sizeof(int32_t)));
    if (!host) {
        return;
    }
    routes = static_cast<int32_t *>(ggml_backend_buffer_get_base(host.get()));
    layers.reserve(hp.n_layer_weights());
    for (uint32_t il = 0; il < hp.n_layer_weights(); ++il) {
        const auto & weights = model.layers[hp.loop_visit(il)];
        auto & l = layers.emplace_back(layer{weights.ffn_gate_up_exps, weights.ffn_down_exps,
                std::vector<float>(hp.n_expert), std::vector<uint8_t>(hp.n_expert), {}});
        for (uint32_t id : model.inkling_expert_admission[il]) {
            l.resident[id] = 1;
        }
        l.initial = l.resident;
    }
    updates = create(dev);
    if (updates) {
        LLAMA_LOG_INFO("inkling adaptive cache: %zu layers, %u route slots, independent transfer queue\n", layers.size(), width);
    }
}

llama_inkling_cache::~llama_inkling_cache() {
    if (updates) {
        destroy(updates);
    }
    if (selections) {
        LLAMA_LOG_INFO("inkling adaptive cache final: tokens=%" PRIu64 " hits=%" PRIu64 "/%" PRIu64
                " static_hits=%" PRIu64 " bad_rows=%" PRIu64 " trades=%" PRIu64 " late=%" PRIu64 " nominal_upload_mib=%.2f\n",
                tokens, hits, selections, static_hits, bad_rows, trades, late, nominal_upload_bytes / 1048576.0);
    }
}

void llama_inkling_cache::collect(ggml_backend_t backend, const std::vector<ggml_tensor *> & tensors, uint32_t rows) {
    captured_rows = 0;
    if (rows && rows <= max_rows && tensors.size() == model.hparams.n_layer() &&
            read(backend, tensors.data(), tensors.size(), rows, width, routes)) {
        captured_rows = rows;
    }
}

void llama_inkling_cache::commit(uint32_t rows) {
    const int status = poll(updates);
    if (status < 0) {
        destroy(updates);
        updates = nullptr;
        pending.clear();
        captured_rows = 0;
        return;
    }
    rows = std::min(rows, captured_rows);
    const uint32_t stride = captured_rows;
    captured_rows = 0;
    // The routes just completed with the previous mapping, even if a new one is ready now.
    const auto & hp = model.hparams;
    bool valid = true;
    for (uint32_t visit = 0; visit < hp.n_layer() && valid; ++visit) {
        const uint32_t used = hp.n_expert_used(visit);
        for (uint32_t row = 0; row < rows; ++row) {
            const int32_t * ids = routes + (visit * stride + row) * width;
            for (uint32_t k = 0; k < used; ++k) {
                valid &= (uint32_t) ids[k] < hp.n_expert && std::find(ids, ids + k, ids[k]) == ids + k;
            }
        }
    }
    if (!valid) {
        if (!bad_rows) {
            LLAMA_LOG_WARN("inkling adaptive cache: invalid router readback, skipping those rows\n");
        }
        bad_rows += rows;
        rows = 0;
    }
    if (rows) {
        constexpr float half_life = 28.0f;
        const float decay = std::exp2(-1.f / half_life);
        std::array<float, max_rows> weights;
        weights[rows - 1] = 1.f;
        for (uint32_t i = rows - 1; i > 0; --i) {
            weights[i - 1] = weights[i] * decay;
        }
        for (auto & l : layers) {
            for (auto & hotness : l.recent) {
                hotness *= weights[0] * decay;
            }
        }
        for (uint32_t visit = 0; visit < hp.n_layer(); ++visit) {
            auto & l = layers[hp.loop_layer(visit)];
            for (uint32_t row = 0; row < rows; ++row) {
                const int32_t * ids = routes + (visit * stride + row) * width;
                for (uint32_t k = 0; k < hp.n_expert_used(visit); ++k) {
                    const uint32_t id = ids[k];
                    l.recent[id] += weights[row];
                    hits += l.resident[id];
                    static_hits += l.initial[id];
                    ++selections;
                }
            }
        }
        tokens += rows;
    }
    if (status > 0) {
        for (const auto & t : pending) {
            auto & l = layers[t.layer];
            l.resident[t.outgoing] = 0;
            l.resident[t.incoming] = 1;
            nominal_upload_bytes += l.gate_up->nb[2] + l.down->nb[2];
            ++trades;
        }
        pending.clear();
    } else if (!pending.empty()) {
        ++late;
    }
    if (tokens >= next_report) {
        LLAMA_LOG_INFO("inkling adaptive cache: tokens=%" PRIu64 " hits=%" PRIu64 "/%" PRIu64
                " static_hits=%" PRIu64 " bad_rows=%" PRIu64 " trades=%" PRIu64 " late=%" PRIu64 " nominal_upload_mib=%.2f\n",
                tokens, hits, selections, static_hits, bad_rows, trades, late, nominal_upload_bytes / 1048576.0);
        next_report = tokens + 32;
    }
    if (!rows || !pending.empty() || tokens < next_trade) {
        return;
    }
    next_trade = tokens + 4;

    std::vector<trade> candidates;
    candidates.reserve(layers.size());
    for (uint32_t il = 0; il < layers.size(); ++il) {
        const auto & l = layers[il];
        uint32_t add = 0;
        uint32_t remove = 0;
        float hot = 0;
        float cold = std::numeric_limits<float>::infinity();
        for (uint32_t id = 0; id < l.recent.size(); ++id) {
            const float recent = l.recent[id];
            if (l.resident[id]) {
                if (recent < cold) {
                    remove = id;
                    cold = recent;
                }
            } else if (recent > hot) {
                add = id;
                hot = recent;
            }
        }
        if (hot >= 2.0f * cold + 2.0f) {
            const size_t bytes = l.gate_up->nb[2] + l.down->nb[2];
            candidates.push_back({il, add, remove, (hot - cold - 1.0) * bytes});
        }
    }
    constexpr size_t max_trades = 4;
    const size_t count = std::min(max_trades, candidates.size());
    if (!count) {
        return;
    }
    std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end(), [](const trade & a, const trade & b) {
        return a.gain > b.gain;
    });
    std::array<const ggml_tensor *, 2 * max_trades> tensors;
    std::array<uint32_t, 2 * max_trades> incoming;
    std::array<uint32_t, 2 * max_trades> outgoing;
    for (size_t i = 0; i < count; ++i) {
        const auto & t = candidates[i];
        const auto & l = layers[t.layer];
        tensors[2 * i] = l.gate_up;
        tensors[2 * i + 1] = l.down;
        incoming[2 * i] = incoming[2 * i + 1] = t.incoming;
        outgoing[2 * i] = outgoing[2 * i + 1] = t.outgoing;
    }
    if (submit(updates, tensors.data(), incoming.data(), outgoing.data(), 2 * count)) {
        pending.assign(candidates.begin(), candidates.begin() + count);
    }
}
