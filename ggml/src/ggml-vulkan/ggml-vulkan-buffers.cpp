#include "ggml-vulkan-common.h"

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>

static void * portable_aligned_malloc(size_t align, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void * ptr = nullptr;
    if (posix_memalign(&ptr, align, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

static void portable_aligned_free(void * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static std::vector<uint32_t> ggml_vk_find_memory_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    return indices;
}

static vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, const std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       void *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    if (size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (size == 0) {
        buf->size = 0;
        return buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    if (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::ExternalMemoryBufferCreateInfo external_memory_bci;
    if (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    const vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    if (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    if (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        try {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } catch (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            return {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        for (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            if (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }
            if (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            if ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                break;
            }
        }
        if (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            return {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        try {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
            buf->allocation_size = size;
            buf->memory_type = memory_type_idx;
        } catch (const vk::SystemError& e) {
        }
    } else {
        for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            const auto & req_flags = *it;

            const std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            if (memory_type_indices.empty()) {
                continue;
            }

            bool done = false;

            for (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                try {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->allocation_size = mem_req.size;
                    buf->memory_type = *mtype_it;
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = true;
                    break;
                } catch (const vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    if (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        throw e;
                    }
                }
            }

            if (done) {
                break;
            }
        }
    }

    if (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        throw vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    if (import_ptr) {
        buf->ptr = import_ptr;
    } else {
        if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    if (device->buffer_device_address) {
        const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    return buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    try {
        return ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    try {
        if (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } else if (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } else if (!device->rebar || device->disable_host_visible_vidmem) {
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } else {
            // use rebar if available, otherwise fallback to device only visible memory
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    return buf;
}

void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }

    if (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
static size_t ggml_vk_host_allocation_find(const void * ptr) {
    for (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        const uint8_t * base = (const uint8_t *) vk_instance.host_allocations[i].base;
        if (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            return i;
        }
    }
    return SIZE_MAX;
}

void * ggml_vk_host_malloc(size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    if (vk_instance.multi_device) {
        const size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        const size_t alloc_size = (size + align - 1) / align * align;

        void * ptr = portable_aligned_malloc(align, alloc_size);
        if (ptr == nullptr) {
            return nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_size;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        return ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_buffer buf;
    try {
        buf = ggml_vk_create_buffer(dev0, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
    } catch (vk::SystemError&) {
        return nullptr;
    }

    if (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_buffer(buf);
        return nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->ptr;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    return buf->ptr;
}

void ggml_vk_host_free(void * ptr) {
    if (ptr == nullptr) {
        return;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        const size_t index = ggml_vk_host_allocation_find(ptr);
        if (index == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            return;
        }
        entry = std::move(vk_instance.host_allocations[index]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + index);
    }

    for (auto & view : entry.views) {
        ggml_vk_destroy_buffer(view);
    }
    if (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

void ggml_vk_host_get(const vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    const size_t index = ggml_vk_host_allocation_find(ptr);
    if (index == SIZE_MAX) {
        return;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[index];

    if (entry.views[device->idx] == nullptr) {
        const uint32_t bit = 1u << device->idx;
        if (!vk_instance.multi_device || !vk_instance.device_host_buffer_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            return;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_buffer_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        if (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            return;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (const uint8_t *) ptr - (const uint8_t *) entry.base;
}

void ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    if (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

void ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    if (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static vk_context ggml_vk_begin_sync_ctx(vk_device & device, vk_command_pool & pool) {
    vk_context subctx = ggml_vk_create_temporary_context(pool);
    ggml_vk_ctx_begin(device, subctx);
    return subctx;
}

static void ggml_vk_submit_sync(vk_device & device, vk_context & subctx) {
    ggml_vk_ctx_end(subctx);

    for (auto & cpy : subctx->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }

    for (auto & mset : subctx->memsets) {
        memset(mset.dst, mset.val, mset.n);
    }

    ggml_vk_submit(subctx, device->fence);
    VK_CHECK(device->device.waitForFences({ device->fence }, true, UINT64_MAX), "vk_submit_sync waitForFences", device);
    device->device.resetFences({ device->fence });
    ggml_vk_queue_command_pools_cleanup(device);

    for (auto & cpy : subctx->out_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }
}

static void ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, const ggml_tensor * tensor, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    const uint64_t ne0 = tensor->ne[0];
    const uint64_t ne1 = tensor->ne[1];
    const uint64_t ne2 = tensor->ne[2];
    const uint64_t ne3 = tensor->ne[3];
    const uint64_t nb0 = tensor->nb[0];
    const uint64_t nb1 = tensor->nb[1];
    const uint64_t nb2 = tensor->nb[2];
    const uint64_t nb3 = tensor->nb[3];
    const ggml_type type = tensor->type;
    const uint64_t ts = ggml_type_size(type);
    const uint64_t bs = ggml_blck_size(type);

    const uint64_t dstnb0 = ts;
    const uint64_t dstnb1 = dstnb0*(ne0/bs);
    const uint64_t dstnb2 = dstnb1*ne1;
    const uint64_t dstnb3 = dstnb2*ne2;

    const uint64_t ne = ggml_nelements(tensor);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        for (uint64_t i3 = 0; i3 < ne3; i3++) {
            for (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                if (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } else {
                    for (uint64_t i1 = 0; i1 < ne1; i1++) {
                        if (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } else {
                            const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            const uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            for (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }

    if (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    const uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    for (uint64_t i3 = 0; i3 < ne3; i3++) {
        for (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            if (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } else {
                for (uint64_t i1 = 0; i1 < ne1; i1++) {
                    if (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } else {
                        const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        const uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        for (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (const uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Staging buffer required
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    if (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    if (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (const uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    return true;
}

bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    return ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

void ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        if (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (const uint8_t *) src + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_begin_sync_ctx(dst->device, dst->device->transfer_queue->cmd_pool);
        bool ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_submit_sync(dst->device, subctx);
    }
}

void ggml_vk_buffer_write(vk_buffer& dst, size_t offset, const void * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Fall back to staging buffer
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    if (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } else {
        staging_slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    if (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    return true;
}

static bool ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t size, bool sync_staging = false) {
    return ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

static bool ggml_vk_buffer_host_read_direct(const vk_buffer & buf) {
    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // UMA: uncached host-visible memory is write-combined, CPU reads are slow (except Nvidia, based on benchmarks)
    const bool slow_host_read = buf->device->vendor_id != VK_VENDOR_ID_NVIDIA &&
                                !(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    return (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) && buf->device->uma && !slow_host_read;
}

void ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    if (ggml_vk_buffer_host_read_direct(src)) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_begin_sync_ctx(src->device, src->device->compute_queue->cmd_pool);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_submit_sync(src->device, subctx);

        if (width == spitch && width == dpitch) {
            memcpy(dst, (const uint8_t *) src->ptr + offset, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_begin_sync_ctx(src->device, src->device->transfer_queue->cmd_pool);
        bool ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_submit_sync(src->device, subctx);
    }
}

void ggml_vk_buffer_read(vk_buffer& src, size_t offset, void * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

void ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

void ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_begin_sync_ctx(src->device, src->device->transfer_queue->cmd_pool);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_submit_sync(src->device, subctx);
    } else {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

void ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        return;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

void ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + offset, c, size);
        return;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_begin_sync_ctx(dst->device, dst->device->transfer_queue->cmd_pool);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_submit_sync(dst->device, subctx);
}

static size_t ggml_vk_staging_size_n(const vk_device & device, const std::vector<vk_buffer_region> & regions) {
    size_t total = 0;
    size_t max   = 0;
    for (const auto & r : regions) {
        vk_buffer pinned;
        size_t pinned_offset;
        ggml_vk_host_get(device, r.host, pinned, pinned_offset);
        if (pinned != nullptr) {
            continue;
        }
        total += r.size;
        max = std::max(max, r.size);
    }
    return std::max(max, std::min(total, (size_t) device->suballocation_block_size));
}

// returns number of regions recorded, stops when staging is full
size_t ggml_vk_buffer_write_n_async(vk_context& subctx, const vk_buffer_region * regions, size_t n, vk_buffer * staging, size_t * staging_offset) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_n_async(" << n << ")");
    ggml_vk_sync_buffers(nullptr, subctx);

    size_t i = 0;
    for (; i < n; i++) {
        const vk_buffer_region & r = regions[i];

        vk_buffer pinned = nullptr;
        size_t pinned_offset = 0;
        ggml_vk_host_get(r.buf->device, r.host, pinned, pinned_offset);

        if (pinned != nullptr) {
            subctx->s->buffer->buf.copyBuffer(pinned->buffer, r.buf->buffer, { vk::BufferCopy{ pinned_offset, r.offset, r.size } });
            continue;
        }

        if (staging == nullptr || *staging_offset + r.size > (*staging)->size) {
            break;
        }

        subctx->s->buffer->buf.copyBuffer((*staging)->buffer, r.buf->buffer, { vk::BufferCopy{ *staging_offset, r.offset, r.size } });
        deferred_memcpy((uint8_t *) (*staging)->ptr + *staging_offset, r.host, r.size, &subctx->in_memcpys);
        *staging_offset += r.size;
    }
    return i;
}

// returns number of regions recorded, stops when staging is full
size_t ggml_vk_buffer_read_n_async(vk_context& subctx, const vk_buffer_region * regions, size_t n, vk_buffer * staging, size_t * staging_offset) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_n_async(" << n << ")");
    ggml_vk_sync_buffers(nullptr, subctx);

    size_t i = 0;
    for (; i < n; i++) {
        const vk_buffer_region & r = regions[i];

        vk_buffer pinned = nullptr;
        size_t pinned_offset = 0;
        ggml_vk_host_get(r.buf->device, r.host, pinned, pinned_offset);

        if (pinned != nullptr) {
            subctx->s->buffer->buf.copyBuffer(r.buf->buffer, pinned->buffer, { vk::BufferCopy{ r.offset, pinned_offset, r.size } });
            continue;
        }

        if (staging == nullptr || *staging_offset + r.size > (*staging)->size) {
            break;
        }

        subctx->s->buffer->buf.copyBuffer(r.buf->buffer, (*staging)->buffer, { vk::BufferCopy{ r.offset, *staging_offset, r.size } });
        deferred_memcpy(r.host, (const uint8_t *) (*staging)->ptr + *staging_offset, r.size, &subctx->out_memcpys);
        *staging_offset += r.size;
    }
    return i;
}

void ggml_vk_buffer_copy_n_async(vk_context& subctx, const vk_buffer_copy_region * regions, size_t n) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_n_async(" << n << ")");
    ggml_vk_sync_buffers(nullptr, subctx);

    for (size_t i = 0; i < n; i++) {
        const vk_buffer_copy_region & r = regions[i];
        GGML_ASSERT(r.src->device == r.dst->device);
        subctx->s->buffer->buf.copyBuffer(r.src->buffer, r.dst->buffer, { vk::BufferCopy{ r.src_offset, r.dst_offset, r.size } });
    }
}

void ggml_vk_buffer_write_n(const vk_buffer_region * regions, size_t n) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_n(" << n << ")");
    std::vector<vk_buffer_region> pending;

    for (size_t i = 0; i < n; i++) {
        const vk_buffer_region & r = regions[i];
        if (r.size == 0) {
            continue;
        }
        if (r.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            GGML_ASSERT(r.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
            memcpy((uint8_t *) r.buf->ptr + r.offset, r.host, r.size);
            continue;
        }
        GGML_ASSERT(pending.empty() || r.buf->device == pending[0].buf->device);
        pending.push_back(r);
    }

    if (pending.empty()) {
        return;
    }

    vk_device device = pending[0].buf->device;
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    const size_t staging_size = ggml_vk_staging_size_n(device, pending);
    if (staging_size > 0) {
        ggml_vk_ensure_sync_staging_buffer(device, staging_size);
    }

    for (size_t i = 0; i < pending.size();) {
        vk_context subctx = ggml_vk_begin_sync_ctx(device, device->transfer_queue->cmd_pool);
        size_t staging_offset = 0;
        const size_t recorded = ggml_vk_buffer_write_n_async(subctx, pending.data() + i, pending.size() - i, &device->sync_staging, &staging_offset);
        GGML_ASSERT(recorded > 0);
        ggml_vk_submit_sync(device, subctx);
        i += recorded;
    }
}

void ggml_vk_buffer_read_n(const vk_buffer_region * regions, size_t n) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_n(" << n << ")");
    std::vector<vk_buffer_region> direct;
    std::vector<vk_buffer_region> pending;

    for (size_t i = 0; i < n; i++) {
        const vk_buffer_region & r = regions[i];
        if (r.size == 0) {
            continue;
        }
        GGML_ASSERT((direct.empty() && pending.empty()) || r.buf->device == (direct.empty() ? pending[0] : direct[0]).buf->device);
        if (ggml_vk_buffer_host_read_direct(r.buf)) {
            GGML_ASSERT(r.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
            direct.push_back(r);
        } else {
            pending.push_back(r);
        }
    }

    if (direct.empty() && pending.empty()) {
        return;
    }

    vk_device device = (direct.empty() ? pending[0] : direct[0]).buf->device;
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    if (!direct.empty()) {
        vk_context subctx = ggml_vk_begin_sync_ctx(device, device->compute_queue->cmd_pool);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_submit_sync(device, subctx);

        for (const auto & r : direct) {
            memcpy(r.host, (const uint8_t *) r.buf->ptr + r.offset, r.size);
        }
    }

    if (pending.empty()) {
        return;
    }

    const size_t staging_size = ggml_vk_staging_size_n(device, pending);
    if (staging_size > 0) {
        ggml_vk_ensure_sync_staging_buffer(device, staging_size);
    }

    for (size_t i = 0; i < pending.size();) {
        vk_context subctx = ggml_vk_begin_sync_ctx(device, device->transfer_queue->cmd_pool);
        size_t staging_offset = 0;
        const size_t recorded = ggml_vk_buffer_read_n_async(subctx, pending.data() + i, pending.size() - i, &device->sync_staging, &staging_offset);
        GGML_ASSERT(recorded > 0);
        ggml_vk_submit_sync(device, subctx);
        i += recorded;
    }
}

void ggml_vk_buffer_copy_n(const vk_buffer_copy_region * regions, size_t n) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_n(" << n << ")");
    std::vector<vk_buffer_copy_region> same_device;

    for (size_t i = 0; i < n; i++) {
        vk_buffer_copy_region r = regions[i];
        if (r.size == 0) {
            continue;
        }
        if (r.src->device != r.dst->device) {
            ggml_vk_buffer_copy(r.dst, r.dst_offset, r.src, r.src_offset, r.size);
            continue;
        }
        GGML_ASSERT(same_device.empty() || r.dst->device == same_device[0].dst->device);
        same_device.push_back(std::move(r));
    }

    if (same_device.empty()) {
        return;
    }

    vk_device device = same_device[0].dst->device;
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    vk_context subctx = ggml_vk_begin_sync_ctx(device, device->transfer_queue->cmd_pool);
    ggml_vk_buffer_copy_n_async(subctx, same_device.data(), same_device.size());
    ggml_vk_submit_sync(device, subctx);
}

struct vk_expert_page_bank {
    struct page_entry {
        int32_t slot = -1;
        uint16_t refs = 0;
    };

    vk_buffer inactive;
    size_t base = 0;
    size_t offset = 0;
    size_t stride = 0;
    size_t page = 0;
    size_t capacity = 0;
    size_t experts = 0;
    std::vector<page_entry> active_pages;
    std::vector<page_entry> inactive_pages;

    size_t first_page(size_t id) const { return (offset + id * stride) / page; }
    size_t end_page(size_t id) const { return CEIL_DIV(offset + (id + 1) * stride, page); }
};

static void ggml_vk_bind_sparse(vk_queue & target, const vk::BindSparseInfo & info, vk::Fence fence = {}) {
    auto & queue = *target.handle;
    std::unique_lock<std::mutex> submit_guard;
    if (queue.device_submit_mutex) {
        submit_guard = std::unique_lock<std::mutex>(*queue.device_submit_mutex);
    }
    std::lock_guard<vk_queue_handle> queue_guard(queue);
    queue.queue.bindSparse(info, fence);
}

struct vk_expert_update {
    struct bank_update {
        vk_buffer host;
        vk_expert_alias * alias;
        std::vector<vk_expert_page_bank::page_entry> pages;
        std::vector<vk::SparseMemoryBind> bindings;
    };

    vk_device device;
    vk_queue & queue;
    vk::UniqueCommandPool pool;
    vk::CommandBuffer cmd;
    vk::UniqueSemaphore uploaded;
    vk::UniqueSemaphore mapped;
    uint64_t epoch = 0;
    bool enabled = true;
    std::vector<bank_update> pending;

    vk_expert_update(vk_device device, vk_queue & queue) : device(std::move(device)), queue(queue) {
        auto & d = *this->device;
        const auto family = queue.queue_family_index;
        pool = d.device.createCommandPoolUnique({ vk::CommandPoolCreateFlagBits::eTransient, family });
        cmd = d.device.allocateCommandBuffers({ *pool, vk::CommandBufferLevel::ePrimary, 1 }).front();
        const vk::SemaphoreTypeCreateInfo type(vk::SemaphoreType::eTimeline, 0);
        vk::SemaphoreCreateInfo info;
        info.setPNext(&type);
        uploaded = d.device.createSemaphoreUnique(info);
        mapped = d.device.createSemaphoreUnique(info);
    }

    ~vk_expert_update() {
        if (epoch == 0) {
            return;
        }
        // Both queues can reference the update semaphores until the context is closed.
        try {
            for (auto * q : { &queue, device->compute_queue.get() }) {
                std::lock_guard<vk_queue_handle> guard(*q->handle);
                q->handle->queue.waitIdle();
            }
        } catch (const vk::SystemError & error) {
            GGML_LOG_WARN("ggml_vulkan: draining expert updates: %s\n", error.what());
        }
    }
};

void * ggml_vk_expert_updates_create_for_device(vk_device device) {
    auto * queue = device->expert_transfer_queue ? device->expert_transfer_queue.get() : device->transfer_queue.get();
    if (!device->inkling_sparse_experts || queue->handle == device->compute_queue->handle ||
            queue->queue_family_index != device->compute_queue->queue_family_index) {
        return nullptr;
    }
    try {
        return new vk_expert_update(std::move(device), *queue);
    } catch (const vk::SystemError & error) {
        GGML_LOG_WARN("ggml_vulkan: creating expert updates: %s\n", error.what());
        return nullptr;
    }
}

void ggml_vk_expert_updates_destroy(void * handle) {
    delete static_cast<vk_expert_update *>(handle);
}

// The target context calls this after its graph has completed. A late upload
// or mapping keeps the previous aliases available for another graph.
int ggml_vk_expert_updates_poll(void * handle) {
    auto & tx = *static_cast<vk_expert_update *>(handle);
    if (!tx.enabled) {
        return -1;
    }
    if (tx.pending.empty()) {
        return 0;
    }
    auto & d = *tx.device;
    try {
        if (d.device.getSemaphoreCounterValue(*tx.mapped) < tx.epoch ||
                d.device.getSemaphoreCounterValue(*tx.uploaded) < tx.epoch) {
            return 0;
        }

        const vk::Semaphore ready[] = { *tx.uploaded, *tx.mapped };
        const uint64_t values[] = { tx.epoch, tx.epoch };
        const vk::PipelineStageFlags stages[] = { vk::PipelineStageFlagBits::eAllCommands,
                                                vk::PipelineStageFlagBits::eAllCommands };
        const vk::TimelineSemaphoreSubmitInfo time(2, values, 0, nullptr);
        vk::SubmitInfo wait(2, ready, stages, 0, nullptr, 0, nullptr);
        wait.setPNext(&time);
        d.compute_queue->handle->submit(wait, {});
        for (auto & update : tx.pending) {
            auto & bank = *update.alias->bank;
            update.alias->buffer.swap(bank.inactive);
            bank.inactive_pages = std::move(bank.active_pages);
            bank.active_pages = std::move(update.pages);
        }
        tx.pending.clear();
        return 1;
    } catch (const vk::SystemError & error) {
        GGML_LOG_WARN("ggml_vulkan: publishing expert updates: %s\n", error.what());
        tx.enabled = false;
        return -1;
    }
}

static bool ggml_vk_stage_expert(vk_expert_update & tx, const ggml_tensor * tensor, uint32_t add, uint32_t remove,
        std::vector<vk_buffer_copy_region> & copies) {
    vk_buffer host;
    size_t offset;
    ggml_vk_host_get(tx.device, tensor->data, host, offset);
    if (!host) {
        return false;
    }
    const auto it = host->expert_aliases.find(offset);
    if (it == host->expert_aliases.end() || !it->second.bank) {
        return false;
    }
    auto & alias = it->second;
    auto & bank = *alias.bank;
    if (add == remove || add >= bank.experts || remove >= bank.experts) {
        return false;
    }
    for (const auto & update : tx.pending) {
        if (update.alias == &alias) {
            return false;
        }
    }
    auto pages = bank.active_pages;
    const size_t page = bank.page;
    std::vector<uint8_t> occupied(bank.capacity);
    for (const auto & entry : pages) {
        if (entry.slot >= 0) {
            occupied[entry.slot] = 1;
        }
    }
    for (size_t i = bank.first_page(remove); i < bank.end_page(remove); ++i) {
        if (pages[i].refs == 0) {
            return false;
        }
        --pages[i].refs;
    }
    size_t free_slot = 0;
    const auto local = alias.buffer->expert_pages;
    size_t last_copy = copies.size();
    for (size_t i = bank.first_page(add); i < bank.end_page(add); ++i) {
        auto & entry = pages[i];
        ++entry.refs;
        if (entry.slot >= 0) {
            continue;
        }
        while (free_slot < occupied.size() && occupied[free_slot]) {
            ++free_slot;
        }
        if (free_slot == occupied.size()) {
            return false;
        }
        entry.slot = free_slot++;
        const size_t dst = entry.slot * page;
        const size_t src = bank.base + i * page;
        const size_t bytes = std::min(page, host->size > src ? host->size - src : 0) / 4 * 4;
        if (!bytes) {
            return false;
        }
        if (copies.size() > last_copy && copies.back().dst_offset + copies.back().size == dst &&
                copies.back().src_offset + copies.back().size == src) {
            copies.back().size += bytes;
        } else {
            copies.push_back({ local, dst, host, src, bytes });
        }
    }

    std::vector<vk::SparseMemoryBind> binds;
    for (size_t i = 0; i < pages.size(); ++i) {
        auto & entry = pages[i];
        if (!entry.refs) {
            entry.slot = -1;
        }
        if (entry.slot == bank.inactive_pages[i].slot) {
            continue;
        }
        const auto mem = entry.slot >= 0 ? local->device_memory : host->device_memory;
        const size_t off = entry.slot >= 0 ? entry.slot * page : bank.base + i * page;
        if (!binds.empty() && binds.back().memory == mem && binds.back().memoryOffset + binds.back().size == off &&
                binds.back().resourceOffset + binds.back().size == i * page) {
            binds.back().size += page;
        } else {
            binds.emplace_back(i * page, page, mem, off);
        }
    }
    tx.pending.push_back({ std::move(host), &alias, std::move(pages), std::move(binds) });
    return true;
}

bool ggml_vk_expert_updates_submit(void * handle, const ggml_tensor * const * tensors,
        const uint32_t * incoming, const uint32_t * outgoing, size_t count) {
    auto & tx = *static_cast<vk_expert_update *>(handle);
    if (!tx.enabled || !tx.pending.empty() || !count) {
        return false;
    }
    std::vector<vk_buffer_copy_region> copies;
    for (size_t i = 0; i < count; ++i) {
        if (!ggml_vk_stage_expert(tx, tensors[i], incoming[i], outgoing[i], copies)) {
            tx.pending.clear();
            return false;
        }
    }
    auto & d = *tx.device;
    try {
        d.device.resetCommandPool(*tx.pool);
        tx.cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        for (const auto & copy : copies) {
            tx.cmd.copyBuffer(copy.src->buffer, copy.dst->buffer, { vk::BufferCopy(copy.src_offset, copy.dst_offset, copy.size) });
        }
        tx.cmd.end();

        ++tx.epoch;
        const vk::TimelineSemaphoreSubmitInfo time(0, nullptr, 1, &tx.epoch);
        vk::SubmitInfo upload(0, nullptr, nullptr, 1, &tx.cmd, 1, &*tx.uploaded);
        upload.setPNext(&time);
        tx.queue.handle->submit(upload, {});

        std::vector<vk::SparseBufferMemoryBindInfo> bindings;
        for (const auto & update : tx.pending) {
            if (!update.bindings.empty()) {
                bindings.emplace_back(update.alias->bank->inactive->buffer, update.bindings);
            }
        }
        vk::BindSparseInfo map;
        map.setPNext(&time).setBufferBinds(bindings).setSignalSemaphores(*tx.mapped);
        ggml_vk_bind_sparse(tx.queue, map);
        return true;
    } catch (const vk::SystemError & error) {
        GGML_LOG_WARN("ggml_vulkan: queueing expert updates: %s\n", error.what());
        tx.enabled = false;
        return false;
    }
}

// Preserve the original offsets as seen by indexed GEMMs. Vulkan translates the
// virtual pages into either their pinned host backing or a read-only VRAM copy.
size_t ggml_vk_cache_expert_pages(vk_device & device, const ggml_tensor * tensor,
        const uint32_t * expert_ids, size_t count, size_t budget, size_t * admitted, bool adaptive) {
    *admitted = 0;
    vk_buffer host;
    size_t host_offset;
    ggml_vk_host_get(device, tensor->data, host, host_offset);
    if (!device->inkling_sparse_experts || !host || !host->ptr ||
        host->memory_type == UINT32_MAX || host->expert_aliases.count(host_offset)) {
        return 0;
    }

    std::lock_guard<std::recursive_mutex> device_guard(device->mutex);
    auto alias = std::make_shared<vk_buffer_struct>();
    alias->device = device;
    // Leave room for the tensor offset within its first sparse page.
    constexpr size_t max_page = 65536;
    const auto usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc;
    const auto flags = vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseAliased;
    try {
        alias->buffer = device->device.createBuffer({ flags, ggml_nbytes(tensor) + max_page, usage });
        alias->size = ggml_nbytes(tensor) + max_page;
        const auto req = device->device.getBufferMemoryRequirements(alias->buffer);
        const size_t page = req.alignment;
        if (page > max_page || page < 4 || req.size > device->max_buffer_size ||
                !(req.memoryTypeBits & (1u << host->memory_type))) {
            return 0;
        }

        const size_t base = host_offset / page * page;
        const size_t local_offset = host_offset - base;
        const size_t n_pages = CEIL_DIV(req.size, page);
        const size_t used_pages = CEIL_DIV(local_offset + ggml_nbytes(tensor), page);
        if (base + used_pages * page > host->allocation_size) {
            return 0;
        }

        std::vector<bool> resident(used_pages);
        size_t resident_pages = 0;
        size_t chosen = 0;
        for (; chosen < count; ++chosen) {
            const size_t expert = expert_ids[chosen];
            if (expert >= (size_t) tensor->ne[2]) {
                break;
            }
            const size_t begin = (local_offset + expert * tensor->nb[2]) / page;
            const size_t end = CEIL_DIV(local_offset + (expert + 1) * tensor->nb[2], page);
            size_t needed = 0;
            for (size_t j = begin; j < end; ++j) {
                needed += !resident[j];
            }
            if ((resident_pages + needed) * page > budget) {
                break;
            }
            resident_pages += needed;
            for (size_t j = begin; j < end; ++j) {
                resident[j] = true;
            }
        }
        if (!resident_pages) {
            return 0;
        }

        // Sparse binding requires the whole virtual resource to be mapped. The
        // last padding pages alias a valid host page and are never addressed.
        const size_t max_pages_per_expert = CEIL_DIV(tensor->nb[2], page) + 1;
        const size_t capacity = adaptive ? (chosen + 1) * max_pages_per_expert : resident_pages;
        if (capacity > budget / page) {
            return 0;
        }
        auto local = ggml_vk_create_buffer_check(device, capacity * page, vk::MemoryPropertyFlagBits::eDeviceLocal);
        if (!local || !(req.memoryTypeBits & (1u << local->memory_type))) {
            return 0;
        }
        alias->expert_pages = local;
        std::vector<vk::SparseMemoryBind> bindings;
        std::vector<vk_buffer_copy_region> copies;
        std::shared_ptr<vk_expert_page_bank> bank;
        if (adaptive) {
            bank = std::make_shared<vk_expert_page_bank>();
            bank->inactive = std::make_shared<vk_buffer_struct>();
            bank->inactive->device = device;
            bank->inactive->buffer = device->device.createBuffer({ flags, alias->size, usage });
            bank->inactive->size = alias->size;
            bank->inactive->expert_pages = local;
            bank->base = base;
            bank->offset = local_offset;
            bank->stride = tensor->nb[2];
            bank->page = page;
            bank->capacity = capacity;
            bank->experts = tensor->ne[2];
            bank->active_pages.resize(used_pages);
            for (size_t i = 0; i < chosen; ++i) {
                for (size_t j = bank->first_page(expert_ids[i]); j < bank->end_page(expert_ids[i]); ++j) {
                    ++bank->active_pages[j].refs;
                }
            }
        }
        size_t dst_page = 0;
        for (size_t j = 0; j < used_pages;) {
            const size_t first = j;
            const bool in_vram = resident[j++];
            while (j < used_pages && resident[j] == in_vram) {
                ++j;
            }
            const size_t bytes = (j - first) * page;
            if (in_vram) {
                const size_t source = base + first * page;
                const size_t copy_bytes = std::min(bytes, host->size > source ? host->size - source : 0);
                copies.push_back({ local, dst_page * page, host, source, copy_bytes / 4 * 4 });
                bindings.emplace_back(first * page, bytes, local->device_memory, dst_page * page);
                if (bank) {
                    for (size_t k = first; k < j; ++k) {
                        bank->active_pages[k].slot = dst_page + k - first;
                    }
                }
                dst_page += j - first;
            } else {
                bindings.emplace_back(first * page, bytes, host->device_memory, base + first * page);
            }
        }
        for (size_t j = used_pages; j < n_pages; ++j) {
            bindings.emplace_back(j * page, page, host->device_memory, base);
        }

        // The existing copy helper records one transfer batch and waits for its
        // completion. The sparse mapping is published only after its own fence.
        auto transfer = ggml_vk_begin_sync_ctx(device, device->compute_queue->cmd_pool);
        ggml_vk_buffer_copy_n_async(transfer, copies.data(), copies.size());
        const vk::MemoryBarrier ready(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
        transfer->s->buffer->buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader, {}, ready, {}, {});
        ggml_vk_submit_sync(device, transfer);
        std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
        buffer_binds.emplace_back(alias->buffer, bindings);
        if (bank) {
            buffer_binds.emplace_back(bank->inactive->buffer, bindings);
            bank->inactive_pages = bank->active_pages;
        }
        vk::BindSparseInfo bind_info;
        bind_info.setBufferBinds(buffer_binds);
        auto fence = device->device.createFenceUnique({});
        ggml_vk_bind_sparse(*device->compute_queue, bind_info, *fence);
        const auto result = device->device.waitForFences(*fence, VK_TRUE, UINT64_MAX);
        if (result != vk::Result::eSuccess) {
            return 0;
        }
        host->expert_aliases.emplace(host_offset, vk_expert_alias{std::move(alias), local_offset, std::move(bank)});
        *admitted = chosen;
        return capacity * page;
    } catch (const vk::SystemError & error) {
        GGML_LOG_WARN("ggml_vulkan: expert page cache for %s: %s\n", tensor->name, error.what());
        return 0;
    }
}

ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ ggml_backend_vk_buffer_set_tensor_n,
    /* .get_tensor_n    = */ ggml_backend_vk_buffer_get_tensor_n,
    /* .cpy_tensor_n    = */ ggml_backend_vk_buffer_cpy_tensor_n,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, void * ptr, size_t size) {
    if (!device->external_memory_host) {
        return {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    if (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }
    if (size & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }

    const vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    try {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    return buf;
}

