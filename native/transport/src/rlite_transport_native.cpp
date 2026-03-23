#include "rlite_transport/native_api.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include "rdma/fabric.h"
#include "rdma/fi_cm.h"
#include "rdma/fi_domain.h"
#include "rdma/fi_endpoint.h"
#include "rdma/fi_eq.h"
#include "rdma/fi_errno.h"
#include "rdma/fi_rma.h"

#include "gdrapi.h"

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#define RLITE_HAS_CUDA_HEADERS 1
#else
#define RLITE_HAS_CUDA_HEADERS 0
typedef int cudaError_t;
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};
static constexpr cudaError_t cudaSuccess = 0;
static constexpr unsigned int cudaIpcMemLazyEnablePeerAccess = 1U;
struct cudaIpcMemHandle_t {
    char reserved[64];
};
#endif

#ifndef FI_HMEM_SYSTEM
#define FI_HMEM_SYSTEM 0
#endif

#ifndef FI_HMEM_CUDA
#define FI_HMEM_CUDA 2
#endif

#ifndef FI_HMEM_DEVICE_ONLY
#define FI_HMEM_DEVICE_ONLY 0
#endif

#ifndef FI_MR_HMEM
#define FI_MR_HMEM 0
#endif

namespace {

constexpr uint64_t kDefaultChunkBytes = 16ULL * 1024ULL * 1024ULL;

uint64_t stable_key(int32_t rank, const std::string &name) {
    std::hash<std::string> hasher;
    return static_cast<uint64_t>(hasher(std::to_string(rank) + ":" + name));
}

void copy_string(char *dst, size_t dst_size, const char *src) {
    if (dst == nullptr || dst_size == 0) {
        return;
    }
    if (src == nullptr) {
        dst[0] = '\0';
        return;
    }
    std::snprintf(dst, dst_size, "%s", src);
}

void set_error(char *buffer, size_t buffer_size, const std::string &message) {
    if (buffer == nullptr || buffer_size == 0) {
        return;
    }
    std::snprintf(buffer, buffer_size, "%s", message.c_str());
}

struct LibfabricApi {
#if defined(__linux__)
    void *handle = nullptr;
#endif
    decltype(&fi_getinfo) getinfo = nullptr;
    decltype(&fi_allocinfo) allocinfo = nullptr;
    decltype(&fi_freeinfo) freeinfo = nullptr;
    decltype(&fi_fabric) fabric = nullptr;
    decltype(&fi_domain) domain = nullptr;
    decltype(&fi_endpoint) endpoint = nullptr;
    decltype(&fi_cq_open) cq_open = nullptr;
    decltype(&fi_av_open) av_open = nullptr;
    decltype(&fi_ep_bind) ep_bind = nullptr;
    decltype(&fi_enable) enable = nullptr;
    decltype(&fi_getname) getname = nullptr;
    decltype(&fi_mr_regattr) mr_regattr = nullptr;
    decltype(&fi_av_insert) av_insert = nullptr;
    decltype(&fi_write) write = nullptr;
    decltype(&fi_read) read = nullptr;
    decltype(&fi_cq_read) cq_read = nullptr;
    decltype(&fi_cq_readerr) cq_readerr = nullptr;
    decltype(&fi_strerror) strerror_fn = nullptr;
    bool available = false;

    bool load() {
#if !defined(__linux__)
        return false;
#else
        const char *names[] = {"libfabric.so.1", "libfabric.so", nullptr};
        for (const char **name = names; *name != nullptr; ++name) {
            handle = dlopen(*name, RTLD_LAZY | RTLD_LOCAL);
            if (handle != nullptr) {
                break;
            }
        }
        if (handle == nullptr) {
            return false;
        }
#define RLITE_LOAD_FI(symbol_name, target)                                                           \
    target = reinterpret_cast<decltype(target)>(dlsym(handle, symbol_name));                         \
    if (target == nullptr) {                                                                         \
        return false;                                                                                \
    }
        RLITE_LOAD_FI("fi_getinfo", getinfo)
        RLITE_LOAD_FI("fi_allocinfo", allocinfo)
        RLITE_LOAD_FI("fi_freeinfo", freeinfo)
        RLITE_LOAD_FI("fi_fabric", fabric)
        RLITE_LOAD_FI("fi_domain", domain)
        RLITE_LOAD_FI("fi_endpoint", endpoint)
        RLITE_LOAD_FI("fi_cq_open", cq_open)
        RLITE_LOAD_FI("fi_av_open", av_open)
        RLITE_LOAD_FI("fi_ep_bind", ep_bind)
        RLITE_LOAD_FI("fi_enable", enable)
        RLITE_LOAD_FI("fi_getname", getname)
        RLITE_LOAD_FI("fi_mr_regattr", mr_regattr)
        RLITE_LOAD_FI("fi_av_insert", av_insert)
        RLITE_LOAD_FI("fi_write", write)
        RLITE_LOAD_FI("fi_read", read)
        RLITE_LOAD_FI("fi_cq_read", cq_read)
        RLITE_LOAD_FI("fi_cq_readerr", cq_readerr)
        RLITE_LOAD_FI("fi_strerror", strerror_fn)
#undef RLITE_LOAD_FI
        available = true;
        return true;
#endif
    }

    std::string strerror_value(int error_code) const {
        if (!available || strerror_fn == nullptr) {
            return "libfabric unavailable";
        }
        const char *value = strerror_fn(-error_code);
        return value != nullptr ? std::string(value) : std::string("libfabric error");
    }

    ~LibfabricApi() {
#if defined(__linux__)
        if (handle != nullptr) {
            dlclose(handle);
            handle = nullptr;
        }
#endif
    }
};

struct GdrCopyApi {
#if defined(__linux__)
    void *handle = nullptr;
#endif
    decltype(&gdr_open) open = nullptr;
    decltype(&gdr_close) close = nullptr;
    decltype(&gdr_pin_buffer_v2) pin_v2 = nullptr;
    decltype(&gdr_pin_buffer) pin = nullptr;
    decltype(&gdr_unpin_buffer) unpin = nullptr;
    decltype(&gdr_map) map = nullptr;
    decltype(&gdr_unmap) unmap = nullptr;
    decltype(&gdr_get_info_v2) get_info = nullptr;
    decltype(&gdr_copy_to_mapping) copy_to_mapping = nullptr;
    decltype(&gdr_copy_from_mapping) copy_from_mapping = nullptr;
    bool available = false;

    bool load() {
#if !defined(__linux__)
        return false;
#else
        const char *names[] = {"libgdrapi.so.2", "libgdrapi.so", nullptr};
        for (const char **name = names; *name != nullptr; ++name) {
            handle = dlopen(*name, RTLD_LAZY | RTLD_LOCAL);
            if (handle != nullptr) {
                break;
            }
        }
        if (handle == nullptr) {
            return false;
        }
#define RLITE_LOAD_GDR(symbol_name, target)                                                          \
    target = reinterpret_cast<decltype(target)>(dlsym(handle, symbol_name));                         \
    if (target == nullptr) {                                                                         \
        return false;                                                                                \
    }
        RLITE_LOAD_GDR("gdr_open", open)
        RLITE_LOAD_GDR("gdr_close", close)
        pin_v2 = reinterpret_cast<decltype(pin_v2)>(dlsym(handle, "gdr_pin_buffer_v2"));
        RLITE_LOAD_GDR("gdr_pin_buffer", pin)
        RLITE_LOAD_GDR("gdr_unpin_buffer", unpin)
        RLITE_LOAD_GDR("gdr_map", map)
        RLITE_LOAD_GDR("gdr_unmap", unmap)
        RLITE_LOAD_GDR("gdr_get_info_v2", get_info)
        RLITE_LOAD_GDR("gdr_copy_to_mapping", copy_to_mapping)
        RLITE_LOAD_GDR("gdr_copy_from_mapping", copy_from_mapping)
#undef RLITE_LOAD_GDR
        available = true;
        return true;
#endif
    }

    ~GdrCopyApi() {
#if defined(__linux__)
        if (handle != nullptr) {
            dlclose(handle);
            handle = nullptr;
        }
#endif
    }
};

struct CudaRuntime {
#if defined(__linux__)
    void *handle = nullptr;
#endif
    using GetDeviceCountFn = cudaError_t (*)(int *);
    using MemcpyFn = cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind);
    using DeviceSynchronizeFn = cudaError_t (*)();
    using IpcGetMemHandleFn = cudaError_t (*)(cudaIpcMemHandle_t *, void *);
    using IpcOpenMemHandleFn = cudaError_t (*)(void **, cudaIpcMemHandle_t, unsigned int);
    using IpcCloseMemHandleFn = cudaError_t (*)(void *);
    using GetErrorStringFn = const char *(*)(cudaError_t);

    GetDeviceCountFn get_device_count = nullptr;
    MemcpyFn memcpy_fn = nullptr;
    DeviceSynchronizeFn device_synchronize = nullptr;
    IpcGetMemHandleFn ipc_get_mem_handle = nullptr;
    IpcOpenMemHandleFn ipc_open_mem_handle = nullptr;
    IpcCloseMemHandleFn ipc_close_mem_handle = nullptr;
    GetErrorStringFn get_error_string = nullptr;
    bool available = false;
    bool supports_ipc = false;
    bool supports_peer_access = false;

    bool load() {
#if !defined(__linux__)
        return false;
#else
        const char *names[] = {"libcudart.so.12", "libcudart.so.11.0", "libcudart.so", nullptr};
        for (const char **name = names; *name != nullptr; ++name) {
            handle = dlopen(*name, RTLD_LAZY | RTLD_LOCAL);
            if (handle != nullptr) {
                break;
            }
        }
        if (handle == nullptr) {
            return false;
        }
#define RLITE_LOAD_CUDA(symbol_name, target)                                                         \
    target = reinterpret_cast<decltype(target)>(dlsym(handle, symbol_name));                         \
    if (target == nullptr) {                                                                         \
        return false;                                                                                \
    }
        RLITE_LOAD_CUDA("cudaGetDeviceCount", get_device_count)
        RLITE_LOAD_CUDA("cudaMemcpy", memcpy_fn)
        RLITE_LOAD_CUDA("cudaDeviceSynchronize", device_synchronize)
        ipc_get_mem_handle =
            reinterpret_cast<IpcGetMemHandleFn>(dlsym(handle, "cudaIpcGetMemHandle"));
        ipc_open_mem_handle =
            reinterpret_cast<IpcOpenMemHandleFn>(dlsym(handle, "cudaIpcOpenMemHandle"));
        ipc_close_mem_handle =
            reinterpret_cast<IpcCloseMemHandleFn>(dlsym(handle, "cudaIpcCloseMemHandle"));
        get_error_string =
            reinterpret_cast<GetErrorStringFn>(dlsym(handle, "cudaGetErrorString"));
#undef RLITE_LOAD_CUDA
        int count = 0;
        if (get_device_count(&count) == cudaSuccess && count > 0) {
            available = true;
            supports_ipc =
                ipc_get_mem_handle != nullptr &&
                ipc_open_mem_handle != nullptr &&
                ipc_close_mem_handle != nullptr;
            supports_peer_access = count > 1;
        }
        return available;
#endif
    }

    std::string error_string(cudaError_t status) const {
        if (get_error_string == nullptr) {
            return "cuda runtime unavailable";
        }
        const char *value = get_error_string(status);
        return value != nullptr ? std::string(value) : std::string("cuda runtime error");
    }

    ~CudaRuntime() {
#if defined(__linux__)
        if (handle != nullptr) {
            dlclose(handle);
            handle = nullptr;
        }
#endif
    }
};

struct LocalRegion {
    std::string name;
    void *base_ptr = nullptr;
    uint64_t num_bytes = 0;
    rlite_memory_kind memory_kind = RLITE_MEMORY_CPU;
    int32_t device_id = -1;
    std::string gpu_uuid;
    uint64_t base_address = 0;
    uint64_t remote_key = 0;
    struct fid_mr *mr = nullptr;
    void *mr_desc = nullptr;
    std::vector<uint8_t> ipc_handle;
    bool gdr_mapped = false;
    gdr_t gdr_context = nullptr;
    gdr_mh_t gdr_handle{};
    void *gdr_map_ptr = nullptr;
    void *gdr_user_ptr = nullptr;
    size_t gdr_map_size = 0;
};

struct PeerRegion {
    std::string name;
    uint64_t base_address = 0;
    uint64_t num_bytes = 0;
    rlite_memory_kind memory_kind = RLITE_MEMORY_CPU;
    int32_t device_id = -1;
    std::string gpu_uuid;
    uint64_t remote_key = 0;
    std::vector<uint8_t> ipc_handle;
};

struct PeerInfo {
    rlite_peer_descriptor descriptor{};
    fi_addr_t fi_addr = FI_ADDR_NOTAVAIL;
    bool av_inserted = false;
    std::unordered_map<std::string, PeerRegion> regions;
};

class NativeSession {
public:
    explicit NativeSession(const rlite_session_options &options)
        : rank_(options.rank),
          world_size_(options.world_size),
          host_(options.host != nullptr ? options.host : ""),
          nic_name_(options.nic_name != nullptr ? options.nic_name : ""),
          provider_name_(options.provider_name != nullptr ? options.provider_name : ""),
          max_chunk_bytes_(kDefaultChunkBytes) {
        const char *chunk_env = std::getenv("RLITE_TRANSPORT_MAX_CHUNK_BYTES");
        if (chunk_env != nullptr) {
            const uint64_t parsed = static_cast<uint64_t>(std::strtoull(chunk_env, nullptr, 10));
            if (parsed > 0) {
                max_chunk_bytes_ = parsed;
            }
        }
    }

    ~NativeSession() { cleanup(); }

    rlite_transport_status initialize(std::string *error) {
        cuda_.load();
        gdr_.load();
        fabric_api_.load();
        if (!fabric_api_.available) {
            return RLITE_TRANSPORT_STATUS_OK;
        }
        return initialize_fabric(error);
    }

    rlite_capability_report capability_report() const {
        rlite_capability_report report{};
        report.supports_fi_rma = fabric_ready_ ? 1 : 0;
        report.supports_fi_hmem = supports_fi_hmem_ ? 1 : 0;
        report.supports_cuda_ipc = cuda_.supports_ipc ? 1 : 0;
        report.supports_gdrcopy = gdr_.available ? 1 : 0;
        report.supports_peer_access = cuda_.supports_peer_access ? 1 : 0;
        report.preferred_remote_path = fabric_ready_ ? RLITE_PATH_LIBFABRIC_RMA : RLITE_PATH_UNAVAILABLE;
        report.fallback_remote_path = gdr_.available ? RLITE_PATH_STAGED_HOST : RLITE_PATH_UNAVAILABLE;
        copy_string(report.provider_name, sizeof(report.provider_name), effective_provider_name().c_str());
        const std::string note = fabric_ready_
            ? "native transport ready"
            : "native transport opened without libfabric fabric";
        copy_string(report.note, sizeof(report.note), note.c_str());
        return report;
    }

    rlite_transport_status register_region(
        const rlite_region_registration &registration,
        rlite_region_descriptor *out_descriptor,
        std::string *error
    ) {
        if (registration.tensor_name == nullptr || registration.base_ptr == nullptr || registration.num_bytes == 0) {
            *error = "invalid region registration";
            return RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT;
        }

        LocalRegion region{};
        region.name = registration.tensor_name;
        region.base_ptr = registration.base_ptr;
        region.num_bytes = registration.num_bytes;
        region.memory_kind = static_cast<rlite_memory_kind>(registration.memory_kind);
        region.device_id = registration.device_id;
        region.gpu_uuid = registration.gpu_uuid != nullptr ? registration.gpu_uuid : "";

        if (fabric_ready_) {
            const auto status = register_memory_region(region, registration.requested_key, error);
            if (status != RLITE_TRANSPORT_STATUS_OK &&
                !(region.memory_kind == RLITE_MEMORY_CUDA && gdr_.available)) {
                return status;
            }
        }

        if (region.memory_kind == RLITE_MEMORY_CUDA) {
            if (cuda_.supports_ipc && cuda_.ipc_get_mem_handle != nullptr) {
                cudaIpcMemHandle_t ipc_handle{};
                if (cuda_.ipc_get_mem_handle(&ipc_handle, region.base_ptr) == cudaSuccess) {
                    region.ipc_handle.assign(
                        reinterpret_cast<const uint8_t *>(&ipc_handle),
                        reinterpret_cast<const uint8_t *>(&ipc_handle) + sizeof(ipc_handle)
                    );
                }
            }
            map_gdrcopy(region);
        }

        local_regions_[region.name] = std::move(region);
        fill_region_descriptor(local_regions_[registration.tensor_name], out_descriptor);
        return RLITE_TRANSPORT_STATUS_OK;
    }

    rlite_transport_status install_peer(
        const rlite_peer_descriptor &peer,
        const rlite_region_descriptor *regions,
        size_t region_count,
        std::string *error
    ) {
        PeerInfo peer_info{};
        peer_info.descriptor = peer;
        for (size_t index = 0; index < region_count; ++index) {
            PeerRegion region{};
            region.name = regions[index].tensor_name;
            region.base_address = regions[index].base_address;
            region.num_bytes = regions[index].num_bytes;
            region.memory_kind = static_cast<rlite_memory_kind>(regions[index].memory_kind);
            region.device_id = regions[index].device_id;
            region.gpu_uuid = regions[index].gpu_uuid;
            region.remote_key = regions[index].remote_key;
            region.ipc_handle.assign(
                regions[index].ipc_handle,
                regions[index].ipc_handle + regions[index].ipc_handle_length
            );
            peer_info.regions[region.name] = std::move(region);
        }

        if (fabric_ready_ && peer.fabric_address_length > 0) {
            const int result = fabric_api_.av_insert(
                av_,
                peer.fabric_address,
                1,
                &peer_info.fi_addr,
                0,
                nullptr
            );
            if (result != 1) {
                *error = "fi_av_insert failed";
                return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
            }
            peer_info.av_inserted = true;
        }

        peers_[peer.rank] = std::move(peer_info);
        return RLITE_TRANSPORT_STATUS_OK;
    }

    rlite_transport_status query_local_peer(rlite_peer_descriptor *out_peer, std::string *error) const {
        if (out_peer == nullptr) {
            *error = "peer descriptor output must be provided";
            return RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT;
        }
        std::memset(out_peer, 0, sizeof(*out_peer));
        out_peer->rank = rank_;
        out_peer->cuda_device_id = -1;
        copy_string(out_peer->host, sizeof(out_peer->host), host_.c_str());
        copy_string(out_peer->nic_name, sizeof(out_peer->nic_name), nic_name_.c_str());
        copy_string(
            out_peer->provider_name,
            sizeof(out_peer->provider_name),
            effective_provider_name().c_str()
        );
        out_peer->fabric_address_length =
            std::min(local_fabric_address_.size(), sizeof(out_peer->fabric_address));
        if (out_peer->fabric_address_length > 0) {
            std::memcpy(
                out_peer->fabric_address,
                local_fabric_address_.data(),
                out_peer->fabric_address_length
            );
        }

        for (const auto &item : local_regions_) {
            const LocalRegion &region = item.second;
            if (region.memory_kind != RLITE_MEMORY_CUDA) {
                continue;
            }
            out_peer->cuda_device_id = region.device_id;
            copy_string(out_peer->gpu_uuid, sizeof(out_peer->gpu_uuid), region.gpu_uuid.c_str());
            break;
        }
        return RLITE_TRANSPORT_STATUS_OK;
    }

    rlite_transport_status execute(
        const rlite_transfer_task *tasks,
        size_t task_count,
        rlite_execution_stats *stats,
        std::string *error
    ) {
        if (stats == nullptr) {
            *error = "stats output must be provided";
            return RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT;
        }
        std::memset(stats, 0, sizeof(*stats));

        for (size_t index = 0; index < task_count; ++index) {
            const rlite_transfer_task &task = tasks[index];
            if (task.src_rank != rank_) {
                continue;
            }
            const auto status = execute_one(task, stats, error);
            if (status != RLITE_TRANSPORT_STATUS_OK) {
                copy_string(stats->last_error, sizeof(stats->last_error), error->c_str());
                return status;
            }
        }
        return RLITE_TRANSPORT_STATUS_OK;
    }

private:
    rlite_transport_status initialize_fabric(std::string *error) {
        std::unique_ptr<fi_info, void (*)(fi_info *)> hints(fabric_api_.allocinfo(), fabric_api_.freeinfo);
        if (!hints) {
            *error = "fi_allocinfo failed";
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }
        hints->caps = FI_RMA | FI_MSG;
        hints->mode = FI_CONTEXT;
        hints->ep_attr->type = FI_EP_RDM;
        if (!provider_name_.empty()) {
            hints->fabric_attr->prov_name = const_cast<char *>(provider_name_.c_str());
        }

        fi_info *info = nullptr;
        const int ret = fabric_api_.getinfo(
            FI_VERSION(1, 18),
            nullptr,
            nullptr,
            FI_SOURCE,
            hints.get(),
            &info
        );
        if (ret != 0) {
            *error = "fi_getinfo failed: " + fabric_api_.strerror_value(ret);
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }

        info_ = info;
        if (fabric_api_.fabric(info_->fabric_attr, &fabric_, nullptr) != 0 ||
            fabric_api_.domain(fabric_, info_, &domain_, nullptr) != 0 ||
            fabric_api_.endpoint(domain_, info_, &ep_, nullptr) != 0) {
            *error = "failed to initialize libfabric fabric/domain/endpoint";
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }

        fi_cq_attr cq_attr{};
        cq_attr.format = FI_CQ_FORMAT_CONTEXT;
        cq_attr.size = 1024;
        if (fabric_api_.cq_open(domain_, &cq_attr, &cq_, nullptr) != 0) {
            *error = "fi_cq_open failed";
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }

        fi_av_attr av_attr{};
        av_attr.type = FI_AV_TABLE;
        av_attr.count = static_cast<size_t>(std::max(1, world_size_));
        if (fabric_api_.av_open(domain_, &av_attr, &av_, nullptr) != 0) {
            *error = "fi_av_open failed";
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }

        if (fabric_api_.ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV) != 0 ||
            fabric_api_.ep_bind(ep_, &av_->fid, 0) != 0 ||
            fabric_api_.enable(ep_) != 0) {
            *error = "failed to bind/enable libfabric endpoint";
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }

        size_t addr_len = 0;
        if (fabric_api_.getname(&ep_->fid, nullptr, &addr_len) != -FI_ETOOSMALL) {
            *error = "fi_getname size query failed";
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }
        local_fabric_address_.resize(addr_len);
        if (fabric_api_.getname(&ep_->fid, local_fabric_address_.data(), &addr_len) != 0) {
            *error = "fi_getname payload query failed";
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }
        local_fabric_address_.resize(addr_len);

        supports_fi_hmem_ = (info_->domain_attr->mr_mode & FI_MR_HMEM) != 0;
        fabric_ready_ = true;
        return RLITE_TRANSPORT_STATUS_OK;
    }

    rlite_transport_status register_memory_region(LocalRegion &region, uint64_t requested_key, std::string *error) {
        fi_mr_attr attr{};
        iovec iov{};
        iov.iov_base = region.base_ptr;
        iov.iov_len = static_cast<size_t>(region.num_bytes);
        attr.mr_iov = &iov;
        attr.iov_count = 1;
        attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        attr.offset = 0;
        attr.requested_key = requested_key;
        attr.iface = static_cast<fi_hmem_iface>(
            region.memory_kind == RLITE_MEMORY_CUDA ? FI_HMEM_CUDA : FI_HMEM_SYSTEM
        );
        if (region.memory_kind == RLITE_MEMORY_CUDA) {
            attr.device.cuda = region.device_id < 0 ? 0 : static_cast<uint64_t>(region.device_id);
        }

        const uint64_t flags = region.memory_kind == RLITE_MEMORY_CUDA ? FI_HMEM_DEVICE_ONLY : 0;
        const int ret = fabric_api_.mr_regattr(domain_, &attr, flags, &region.mr);
        if (ret != 0) {
            *error = "fi_mr_regattr failed: " + fabric_api_.strerror_value(ret);
            return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
        }

        region.mr_desc = fi_mr_desc(region.mr);
        region.remote_key = fi_mr_key(region.mr);
        region.base_address = (info_->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0
            ? reinterpret_cast<uint64_t>(region.base_ptr)
            : 0;

        if ((info_->domain_attr->mr_mode & FI_MR_ENDPOINT) != 0) {
            if (fi_mr_bind(region.mr, &ep_->fid, 0) != 0 || fi_mr_enable(region.mr) != 0) {
                *error = "failed to bind/enable memory region";
                return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
            }
        }
        return RLITE_TRANSPORT_STATUS_OK;
    }

    void map_gdrcopy(LocalRegion &region) {
        if (!gdr_.available || region.memory_kind != RLITE_MEMORY_CUDA) {
            return;
        }
        const uintptr_t addr = reinterpret_cast<uintptr_t>(region.base_ptr);
        const uintptr_t aligned = addr & GPU_PAGE_MASK;
        const size_t offset = static_cast<size_t>(addr - aligned);
        const size_t aligned_size = static_cast<size_t>(
            ((addr + region.num_bytes + GPU_PAGE_OFFSET) & GPU_PAGE_MASK) - aligned
        );
        region.gdr_context = gdr_.open();
        if (region.gdr_context == nullptr) {
            return;
        }
        gdr_mh_t handle{};
        int result = 0;
        if (gdr_.pin_v2 != nullptr) {
            result = gdr_.pin_v2(region.gdr_context, aligned, aligned_size, GDR_PIN_FLAG_DEFAULT, &handle);
        } else {
            result = gdr_.pin(region.gdr_context, aligned, aligned_size, 0, 0, &handle);
        }
        if (result != 0) {
            gdr_.close(region.gdr_context);
            region.gdr_context = nullptr;
            return;
        }
        void *mapped = nullptr;
        if (gdr_.map(region.gdr_context, handle, &mapped, aligned_size) != 0) {
            gdr_.unpin(region.gdr_context, handle);
            gdr_.close(region.gdr_context);
            region.gdr_context = nullptr;
            return;
        }
        region.gdr_handle = handle;
        region.gdr_map_ptr = mapped;
        region.gdr_user_ptr = static_cast<char *>(mapped) + offset;
        region.gdr_map_size = aligned_size;
        region.gdr_mapped = true;
    }

    void fill_region_descriptor(const LocalRegion &region, rlite_region_descriptor *out) const {
        std::memset(out, 0, sizeof(*out));
        copy_string(out->tensor_name, sizeof(out->tensor_name), region.name.c_str());
        out->base_address = region.base_address;
        out->num_bytes = region.num_bytes;
        out->memory_kind = region.memory_kind;
        out->device_id = region.device_id;
        copy_string(out->gpu_uuid, sizeof(out->gpu_uuid), region.gpu_uuid.c_str());
        out->remote_key = region.remote_key;
        out->ipc_handle_length = std::min(region.ipc_handle.size(), sizeof(out->ipc_handle));
        if (out->ipc_handle_length > 0) {
            std::memcpy(out->ipc_handle, region.ipc_handle.data(), out->ipc_handle_length);
        }
    }

    rlite_transport_status execute_one(
        const rlite_transfer_task &task,
        rlite_execution_stats *stats,
        std::string *error
    ) {
        auto local_it = local_regions_.find(task.tensor_name);
        if (local_it == local_regions_.end()) {
            *error = "local source tensor not registered";
            return RLITE_TRANSPORT_STATUS_NOT_FOUND;
        }
        LocalRegion &local = local_it->second;

        if (task.dst_rank == rank_) {
            return local_copy(local, task, stats, error);
        }

        auto peer_it = peers_.find(task.dst_rank);
        if (peer_it == peers_.end()) {
            *error = "peer rank not installed";
            return RLITE_TRANSPORT_STATUS_NOT_FOUND;
        }
        auto remote_region_it = peer_it->second.regions.find(task.tensor_name);
        if (remote_region_it == peer_it->second.regions.end()) {
            *error = "peer tensor descriptor not installed";
            return RLITE_TRANSPORT_STATUS_NOT_FOUND;
        }

        const PeerInfo &peer = peer_it->second;
        const PeerRegion &remote = remote_region_it->second;

        if (peer.descriptor.host[0] != '\0' && host_ == peer.descriptor.host &&
            remote.memory_kind == RLITE_MEMORY_CUDA && !remote.ipc_handle.empty()) {
            const auto status = cuda_ipc_write(local, remote, task, stats, error);
            if (status == RLITE_TRANSPORT_STATUS_OK) {
                return status;
            }
        }

        if (fabric_ready_ && peer.av_inserted && remote.remote_key != 0) {
            return remote_write(local, peer, remote, task, stats, error);
        }

        *error = "no supported path for transfer task";
        return RLITE_TRANSPORT_STATUS_UNAVAILABLE;
    }

    rlite_transport_status local_copy(
        LocalRegion &local,
        const rlite_transfer_task &task,
        rlite_execution_stats *stats,
        std::string *error
    ) {
        auto dst_it = local_regions_.find(task.tensor_name);
        if (dst_it == local_regions_.end()) {
            *error = "local destination tensor not registered";
            return RLITE_TRANSPORT_STATUS_NOT_FOUND;
        }
        LocalRegion &dst = dst_it->second;
        if (local.memory_kind == RLITE_MEMORY_CPU && dst.memory_kind == RLITE_MEMORY_CPU) {
            std::memcpy(
                static_cast<char *>(dst.base_ptr) + task.dst_offset,
                static_cast<char *>(local.base_ptr) + task.src_offset,
                static_cast<size_t>(task.num_bytes)
            );
            stats->bytes_copied += task.num_bytes;
            stats->completed_tasks += 1;
            stats->path_counts[RLITE_PATH_MEMCPY] += 1;
            return RLITE_TRANSPORT_STATUS_OK;
        }
        if (cuda_.available &&
            cuda_.memcpy_fn != nullptr &&
            cuda_.device_synchronize != nullptr) {
            const cudaError_t copy_status = cuda_.memcpy_fn(
                static_cast<char *>(dst.base_ptr) + task.dst_offset,
                static_cast<char *>(local.base_ptr) + task.src_offset,
                static_cast<size_t>(task.num_bytes),
                cudaMemcpyDefault
            );
            if (copy_status == cudaSuccess && cuda_.device_synchronize() == cudaSuccess) {
                stats->bytes_copied += task.num_bytes;
                stats->completed_tasks += 1;
                stats->path_counts[RLITE_PATH_MEMCPY] += 1;
                return RLITE_TRANSPORT_STATUS_OK;
            }
            *error = "cudaMemcpy failed: " + cuda_.error_string(copy_status);
            return RLITE_TRANSPORT_STATUS_CUDA_ERROR;
        }
        *error = "local CUDA copy unavailable";
        return RLITE_TRANSPORT_STATUS_UNAVAILABLE;
    }

    rlite_transport_status cuda_ipc_write(
        LocalRegion &local,
        const PeerRegion &remote,
        const rlite_transfer_task &task,
        rlite_execution_stats *stats,
        std::string *error
    ) {
        if (!cuda_.supports_ipc ||
            cuda_.ipc_open_mem_handle == nullptr ||
            cuda_.ipc_close_mem_handle == nullptr ||
            cuda_.memcpy_fn == nullptr ||
            cuda_.device_synchronize == nullptr ||
            remote.ipc_handle.size() != sizeof(cudaIpcMemHandle_t)) {
            *error = "CUDA IPC unavailable";
            return RLITE_TRANSPORT_STATUS_UNAVAILABLE;
        }
        cudaIpcMemHandle_t ipc_handle{};
        std::memcpy(&ipc_handle, remote.ipc_handle.data(), sizeof(ipc_handle));
        void *mapped = nullptr;
        const cudaError_t open_status =
            cuda_.ipc_open_mem_handle(&mapped, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
        if (open_status != cudaSuccess) {
            *error = "cudaIpcOpenMemHandle failed: " + cuda_.error_string(open_status);
            return RLITE_TRANSPORT_STATUS_CUDA_ERROR;
        }
        const cudaError_t copy_status = cuda_.memcpy_fn(
            static_cast<char *>(mapped) + task.dst_offset,
            static_cast<char *>(local.base_ptr) + task.src_offset,
            static_cast<size_t>(task.num_bytes),
            cudaMemcpyDefault
        );
        const cudaError_t close_status = cuda_.ipc_close_mem_handle(mapped);
        const cudaError_t sync_status = cuda_.device_synchronize();
        if (copy_status != cudaSuccess || sync_status != cudaSuccess || close_status != cudaSuccess) {
            *error = "cudaMemcpy via IPC failed";
            return RLITE_TRANSPORT_STATUS_CUDA_ERROR;
        }
        stats->bytes_copied += task.num_bytes;
        stats->completed_tasks += 1;
        stats->path_counts[RLITE_PATH_CUDA_IPC] += 1;
        return RLITE_TRANSPORT_STATUS_OK;
    }

    rlite_transport_status remote_write(
        LocalRegion &local,
        const PeerInfo &peer,
        const PeerRegion &remote,
        const rlite_transfer_task &task,
        rlite_execution_stats *stats,
        std::string *error
    ) {
        if (local.mr != nullptr) {
            return post_remote_write(
                static_cast<char *>(local.base_ptr) + task.src_offset,
                task.num_bytes,
                local.mr_desc,
                peer,
                remote.base_address + task.dst_offset,
                remote.remote_key,
                RLITE_PATH_LIBFABRIC_RMA,
                stats,
                error
            );
        }

        if (local.memory_kind == RLITE_MEMORY_CUDA && local.gdr_mapped) {
            const size_t chunk_size = static_cast<size_t>(std::min<uint64_t>(task.num_bytes, max_chunk_bytes_));
            std::vector<uint8_t> staging(chunk_size);
            return staged_remote_write(local, peer, remote, task, staging, stats, error);
        }

        *error = "local memory region is not fabric-registered";
        return RLITE_TRANSPORT_STATUS_UNAVAILABLE;
    }

    rlite_transport_status staged_remote_write(
        LocalRegion &local,
        const PeerInfo &peer,
        const PeerRegion &remote,
        const rlite_transfer_task &task,
        std::vector<uint8_t> &staging,
        rlite_execution_stats *stats,
        std::string *error
    ) {
        LocalRegion stage_region{};
        stage_region.name = "__staging__";
        stage_region.base_ptr = staging.data();
        stage_region.num_bytes = staging.size();
        stage_region.memory_kind = RLITE_MEMORY_CPU;

        const auto status = register_memory_region(stage_region, stable_key(rank_, stage_region.name), error);
        if (status != RLITE_TRANSPORT_STATUS_OK) {
            return status;
        }

        uint64_t moved = 0;
        while (moved < task.num_bytes) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(staging.size(), task.num_bytes - moved));
            if (gdr_.copy_from_mapping(
                    local.gdr_handle,
                    staging.data(),
                    static_cast<const char *>(local.gdr_user_ptr) + task.src_offset + moved,
                    chunk
                ) != 0) {
                *error = "gdr_copy_from_mapping failed";
                fi_close(&stage_region.mr->fid);
                return RLITE_TRANSPORT_STATUS_GDRCOPY_ERROR;
            }
            const auto chunk_status = post_remote_write(
                staging.data(),
                chunk,
                stage_region.mr_desc,
                peer,
                remote.base_address + task.dst_offset + moved,
                remote.remote_key,
                RLITE_PATH_STAGED_HOST,
                stats,
                error
            );
            if (chunk_status != RLITE_TRANSPORT_STATUS_OK) {
                fi_close(&stage_region.mr->fid);
                return chunk_status;
            }
            moved += chunk;
        }

        fi_close(&stage_region.mr->fid);
        return RLITE_TRANSPORT_STATUS_OK;
    }

    rlite_transport_status post_remote_write(
        void *local_ptr,
        uint64_t num_bytes,
        void *mr_desc,
        const PeerInfo &peer,
        uint64_t remote_address,
        uint64_t remote_key,
        rlite_transfer_path path,
        rlite_execution_stats *stats,
        std::string *error
    ) {
        uint64_t remaining = num_bytes;
        uint64_t moved = 0;
        size_t posted = 0;
        const size_t total_posts = static_cast<size_t>(
            (num_bytes + max_chunk_bytes_ - 1) / max_chunk_bytes_
        );
        std::vector<std::array<uint64_t, 2>> contexts(total_posts);

        while (remaining > 0) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, max_chunk_bytes_));
            contexts[posted] = {moved, chunk};
            int ret = 0;
            do {
                ret = fabric_api_.write(
                    ep_,
                    static_cast<char *>(local_ptr) + moved,
                    chunk,
                    mr_desc,
                    peer.fi_addr,
                    remote_address + moved,
                    remote_key,
                    &contexts[posted]
                );
                if (ret == -FI_EAGAIN) {
                    const auto wait_status = wait_for_completions(1, error);
                    if (wait_status != RLITE_TRANSPORT_STATUS_OK) {
                        return wait_status;
                    }
                }
            } while (ret == -FI_EAGAIN);
            if (ret != 0) {
                *error = "fi_write failed: " + fabric_api_.strerror_value(ret);
                return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
            }
            remaining -= chunk;
            moved += chunk;
            posted += 1;
        }

        const auto wait_status = wait_for_completions(posted, error);
        if (wait_status != RLITE_TRANSPORT_STATUS_OK) {
            return wait_status;
        }
        stats->bytes_copied += num_bytes;
        stats->completed_tasks += 1;
        stats->path_counts[path] += 1;
        return RLITE_TRANSPORT_STATUS_OK;
    }

    rlite_transport_status wait_for_completions(size_t expected, std::string *error) {
        size_t completed = 0;
        while (completed < expected) {
            fi_cq_entry entry{};
            const ssize_t ret = fabric_api_.cq_read(cq_, &entry, 1);
            if (ret > 0) {
                completed += static_cast<size_t>(ret);
                continue;
            }
            if (ret == -FI_EAGAIN) {
                continue;
            }
            if (ret < 0) {
                fi_cq_err_entry err_entry{};
                fabric_api_.cq_readerr(cq_, &err_entry, 0);
                *error = "fi_cq_read failed";
                return RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR;
            }
        }
        return RLITE_TRANSPORT_STATUS_OK;
    }

    std::string effective_provider_name() const {
        if (fabric_ready_ && info_ != nullptr && info_->fabric_attr != nullptr && info_->fabric_attr->prov_name != nullptr) {
            return info_->fabric_attr->prov_name;
        }
        if (!provider_name_.empty()) {
            return provider_name_;
        }
        return "mock_loopback";
    }

    void cleanup() {
        for (auto &item : local_regions_) {
            LocalRegion &region = item.second;
            if (region.gdr_mapped && region.gdr_context != nullptr) {
                gdr_.unmap(region.gdr_context, region.gdr_handle, region.gdr_map_ptr, region.gdr_map_size);
                gdr_.unpin(region.gdr_context, region.gdr_handle);
                gdr_.close(region.gdr_context);
                region.gdr_context = nullptr;
            }
            if (region.mr != nullptr) {
                fi_close(&region.mr->fid);
                region.mr = nullptr;
            }
        }
        if (ep_ != nullptr) {
            fi_close(&ep_->fid);
            ep_ = nullptr;
        }
        if (av_ != nullptr) {
            fi_close(&av_->fid);
            av_ = nullptr;
        }
        if (cq_ != nullptr) {
            fi_close(&cq_->fid);
            cq_ = nullptr;
        }
        if (domain_ != nullptr) {
            fi_close(&domain_->fid);
            domain_ = nullptr;
        }
        if (fabric_ != nullptr) {
            fi_close(&fabric_->fid);
            fabric_ = nullptr;
        }
        if (info_ != nullptr && fabric_api_.freeinfo != nullptr) {
            fabric_api_.freeinfo(info_);
            info_ = nullptr;
        }
    }

    int32_t rank_;
    int32_t world_size_;
    std::string host_;
    std::string nic_name_;
    std::string provider_name_;
    uint64_t max_chunk_bytes_;
    LibfabricApi fabric_api_{};
    GdrCopyApi gdr_{};
    CudaRuntime cuda_{};
    bool fabric_ready_ = false;
    bool supports_fi_hmem_ = false;
    struct fi_info *info_ = nullptr;
    struct fid_fabric *fabric_ = nullptr;
    struct fid_domain *domain_ = nullptr;
    struct fid_ep *ep_ = nullptr;
    struct fid_cq *cq_ = nullptr;
    struct fid_av *av_ = nullptr;
    std::vector<uint8_t> local_fabric_address_;
    std::unordered_map<std::string, LocalRegion> local_regions_;
    std::unordered_map<int32_t, PeerInfo> peers_;
};

std::string build_probe_json() {
    LibfabricApi fabric{};
    GdrCopyApi gdr{};
    CudaRuntime cuda{};
    fabric.load();
    gdr.load();
    cuda.load();

    const bool supports_fi_rma = fabric.available;
    bool supports_fi_hmem = false;
    std::string provider_name = supports_fi_rma ? "libfabric" : "mock_loopback";
    if (supports_fi_rma) {
        std::unique_ptr<fi_info, void (*)(fi_info *)> hints(fabric.allocinfo(), fabric.freeinfo);
        if (hints) {
            hints->caps = FI_RMA | FI_MSG;
            hints->mode = FI_CONTEXT;
            hints->ep_attr->type = FI_EP_RDM;
            fi_info *info = nullptr;
            if (fabric.getinfo(FI_VERSION(1, 18), nullptr, nullptr, FI_SOURCE, hints.get(), &info) == 0 &&
                info != nullptr) {
                if (info->domain_attr != nullptr) {
                    supports_fi_hmem = (info->domain_attr->mr_mode & FI_MR_HMEM) != 0;
                }
                if (info->fabric_attr != nullptr && info->fabric_attr->prov_name != nullptr) {
                    provider_name = info->fabric_attr->prov_name;
                }
                fabric.freeinfo(info);
            }
        }
    }
    const bool supports_cuda_ipc = cuda.supports_ipc;
    const bool supports_gdrcopy = gdr.available;
    const bool supports_peer_access = cuda.supports_peer_access;

    return std::string("{") +
        "\"supports_fi_rma\":" + (supports_fi_rma ? "true" : "false") + "," +
        "\"supports_fi_hmem\":" + (supports_fi_hmem ? "true" : "false") + "," +
        "\"supports_cuda_ipc\":" + (supports_cuda_ipc ? "true" : "false") + "," +
        "\"supports_gdrcopy\":" + (supports_gdrcopy ? "true" : "false") + "," +
        "\"supports_peer_access\":" + (supports_peer_access ? "true" : "false") + "," +
        "\"preferred_remote_path\":\"" + (supports_fi_rma ? "libfabric_rma" : "unavailable") + "\"," +
        "\"fallback_remote_path\":\"" + (supports_gdrcopy ? "staged_host" : "mock_loopback") + "\"," +
        "\"provider_name\":\"" + provider_name + "\"," +
        "\"notes\":[\"native transport probe completed\"]" +
        "}";
}

}  // namespace

struct rlite_session_handle {
    NativeSession *impl;
};

extern "C" {

const char *rlite_transport_runtime_version(void) {
    return "rlite-transport-native/0.1.0";
}

void *rlite_transport_probe_json(void) {
    const std::string payload = build_probe_json();
    char *value = static_cast<char *>(std::malloc(payload.size() + 1));
    if (value == nullptr) {
        return nullptr;
    }
    std::memcpy(value, payload.c_str(), payload.size() + 1);
    return value;
}

void rlite_transport_free_string(void *value) {
    std::free(value);
}

const char *rlite_transport_status_string(int status) {
    switch (status) {
        case RLITE_TRANSPORT_STATUS_OK: return "ok";
        case RLITE_TRANSPORT_STATUS_UNAVAILABLE: return "unavailable";
        case RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT: return "invalid_argument";
        case RLITE_TRANSPORT_STATUS_NOT_FOUND: return "not_found";
        case RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR: return "libfabric_error";
        case RLITE_TRANSPORT_STATUS_CUDA_ERROR: return "cuda_error";
        case RLITE_TRANSPORT_STATUS_GDRCOPY_ERROR: return "gdrcopy_error";
        case RLITE_TRANSPORT_STATUS_RUNTIME_ERROR: return "runtime_error";
        default: return "unknown";
    }
}

int rlite_transport_session_open(
    const rlite_session_options *options,
    rlite_session_handle **out_session,
    rlite_capability_report *out_report,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (options == nullptr || out_session == nullptr) {
        set_error(error_buffer, error_buffer_size, "session options and output handle are required");
        return RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT;
    }

    std::unique_ptr<NativeSession> session(new NativeSession(*options));
    std::string error;
    const auto status = session->initialize(&error);
    if (status != RLITE_TRANSPORT_STATUS_OK) {
        set_error(error_buffer, error_buffer_size, error);
        return status;
    }

    rlite_session_handle *handle = new rlite_session_handle();
    handle->impl = session.release();
    *out_session = handle;
    if (out_report != nullptr) {
        *out_report = handle->impl->capability_report();
    }
    return RLITE_TRANSPORT_STATUS_OK;
}

int rlite_transport_session_close(rlite_session_handle *session) {
    if (session == nullptr) {
        return RLITE_TRANSPORT_STATUS_OK;
    }
    delete session->impl;
    delete session;
    return RLITE_TRANSPORT_STATUS_OK;
}

int rlite_transport_session_register_region(
    rlite_session_handle *session,
    const rlite_region_registration *registration,
    rlite_region_descriptor *out_descriptor,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (session == nullptr || session->impl == nullptr || registration == nullptr || out_descriptor == nullptr) {
        set_error(error_buffer, error_buffer_size, "invalid register_region arguments");
        return RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT;
    }
    std::string error;
    const auto status = session->impl->register_region(*registration, out_descriptor, &error);
    if (status != RLITE_TRANSPORT_STATUS_OK) {
        set_error(error_buffer, error_buffer_size, error);
    }
    return status;
}

int rlite_transport_session_query_local_peer(
    rlite_session_handle *session,
    rlite_peer_descriptor *out_peer,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (session == nullptr || session->impl == nullptr || out_peer == nullptr) {
        set_error(error_buffer, error_buffer_size, "invalid query_local_peer arguments");
        return RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT;
    }
    std::string error;
    const auto status = session->impl->query_local_peer(out_peer, &error);
    if (status != RLITE_TRANSPORT_STATUS_OK) {
        set_error(error_buffer, error_buffer_size, error);
    }
    return status;
}

int rlite_transport_session_install_peer(
    rlite_session_handle *session,
    const rlite_peer_descriptor *peer,
    const rlite_region_descriptor *regions,
    size_t region_count,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (session == nullptr || session->impl == nullptr || peer == nullptr) {
        set_error(error_buffer, error_buffer_size, "invalid install_peer arguments");
        return RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT;
    }
    std::string error;
    const auto status = session->impl->install_peer(*peer, regions, region_count, &error);
    if (status != RLITE_TRANSPORT_STATUS_OK) {
        set_error(error_buffer, error_buffer_size, error);
    }
    return status;
}

int rlite_transport_session_execute(
    rlite_session_handle *session,
    const rlite_transfer_task *tasks,
    size_t task_count,
    rlite_execution_stats *stats,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (session == nullptr || session->impl == nullptr || tasks == nullptr || stats == nullptr) {
        set_error(error_buffer, error_buffer_size, "invalid execute arguments");
        return RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT;
    }
    std::string error;
    const auto status = session->impl->execute(tasks, task_count, stats, &error);
    if (status != RLITE_TRANSPORT_STATUS_OK) {
        set_error(error_buffer, error_buffer_size, error);
    }
    return status;
}

}  // extern "C"
