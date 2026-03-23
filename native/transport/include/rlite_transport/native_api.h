#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RLITE_TRANSPORT_NAME_MAX 128
#define RLITE_TRANSPORT_HOST_MAX 128
#define RLITE_TRANSPORT_PROVIDER_MAX 64
#define RLITE_TRANSPORT_UUID_MAX 64
#define RLITE_TRANSPORT_NOTE_MAX 256
#define RLITE_TRANSPORT_ERROR_MAX 512
#define RLITE_TRANSPORT_FABRIC_ADDR_MAX 1024
#define RLITE_TRANSPORT_IPC_HANDLE_MAX 128

typedef enum rlite_transport_status {
    RLITE_TRANSPORT_STATUS_OK = 0,
    RLITE_TRANSPORT_STATUS_UNAVAILABLE = 1,
    RLITE_TRANSPORT_STATUS_INVALID_ARGUMENT = 2,
    RLITE_TRANSPORT_STATUS_NOT_FOUND = 3,
    RLITE_TRANSPORT_STATUS_LIBFABRIC_ERROR = 4,
    RLITE_TRANSPORT_STATUS_CUDA_ERROR = 5,
    RLITE_TRANSPORT_STATUS_GDRCOPY_ERROR = 6,
    RLITE_TRANSPORT_STATUS_RUNTIME_ERROR = 7
} rlite_transport_status;

typedef enum rlite_memory_kind {
    RLITE_MEMORY_CPU = 0,
    RLITE_MEMORY_CUDA = 1
} rlite_memory_kind;

typedef enum rlite_transfer_path {
    RLITE_PATH_ALIAS = 0,
    RLITE_PATH_MEMCPY = 1,
    RLITE_PATH_CUDA_IPC = 2,
    RLITE_PATH_GDRCOPY = 3,
    RLITE_PATH_LIBFABRIC_RMA = 4,
    RLITE_PATH_STAGED_HOST = 5,
    RLITE_PATH_MOCK_LOOPBACK = 6,
    RLITE_PATH_UNAVAILABLE = 7
} rlite_transfer_path;

typedef struct rlite_session_options {
    int32_t rank;
    int32_t world_size;
    const char *host;
    const char *nic_name;
    const char *provider_name;
} rlite_session_options;

typedef struct rlite_capability_report {
    uint8_t supports_fi_rma;
    uint8_t supports_fi_hmem;
    uint8_t supports_cuda_ipc;
    uint8_t supports_gdrcopy;
    uint8_t supports_peer_access;
    uint32_t preferred_remote_path;
    uint32_t fallback_remote_path;
    char provider_name[RLITE_TRANSPORT_PROVIDER_MAX];
    char note[RLITE_TRANSPORT_NOTE_MAX];
} rlite_capability_report;

typedef struct rlite_region_registration {
    const char *tensor_name;
    void *base_ptr;
    uint64_t num_bytes;
    uint32_t memory_kind;
    int32_t device_id;
    const char *gpu_uuid;
    uint64_t requested_key;
} rlite_region_registration;

typedef struct rlite_region_descriptor {
    char tensor_name[RLITE_TRANSPORT_NAME_MAX];
    uint64_t base_address;
    uint64_t num_bytes;
    uint32_t memory_kind;
    int32_t device_id;
    char gpu_uuid[RLITE_TRANSPORT_UUID_MAX];
    uint64_t remote_key;
    uint8_t ipc_handle[RLITE_TRANSPORT_IPC_HANDLE_MAX];
    size_t ipc_handle_length;
} rlite_region_descriptor;

typedef struct rlite_peer_descriptor {
    int32_t rank;
    char host[RLITE_TRANSPORT_HOST_MAX];
    char nic_name[RLITE_TRANSPORT_PROVIDER_MAX];
    char provider_name[RLITE_TRANSPORT_PROVIDER_MAX];
    uint8_t fabric_address[RLITE_TRANSPORT_FABRIC_ADDR_MAX];
    size_t fabric_address_length;
    int32_t cuda_device_id;
    char gpu_uuid[RLITE_TRANSPORT_UUID_MAX];
} rlite_peer_descriptor;

typedef struct rlite_transfer_task {
    const char *tensor_name;
    int32_t src_rank;
    int32_t dst_rank;
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t num_bytes;
    uint32_t src_memory_kind;
    uint32_t dst_memory_kind;
    uint32_t preferred_path;
    uint32_t stream_id;
    int32_t priority;
} rlite_transfer_task;

typedef struct rlite_execution_stats {
    uint64_t bytes_copied;
    uint32_t completed_tasks;
    uint32_t path_counts[8];
    char last_error[RLITE_TRANSPORT_ERROR_MAX];
} rlite_execution_stats;

typedef struct rlite_session_handle rlite_session_handle;

const char *rlite_transport_runtime_version(void);
void *rlite_transport_probe_json(void);
void rlite_transport_free_string(void *value);
const char *rlite_transport_status_string(int status);

int rlite_transport_session_open(
    const rlite_session_options *options,
    rlite_session_handle **out_session,
    rlite_capability_report *out_report,
    char *error_buffer,
    size_t error_buffer_size
);

int rlite_transport_session_close(rlite_session_handle *session);

int rlite_transport_session_register_region(
    rlite_session_handle *session,
    const rlite_region_registration *registration,
    rlite_region_descriptor *out_descriptor,
    char *error_buffer,
    size_t error_buffer_size
);

int rlite_transport_session_query_local_peer(
    rlite_session_handle *session,
    rlite_peer_descriptor *out_peer,
    char *error_buffer,
    size_t error_buffer_size
);

int rlite_transport_session_install_peer(
    rlite_session_handle *session,
    const rlite_peer_descriptor *peer,
    const rlite_region_descriptor *regions,
    size_t region_count,
    char *error_buffer,
    size_t error_buffer_size
);

int rlite_transport_session_execute(
    rlite_session_handle *session,
    const rlite_transfer_task *tasks,
    size_t task_count,
    rlite_execution_stats *stats,
    char *error_buffer,
    size_t error_buffer_size
);

#ifdef __cplusplus
}
#endif
