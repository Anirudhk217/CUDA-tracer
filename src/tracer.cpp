// cudatrace: a CUPTI-based tracer that attaches to an unmodified CUDA program.
//
// Load it either with CUDA_INJECTION64_PATH=/path/to/libcudatrace.so (the
// driver calls InitializeInjection during cuInit) or with LD_PRELOAD. It
// records CUDA runtime (and optionally driver) API calls via the Callback API
// and kernel/memcpy/memset execution via the Activity API, then writes a
// Chrome/Perfetto trace when the process exits.
//
// Environment variables:
//   CUDATRACE_OUTPUT  output path (default: cudatrace_<pid>.json)
//   CUDATRACE_DRIVER  set to 1 to also record driver API calls
//   CUDATRACE_QUIET   set to 1 to suppress the summary on stderr

#include <cupti.h>
#include <cxxabi.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "trace_writer.h"

namespace cudatrace {
namespace {

constexpr size_t kActivityBufferSize = 8 * 1024 * 1024;

std::mutex gMutex;
TraceData gData;
std::atomic<bool> gInitialized{false};
bool gQuiet = false;
std::string gOutputPath;
CUpti_SubscriberHandle gSubscriber;

#define CUPTI_CHECK(call)                                                          \
  do {                                                                             \
    CUptiResult _status = (call);                                                  \
    if (_status != CUPTI_SUCCESS) {                                                \
      const char *_msg = nullptr;                                                  \
      cuptiGetResultString(_status, &_msg);                                        \
      std::fprintf(stderr, "[cudatrace] %s failed: %s\n", #call, _msg ? _msg : "?"); \
      return false;                                                                \
    }                                                                              \
  } while (0)

bool envFlag(const char *name) {
  const char *v = std::getenv(name);
  return v && *v && std::strcmp(v, "0") != 0;
}

uint32_t currentTid() {
  thread_local uint32_t tid = static_cast<uint32_t>(syscall(SYS_gettid));
  return tid;
}

std::string demangle(const char *name) {
  if (!name) return "<unknown>";
  int status = 0;
  std::unique_ptr<char, void (*)(void *)> out(abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free);
  return status == 0 && out ? std::string(out.get()) : std::string(name);
}

const char *memcpyKindName(uint8_t kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "PtoP";
    default: return "Unknown";
  }
}

const char *cudaMemcpyKindName(int kind) {
  switch (kind) {
    case 0: return "HtoH";
    case 1: return "HtoD";
    case 2: return "DtoH";
    case 3: return "DtoD";
    case 4: return "Default";
    default: return "Unknown";
  }
}

std::string dim3Json(const dim3 &d) {
  return "[" + std::to_string(d.x) + "," + std::to_string(d.y) + "," + std::to_string(d.z) + "]";
}

// Argument details for the calls people most often care about.
std::string runtimeArgs(CUpti_CallbackId cbid, const CUpti_CallbackData *info) {
  switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020: {
      auto *p = static_cast<const cudaMalloc_v3020_params *>(info->functionParams);
      return "\"bytes\":" + std::to_string(p->size);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020: {
      auto *p = static_cast<const cudaMemcpy_v3020_params *>(info->functionParams);
      return "\"bytes\":" + std::to_string(p->count) + ",\"kind\":\"" + cudaMemcpyKindName(p->kind) + "\"";
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: {
      auto *p = static_cast<const cudaMemcpyAsync_v3020_params *>(info->functionParams);
      return "\"bytes\":" + std::to_string(p->count) + ",\"kind\":\"" + cudaMemcpyKindName(p->kind) + "\"";
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: {
      auto *p = static_cast<const cudaLaunchKernel_v7000_params *>(info->functionParams);
      return "\"kernel\":\"" + jsonEscape(demangle(info->symbolName)) + "\",\"grid\":" + dim3Json(p->gridDim) +
             ",\"block\":" + dim3Json(p->blockDim) + ",\"sharedMem\":" + std::to_string(p->sharedMem);
    }
    default:
      return {};
  }
}

void CUPTIAPI apiCallback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
  const auto *info = static_cast<const CUpti_CallbackData *>(cbdata);
  uint64_t now = 0;
  cuptiGetTimestamp(&now);

  // correlationData is scratch space that lives from API entry to exit.
  if (info->callbackSite == CUPTI_API_ENTER) {
    *info->correlationData = now;
    return;
  }

  ApiEvent e;
  e.name = info->functionName ? info->functionName : "<unknown>";
  e.domain = domain == CUPTI_CB_DOMAIN_RUNTIME_API ? "runtime" : "driver";
  e.tid = currentTid();
  e.correlationId = info->correlationId;
  e.start = *info->correlationData;
  e.end = now;
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    e.args = runtimeArgs(cbid, info);
    // Both cudaError_t and CUresult are int-sized enums where 0 means success.
    if (info->functionReturnValue) {
      int rc = *static_cast<const int *>(info->functionReturnValue);
      if (rc != 0) e.args += std::string(e.args.empty() ? "" : ",") + "\"error\":" + std::to_string(rc);
    }
  }

  std::lock_guard<std::mutex> lock(gMutex);
  gData.api.push_back(std::move(e));
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  *buffer = static_cast<uint8_t *>(std::aligned_alloc(8, kActivityBufferSize));
  *size = *buffer ? kActivityBufferSize : 0;
  *maxNumRecords = 0;
}

// Older record versions are used deliberately: newer versions only append
// fields, so these layouts read correctly on CUDA 11 and 12 alike.
void handleRecord(const CUpti_Activity *record) {
  GpuEvent e;
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      auto *k = reinterpret_cast<const CUpti_ActivityKernel4 *>(record);
      e = {GpuKind::Kernel, demangle(k->name), k->deviceId, k->streamId, k->correlationId, k->start, k->end, {}};
      e.args = "\"grid\":[" + std::to_string(k->gridX) + "," + std::to_string(k->gridY) + "," +
               std::to_string(k->gridZ) + "],\"block\":[" + std::to_string(k->blockX) + "," +
               std::to_string(k->blockY) + "," + std::to_string(k->blockZ) +
               "],\"registersPerThread\":" + std::to_string(k->registersPerThread) +
               ",\"staticSharedMem\":" + std::to_string(k->staticSharedMemory) +
               ",\"dynamicSharedMem\":" + std::to_string(k->dynamicSharedMemory);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      auto *m = reinterpret_cast<const CUpti_ActivityMemcpy4 *>(record);
      e = {GpuKind::Memcpy, std::string("Memcpy ") + memcpyKindName(m->copyKind), m->deviceId, m->streamId,
           m->correlationId, m->start, m->end, {}};
      e.args = "\"bytes\":" + std::to_string(m->bytes);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
      auto *m = reinterpret_cast<const CUpti_ActivityMemset2 *>(record);
      e = {GpuKind::Memset, "Memset", m->deviceId, m->streamId, m->correlationId, m->start, m->end, {}};
      e.args = "\"bytes\":" + std::to_string(m->bytes) + ",\"value\":" + std::to_string(m->value);
      break;
    }
    default:
      return;
  }
  std::lock_guard<std::mutex> lock(gMutex);
  gData.gpu.push_back(std::move(e));
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t, size_t validSize) {
  CUpti_Activity *record = nullptr;
  while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) handleRecord(record);

  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped) == CUPTI_SUCCESS && dropped) {
    std::lock_guard<std::mutex> lock(gMutex);
    gData.droppedRecords += dropped;
  }
  std::free(buffer);
}

std::string processName() {
  std::ifstream comm("/proc/self/comm");
  std::string name;
  std::getline(comm, name);
  return name.empty() ? "process" : name;
}

void finish() {
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  cuptiUnsubscribe(gSubscriber);

  std::lock_guard<std::mutex> lock(gMutex);
  if (!writeChromeTrace(gData, gOutputPath)) {
    std::fprintf(stderr, "[cudatrace] could not write %s\n", gOutputPath.c_str());
    return;
  }
  if (!gQuiet) {
    std::fprintf(stderr, "[cudatrace] wrote %s (%zu API calls, %zu GPU operations", gOutputPath.c_str(),
                 gData.api.size(), gData.gpu.size());
    if (gData.droppedRecords) std::fprintf(stderr, ", %zu records dropped", gData.droppedRecords);
    std::fprintf(stderr, ")\n  open it at https://ui.perfetto.dev\n");
  }
}

bool initialize() {
  if (gInitialized.exchange(true)) return true;

  gQuiet = envFlag("CUDATRACE_QUIET");
  gData.pid = static_cast<uint32_t>(getpid());
  gData.processName = processName();
  gData.droppedRecords = 0;
  const char *out = std::getenv("CUDATRACE_OUTPUT");
  gOutputPath = out && *out ? out : "cudatrace_" + std::to_string(gData.pid) + ".json";

  CUPTI_CHECK(cuptiGetTimestamp(&gData.baseTimestamp));
  CUPTI_CHECK(cuptiSubscribe(&gSubscriber, apiCallback, nullptr));
  CUPTI_CHECK(cuptiEnableDomain(1, gSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  if (envFlag("CUDATRACE_DRIVER")) CUPTI_CHECK(cuptiEnableDomain(1, gSubscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  CUPTI_CHECK(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));

  std::atexit(finish);
  if (!gQuiet) std::fprintf(stderr, "[cudatrace] tracing pid %u -> %s\n", gData.pid, gOutputPath.c_str());
  return true;
}

// LD_PRELOAD path. When loaded through CUDA_INJECTION64_PATH the driver calls
// InitializeInjection instead, so skip the constructor to avoid initializing
// CUPTI from inside cuInit's dlopen.
__attribute__((constructor)) void preloadInit() {
  if (!std::getenv("CUDA_INJECTION64_PATH")) initialize();
}

}  // namespace
}  // namespace cudatrace

extern "C" __attribute__((visibility("default"))) int InitializeInjection(void) {
  return cudatrace::initialize() ? 1 : 0;
}
