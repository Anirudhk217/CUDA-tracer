// Writes collected events as a Chrome Trace Event Format JSON file, which
// opens directly in ui.perfetto.dev and chrome://tracing.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cudatrace {

// A host-side CUDA API call (runtime or driver), timed with the CUPTI clock.
struct ApiEvent {
  std::string name;
  const char *domain;  // "runtime" or "driver"
  uint32_t tid;
  uint32_t correlationId;
  uint64_t start;  // ns
  uint64_t end;    // ns
  std::string args;  // pre-rendered JSON object members, may be empty
};

enum class GpuKind { Kernel, Memcpy, Memset };

// A device-side operation reported by the CUPTI Activity API.
struct GpuEvent {
  GpuKind kind;
  std::string name;
  uint32_t deviceId;
  uint32_t streamId;
  uint32_t correlationId;
  uint64_t start;  // ns
  uint64_t end;    // ns
  std::string args;  // pre-rendered JSON object members, may be empty
};

struct TraceData {
  uint32_t pid;
  std::string processName;
  uint64_t baseTimestamp;  // subtracted from every timestamp
  std::vector<ApiEvent> api;
  std::vector<GpuEvent> gpu;
  size_t droppedRecords;
};

std::string jsonEscape(const std::string &s);

// Returns false if the file could not be written.
bool writeChromeTrace(const TraceData &data, const std::string &path);

}  // namespace cudatrace
