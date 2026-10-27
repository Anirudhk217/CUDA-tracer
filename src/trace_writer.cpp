#include "trace_writer.h"

#include <cinttypes>
#include <cstdio>
#include <set>
#include <unordered_map>
#include <utility>

namespace cudatrace {

namespace {

// GPU devices are shown as separate "processes" so their streams group
// together under one heading in the viewer.
constexpr uint32_t kGpuPidBase = 1u << 30;

const char *gpuCategory(GpuKind kind) {
  switch (kind) {
    case GpuKind::Kernel: return "kernel";
    case GpuKind::Memcpy: return "memcpy";
    case GpuKind::Memset: return "memset";
  }
  return "gpu";
}

// Chrome traces use microseconds; keep nanosecond precision as decimals.
void writeTs(FILE *f, const char *key, uint64_t ns) {
  std::fprintf(f, "\"%s\":%" PRIu64 ".%03" PRIu64, key, ns / 1000, ns % 1000);
}

uint64_t rel(uint64_t ts, uint64_t base) { return ts > base ? ts - base : 0; }

}  // namespace

std::string jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

bool writeChromeTrace(const TraceData &data, const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "w");
  if (!f) return false;

  const uint64_t base = data.baseTimestamp;
  bool first = true;
  auto sep = [&] {
    std::fputs(first ? "\n" : ",\n", f);
    first = false;
  };

  std::fputs("{\"displayTimeUnit\":\"ns\",\"otherData\":{\"tool\":\"cudatrace\"", f);
  std::fprintf(f, ",\"droppedRecords\":%zu},\"traceEvents\":[", data.droppedRecords);

  // Process and thread names.
  sep();
  std::fprintf(f,
               "{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%u,\"args\":{\"name\":\"%s (pid %u)\"}}",
               data.pid, jsonEscape(data.processName).c_str(), data.pid);
  sep();
  std::fprintf(f, "{\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":%u,\"args\":{\"sort_index\":0}}",
               data.pid);

  std::set<uint32_t> threads;
  for (const auto &e : data.api) threads.insert(e.tid);
  for (uint32_t tid : threads) {
    sep();
    std::fprintf(f,
                 "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%u,\"tid\":%u,\"args\":{\"name\":\"CPU thread %u\"}}",
                 data.pid, tid, tid);
  }

  std::set<std::pair<uint32_t, uint32_t>> streams;
  std::set<uint32_t> devices;
  for (const auto &e : data.gpu) {
    devices.insert(e.deviceId);
    streams.insert({e.deviceId, e.streamId});
  }
  for (uint32_t dev : devices) {
    sep();
    std::fprintf(f, "{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%u,\"args\":{\"name\":\"GPU %u\"}}",
                 kGpuPidBase + dev, dev);
    sep();
    std::fprintf(f, "{\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":%u,\"args\":{\"sort_index\":%u}}",
                 kGpuPidBase + dev, dev + 1);
  }
  for (const auto &s : streams) {
    sep();
    std::fprintf(f,
                 "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%u,\"tid\":%u,\"args\":{\"name\":\"Stream %u\"}}",
                 kGpuPidBase + s.first, s.second, s.second);
  }

  // Host API calls. Remember where each correlation id was issued so GPU work
  // can be linked back to the call that launched it.
  std::unordered_map<uint32_t, const ApiEvent *> byCorrelation;
  for (const auto &e : data.api) {
    sep();
    std::fprintf(f, "{\"ph\":\"X\",\"cat\":\"cuda_%s\",\"name\":\"%s\",\"pid\":%u,\"tid\":%u,", e.domain,
                 jsonEscape(e.name).c_str(), data.pid, e.tid);
    writeTs(f, "ts", rel(e.start, base));
    std::fputc(',', f);
    writeTs(f, "dur", e.end > e.start ? e.end - e.start : 0);
    std::fprintf(f, ",\"args\":{\"correlation\":%u%s%s}}", e.correlationId, e.args.empty() ? "" : ",",
                 e.args.c_str());
    // Runtime calls take priority over the driver calls they make internally.
    auto it = byCorrelation.find(e.correlationId);
    if (it == byCorrelation.end() || std::string(it->second->domain) != "runtime")
      byCorrelation[e.correlationId] = &e;
  }

  // Device activity, plus a flow arrow from the launching API call.
  for (const auto &e : data.gpu) {
    const uint32_t gpuPid = kGpuPidBase + e.deviceId;
    sep();
    std::fprintf(f, "{\"ph\":\"X\",\"cat\":\"%s\",\"name\":\"%s\",\"pid\":%u,\"tid\":%u,", gpuCategory(e.kind),
                 jsonEscape(e.name).c_str(), gpuPid, e.streamId);
    writeTs(f, "ts", rel(e.start, base));
    std::fputc(',', f);
    writeTs(f, "dur", e.end > e.start ? e.end - e.start : 0);
    std::fprintf(f, ",\"args\":{\"correlation\":%u,\"device\":%u,\"stream\":%u%s%s}}", e.correlationId,
                 e.deviceId, e.streamId, e.args.empty() ? "" : ",", e.args.c_str());

    auto it = byCorrelation.find(e.correlationId);
    if (e.correlationId == 0 || it == byCorrelation.end()) continue;
    const ApiEvent &api = *it->second;
    sep();
    std::fprintf(f, "{\"ph\":\"s\",\"cat\":\"launch\",\"name\":\"launch\",\"id\":%u,\"pid\":%u,\"tid\":%u,",
                 e.correlationId, data.pid, api.tid);
    writeTs(f, "ts", rel(api.start, base));
    std::fputs("}", f);
    sep();
    std::fprintf(f,
                 "{\"ph\":\"f\",\"bp\":\"e\",\"cat\":\"launch\",\"name\":\"launch\",\"id\":%u,\"pid\":%u,\"tid\":%u,",
                 e.correlationId, gpuPid, e.streamId);
    writeTs(f, "ts", rel(e.start, base));
    std::fputs("}", f);
  }

  std::fputs("\n]}\n", f);
  return std::fclose(f) == 0;
}

}  // namespace cudatrace
