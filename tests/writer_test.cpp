// Checks the trace writer without a GPU: builds a small synthetic trace and
// writes it to the path given on the command line.
#include <cstdio>

#include "../src/trace_writer.h"

using namespace cudatrace;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: writer_test OUTPUT\n");
    return 2;
  }
  TraceData d;
  d.pid = 4242;
  d.processName = "test \"proc\"";
  d.baseTimestamp = 1000000;
  d.droppedRecords = 0;
  d.api.push_back({"cudaMalloc", "runtime", 1, 1, 1001000, 1005500, "\"bytes\":4096"});
  d.api.push_back({"cudaLaunchKernel", "runtime", 1, 2, 1006000, 1007250,
                   "\"kernel\":\"" + jsonEscape("add(float const*, float*)") + "\""});
  d.api.push_back({"cuLaunchKernel", "driver", 1, 2, 1006100, 1007000, ""});
  d.gpu.push_back({GpuKind::Kernel, "add(float const*, float*)", 0, 7, 2, 1008000, 1012345, "\"grid\":[1,1,1]"});
  d.gpu.push_back({GpuKind::Memcpy, "Memcpy HtoD", 0, 7, 0, 1013000, 1014000, "\"bytes\":4096"});
  return writeChromeTrace(d, argv[1]) ? 0 : 1;
}
