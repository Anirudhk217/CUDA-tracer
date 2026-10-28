# cudatrace

A lightweight CUDA API tracer. Attach it to any CUDA program, with no recompiling, and get a timeline of what the program asked the GPU to do and what the GPU actually did.

It uses NVIDIA's CUPTI library to record:

- **CUDA Runtime API calls** (`cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`, ...) with timing, the calling thread, error codes, and arguments for the most common calls (bytes, copy direction, grid/block size, kernel name).
- **Driver API calls** (`cuLaunchKernel`, ...), optionally.
- **GPU activity**: kernel executions, memcpys and memsets, per device and stream, with launch configuration and register/shared memory usage.

Each GPU operation is linked back to the API call that launched it, so in the viewer you can follow an arrow from `cudaLaunchKernel` on the CPU to the kernel on the GPU.

The output is a [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU) JSON file, so it opens directly in [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing`.

## Requirements

- Linux with an NVIDIA GPU and driver
- CUDA Toolkit 11 or 12 (CUPTI ships with it)
- CMake 3.18+ and a C++17 compiler

## Build

```bash
cmake -B build
cmake --build build -j
ctest --test-dir build        # checks the trace writer, no GPU needed
```

Set `-DCUDATRACE_BUILD_SAMPLES=OFF` to skip building the sample program.

## Use

```bash
./scripts/cudatrace -o vector_add.json -- ./build/vector_add
```

Then open https://ui.perfetto.dev and drag `vector_add.json` onto the page.

The wrapper script only sets environment variables, so you can also do it by hand:

```bash
CUDA_INJECTION64_PATH=$PWD/build/libcudatrace.so ./your_program
# or
LD_PRELOAD=$PWD/build/libcudatrace.so ./your_program
```

It works with anything that uses the CUDA runtime, including Python frameworks:

```bash
./scripts/cudatrace -o torch.json -- python train.py
```

### Options

| Script flag | Environment variable | Effect |
|---|---|---|
| `-o FILE` | `CUDATRACE_OUTPUT=FILE` | Output path (default `cudatrace_<pid>.json`) |
| `--driver` | `CUDATRACE_DRIVER=1` | Also record driver API calls |
| `--quiet` | `CUDATRACE_QUIET=1` | No messages on stderr |

## How it works

- `CUDA_INJECTION64_PATH` tells the CUDA driver to load the library during `cuInit` and call its `InitializeInjection` function. With `LD_PRELOAD`, a constructor does the same setup when the process starts.
- The CUPTI **Callback API** hooks every runtime (and optionally driver) API call on entry and exit, timestamped with `cuptiGetTimestamp` so host and GPU events share one clock.
- The CUPTI **Activity API** delivers GPU-side records (kernels, memcpy, memset) in buffers, which are collected as they complete and flushed at exit.
- At exit, everything is written as a Chrome trace. Each GPU device appears as its own process with one track per stream, and flow events connect each operation to its launching call through CUPTI's correlation ID.

## Limitations

- Linux only.
- Events are kept in memory until the program exits, so very long runs produce large traces.
- If the program exits abnormally (crash, `_exit`, SIGKILL), no trace is written.
- Only one CUPTI client can be active at a time, so it can't run alongside Nsight Systems or another profiler.

## Roadmap

1. ✅ Tracer library with Perfetto-compatible output
2. Validate against real workloads and compare with Nsight Systems
3. Hosted web viewer: drop a trace file in the browser to get a timeline and summary tables
4. Packaging and demo traces
