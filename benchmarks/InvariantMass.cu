#include "SYCLMath/VecOps.h"
#include "SYCLMath/Vector4D.h"
#include <benchmark/benchmark.h>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>

#ifdef SINGLE_PRECISION
using arithmetic_type = float;
#else
using arithmetic_type = double;
#endif

using vec4d = ROOT::Experimental::LorentzVector<
    ROOT::Experimental::PtEtaPhiM4D<arithmetic_type>>;
template <class T> using Vector = std::vector<T>;

vec4d *GenVectors(int n) {
  vec4d *vectors = new vec4d[n];

  for (int i = 0; i < n; ++i) {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

auto GenVectors(int n) {
  auto vectors = std::make_unique<LVector[]>(n);

  // generate n -4 momentum quantities
  std::for_each(vectors.get(), vectors.get() + n,
                [](auto &vec) -> void { vec = {1., 1., 1., 1.}; });

  return std::move(vectors);
}

void BM_InvariantMass(benchmark::State &state) {
  int count;
  cudaGetDeviceCount(&count);
  cudaSetDevice(count - 1);
  for (auto _ : state) {

    const auto N = state.range(0);
    size_t local_size = 128;

    vec4d *u_vectors = GenVectors(N);
    vec4d *v_vectors = GenVectors(N);

    arithmetic_type *masses = new arithmetic_type[N];

    masses = ROOT::Experimental::InvariantMasses<arithmetic_type, vec4d>(
        u_vectors, v_vectors, N, local_size);
  }
}

BENCHMARK(BM_InvariantMass)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

BENCHMARK_MAIN();
