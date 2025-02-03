
#include "SYCLMath/Boost.h"
#include "SYCLMath/VecOps.h"
#include "SYCLMath/Vector4D.h"
#include <benchmark/benchmark.hpp>
#include <chrono>
#include <memory>

#ifdef SINGLE_PRECISION
using Scalar = float;
#else
using Scalar = double;
#endif

using LVector =
    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Scalar>>;
using Boost = ROOT::Experimental::Boost;

LVector *GenVectors(int n) {
  auto vectors = std::make_unique<LVector[]>(n);

  // generate n -4 momentum quantities
  std::for_each(vectors.begin(), vectors.end(), [](auto &vec) -> void {
    std::fill(vec.begin(), vec.end(), 1.);
  });

  return std::move(vectors);
}

bool print_if_false(const bool assertion, size_t i) {
  if (!assertion) {
    std::cout << "Assertion failed at index " << i << std::endl;
  }
  return assertion;
}

static void BM_ApplyBoost(benchmark::State &state) {
  for (auto _ : state) {
    auto lvectors = GenVectors(N);
    auto lvectorsboost = std::make_unique<LVector[]>(N);

    Boost bst(0.3, 0.4, 0.5);
    std::transform(lvectors.begin(), lvectors.end(), lvectorsboost.begin(), [](auto& lvec) -> return LVector {
            });
  }
}

BENCHMARK(BM_ApplyBoost)->RangeMultiplier(2)->Range(1 << 10, 1 << 25);

BENCHMARK_MAIN();
