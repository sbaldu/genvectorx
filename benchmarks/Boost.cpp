
#include "SYCLMath/Boost.h"
#include "SYCLMath/VecOps.h"
#include "SYCLMath/Vector4D.h"
#include <benchmark/benchmark.h>
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

auto GenVectors(int n) {
  auto vectors = std::make_unique<LVector[]>(n);

  // generate n -4 momentum quantities
  std::for_each(vectors.get(), vectors.get() + n, [](auto &vec) -> void {
    vec = {1., 1., 1., 1.};
  });

  return std::move(vectors);
}

static void BM_ApplyBoost(benchmark::State &state) {
  for (auto _ : state) {
    // TODO: take size from state
    auto lvectors = GenVectors(state.range(0));
    auto* lvectorsboost = new LVector[state.range(0)];

    Boost bst(0.3, 0.4, 0.5);
	lvectorsboost = ROOT::Experimental::ApplyBoost(lvectors.get(), bst, state.range(0));

	delete[] lvectorsboost;
  }
}

BENCHMARK(BM_ApplyBoost)->RangeMultiplier(2)->Range(1 << 10, 1 << 25);

BENCHMARK_MAIN();
