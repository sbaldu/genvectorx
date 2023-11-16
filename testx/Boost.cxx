#include "Math/Boost.h"
#include "Math/Vector4D.h"
#include <chrono>
#include <vector>

#ifdef SINGLE_PRECISION
using arithmetic_type = float;
#else
using arithmetic_type = double;
#endif

using vec4d =
    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<arithmetic_type>>;
using Boost = ROOT::Math::Boost;

template <class T> using Vector = std::vector<T>; // ROOT::RVec<T>;

vec4d *ApplyBoost(vec4d *lv, Boost bst, const size_t N) {

  Vector<arithmetic_type> invMasses(N);

  vec4d *lvb = new vec4d[N];

  auto start = std::chrono::system_clock::now();

  for (size_t i = 0; i < N; i++) {
    lvb[i] = bst(lv[i]);
  }

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  // std::cout << "cpu time " << duration << " (s)" << std::endl;

  return lvb;
}

vec4d *GenVectors(int n) {

  vec4d *vectors = new vec4d[n];

  // generate n -4 momentum quantities
  for (int i = 0; i < n; ++i) {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

bool print_if_false(const bool assertion, size_t i) {
  if (!assertion) {
    std::cout << "Assertion failed at index " << i << std::endl;
  }
  return assertion;
}

int main(int argc, char **argv) {

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  size_t local_size = 128;

  vec4d *lv = GenVectors(N);

  Boost bst(0.3, 0.4, 0.5);

  vec4d *lvb = ApplyBoost(lv, bst, N);

  // for (size_t i=0; i<N; i++)
  //   assert(print_if_false((std::abs(masses[i] - 2.) <= 1e-5), i) );

  delete[] lvb;
  return 0;
}
