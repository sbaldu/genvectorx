#include "Math/Vector4D.h"
#include <chrono>
#include <vector>

#ifdef SINGLE_PRECISION
using arithmetic_type = float;
#else
using arithmetic_type = double;
#endif

using vec4dI =
    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<arithmetic_type>>;
using vec4dO =
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<arithmetic_type>>;

template <class T> using Vector = std::vector<T>; // ROOT::RVec<T>;

vec4dO *ChangeCoord(vec4dI *lvi, const size_t N) {

  vec4dO *lvo = new vec4dO[N];

  auto start = std::chrono::system_clock::now();

  for (size_t i = 0; i < N; i++) {
    lvo[i] = lvi[i];
  }

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "cpu time " << duration << " (s)" << std::endl;

  return lvo;
}

vec4dI *GenVectors(int n) {

  vec4dI *vectors = new vec4dI[n];

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

  vec4dI *lvi = GenVectors(N);


  vec4dO *lvo = ChangeCoord(lvi, N);

  for (size_t i=0; i<N; i++)
    std::cout << lvi[i] << ", " << lvo[i] << std::endl;

  delete[] lvi;
  delete[] lvo;
  return 0;
}
