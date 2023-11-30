#include "SYCLMath/PtEtaPhiM4D.h"
#include "SYCLMath/Vector4D.h"
#include "SYCLMath/VecOps.h"
#include <assert.h>
#include <chrono>
#include <vector>

#ifdef SINGLE_PRECISION
using Scalar = float;
#else
using Scalar = double;
#endif

using LVector =
    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Scalar>>;


LVector* GenVectors(int n)
{
  LVector  *vectors = new LVector[n];

  // generate n -4 momentum quantities
  for (int i = 0; i < n; ++i)
  {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

int main(int argc, char **argv)
{

#ifdef ROOT_MEAS_TIMING
  std::cout << "ROOT_MEAS_TIMING defined \n";
#endif
#ifdef ROOT_MATH_SYCL
  std::cout << "ROOT_MATH_SYCL defined \n";
#endif
#ifdef ROOT_MATH_CUDA
  std::cout << "ROOT_MATH_CUDA defined \n";
#endif
#ifdef SINGLE_PRECISION
  std::cout << "SINGLE_PRECISION defined \n";
#endif


  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  std::string arg2 = argv[2];
  std::size_t nruns = std::stoi(arg2, &pos);
  size_t local_size = 128;

  auto u_vectors = GenVectors(N);
  auto v_vectors = GenVectors(N);


  Scalar *masses = new Scalar(N);

  for (size_t i = 0; i < nruns; i++)
  {
    masses = ROOT::Experimental::InvariantMasses<Scalar, LVector>(u_vectors, v_vectors, N);
  }
  assert((std::abs(masses[0] - 2.) <= 1e-5));

  return 0;
}