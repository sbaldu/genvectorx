#include "SYCLMath/Vector4D.h"
#include <chrono>
#include <vector>
#include <cuda_runtime.h>


#define ERRCHECK(err) __checkCudaErrors((err), __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t code, const char *file,
                                     int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}


#ifdef SINGLE_PRECISION
using arithmetic_type = float;
#else
using arithmetic_type = double;
#endif

using LVectorI =
    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<arithmetic_type>>;
using LVectorO =
    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<arithmetic_type>>;

template <class T>
using Vector = std::vector<T>; // ROOT::RVec<T>;

__global__ void ChangeCoordKernel(LVectorI *lvi, LVectorO *lvo, const size_t N)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N)
  {
    lvo[id] = lvi[id]; // bst(lv[id]);
  }
}

LVectorO *ChangeCoord(LVectorI *lvi,  const size_t N, const size_t local_size)
{

  LVectorO *lvo = new LVectorO[N];

    // Allocate device input vector
  LVectorI *d_lvi = NULL;
  ERRCHECK(cudaMalloc((void **)&d_lvi, N * sizeof(LVectorI)));

  // Allocate device input vector
  LVectorO *d_lvo = NULL;
  ERRCHECK(cudaMalloc((void **)&d_lvo, N * sizeof(LVectorO)));

  auto start = std::chrono::system_clock::now();

  cudaMemcpy(d_lvi, lvi, N * sizeof(LVectorI), cudaMemcpyHostToDevice);
  
  ChangeCoordKernel<<<N / local_size + 1, local_size>>>(d_lvi, d_lvo, N);

  cudaMemcpy(lvo, d_lvo, N * sizeof(LVectorO), cudaMemcpyDeviceToHost);

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "cuda time " << duration << " (s)" << std::endl;

  return lvo;
}

LVectorI *GenVectors(int n)
{

  LVectorI *vectors = new LVectorI[n];

  // generate n -4 momentum quantities
  for (int i = 0; i < n; ++i)
  {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

bool print_if_false(const bool assertion, size_t i)
{
  if (!assertion)
  {
    std::cout << "Assertion failed at index " << i << std::endl;
  }
  return assertion;
}

int main(int argc, char **argv)
{

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  size_t local_size = 128;


  LVectorI *lvi = GenVectors(N);
  LVectorO *lvo = ChangeCoord(lvi, N, local_size);

  for (size_t i = 0; i < N; i++)
    std::cout << lvi[i] << ", " << lvo[i] << std::endl;

  delete[] lvi;
  delete[] lvo;
  return 0;
}
