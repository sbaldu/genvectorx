#include "SYCLMath/GenVector/MathUtil.h"
#include <cstdio>
#include <chrono>
#include <iostream>

namespace ROOT {
  namespace Experimental {

    template <class Boost, class LVector>
    __global__ void ApplyBoostKernel(LVector* lv, LVector* lvb, Boost* bst, size_t N) {
      int id = blockDim.x * blockIdx.x + threadIdx.x;
      if (id < N) {
        Boost bst_loc = bst[0];                //.operator();
        lvb[id] = bst_loc.operator()(lv[id]);  // bst(lv[id]);
      }
    }

    template <class Boost, class LVector>
    LVector* ApplyBoost(LVector* lv, Boost bst, const size_t N, const size_t local_size) {
      LVector* lvb = new LVector[N];

      LVector* d_lv = NULL;
      ERRCHECK(hipMalloc((void**)&d_lv, N * sizeof(LVector)));

      // Allocate device input vector
      LVector* d_lvb = NULL;
      ERRCHECK(hipMalloc((void**)&d_lvb, N * sizeof(LVector)));

      // Allocate the device output vector
      Boost* d_bst = NULL;
      ERRCHECK(hipMalloc((void**)&d_bst, sizeof(Boost)));

      hipMemcpy(d_lv, lv, N * sizeof(LVector), cudaMemcpyHostToDevice);
      hipMemcpy(d_bst, &bst, sizeof(Boost), cudaMemcpyHostToDevice);

      ApplyBoostKernel<<<N / local_size + 1, local_size>>>(d_lv, d_lvb, d_bst, N);

      ERRCHECK(hipMemcpy(lvb, d_lvb, N * sizeof(LVector), cudaMemcpyDeviceToHost));

      ERRCHECK(hipFree(d_lv));
      ERRCHECK(hipFree(d_lvb));
      ERRCHECK(hipFree(d_bst));

      return lvb;
    }

    template <class Boost, class LVector>
    LVector* ApplyBoost(LVector* lv,
                        Boost bst,
                        const size_t N,
                        hitStream_t stream,
                        const size_t local_size) {
      LVector* lvb = new LVector[N];

      LVector* d_lv = NULL;
      ERRCHECK(hipMallocAsync((void**)&d_lv, N * sizeof(LVector), stream));

      // Allocate device input vector
      LVector* d_lvb = NULL;
      ERRCHECK(hipMallocAsync((void**)&d_lvb, N * sizeof(LVector), stream));

      // Allocate the device output vector
      Boost* d_bst = NULL;
      ERRCHECK(hipMallocAsync((void**)&d_bst, sizeof(Boost), stream));

      hipMemcpyAsync(d_lv, lv, N * sizeof(LVector), cudaMemcpyHostToDevice, stream);
      hipMemcpyAsync(d_bst, &bst, sizeof(Boost), cudaMemcpyHostToDevice, stream);

      ApplyBoostKernel<<<N / local_size + 1, local_size, 0, stream>>>(
          d_lv, d_lvb, d_bst, N);

      ERRCHECK(hipMemcpyAsync(
          lvb, d_lvb, N * sizeof(LVector), cudaMemcpyDeviceToHost, stream));

      ERRCHECK(hipFreeAsync(d_lv));
      ERRCHECK(hipFreeAsync(d_lvb));
      ERRCHECK(hipFreeAsync(d_bst));

      return lvb;
    }

    template <class Scalar, class LVector>
    __global__ void InvariantMassKernel(LVector* vec, Scalar* m, size_t N) {
      int id = blockDim.x * blockIdx.x + threadIdx.x;
      if (id < N) {
        LVector w = vec[id];
        m[id] = w.mass();
      }
    }

    template <class Scalar, class LVector>
    __global__ void InvariantMassesKernel(LVector* v1, LVector* v2, Scalar* m, size_t N) {
      int id = blockDim.x * blockIdx.x + threadIdx.x;
      if (id < N) {
        LVector w = v1[id] + v2[id];
        m[id] = w.mass();
      }
    }

    template <class Scalar, class LVector>
    Scalar* InvariantMasses(LVector* v1,
                            LVector* v2,
                            const size_t N,
                            const size_t local_size) {
      Scalar* invMasses = new Scalar[N];

      // Allocate device input vector
      LVector* d_v1 = NULL;
      ERRCHECK(hipMalloc((void**)&d_v1, N * sizeof(LVector)));

      // Allocate device input vector
      LVector* d_v2 = NULL;
      ERRCHECK(hipMalloc((void**)&d_v2, N * sizeof(LVector)));

      // Allocate the device output vector
      Scalar* d_invMasses = NULL;
      ERRCHECK(hipMalloc((void**)&d_invMasses, N * sizeof(Scalar)));

      hipMemcpy(d_v1, v1, N * sizeof(LVector), cudaMemcpyHostToDevice);
      hipMemcpy(d_v2, v2, N * sizeof(LVector), cudaMemcpyHostToDevice);

      InvariantMassesKernel<<<N / local_size + 1, local_size>>>(
          d_v1, d_v2, d_invMasses, N);

      ERRCHECK(
          hipMemcpy(invMasses, d_invMasses, N * sizeof(Scalar), cudaMemcpyDeviceToHost));

      ERRCHECK(hipFree(d_v1));
      ERRCHECK(hipFree(d_v2));
      ERRCHECK(hipFree(d_invMasses));

      return invMasses;
    }

    template <class Scalar, class LVector>
    Scalar* InvariantMasses(LVector* v1,
                            LVector* v2,
                            const size_t N,
                            hipStream_t stream,
                            const size_t local_size) {
      Scalar* invMasses = new Scalar[N];

      // Allocate device input vector
      LVector* d_v1 = NULL;
      ERRCHECK(hipMallocAsync((void**)&d_v1, N * sizeof(LVector), stream));

      // Allocate device input vector
      LVector* d_v2 = NULL;
      ERRCHECK(hipMallocAsync((void**)&d_v2, N * sizeof(LVector), stream));

      // Allocate the device output vector
      Scalar* d_invMasses = NULL;
      ERRCHECK(hipMallocAsync((void**)&d_invMasses, N * sizeof(Scalar), stream));

      hipMemcpyAsync(d_v1, v1, N * sizeof(LVector), cudaMemcpyHostToDevice, stream);
      hipMemcpyAsync(d_v2, v2, N * sizeof(LVector), cudaMemcpyHostToDevice, stream);

      InvariantMassesKernel<<<N / local_size + 1, local_size>>>(
          d_v1, d_v2, d_invMasses, N);

      ERRCHECK(hipMemcpyAsync(
          invMasses, d_invMasses, N * sizeof(Scalar), cudaMemcpyDeviceToHost, stream));

      ERRCHECK(hipFreeAsync(d_v1, stream));
      ERRCHECK(hipFreeAsync(d_v2, stream));
      ERRCHECK(hipFreeAsync(d_invMasses, stream));

      return invMasses;
    }

    template <class Scalar, class LVector>
    Scalar* InvariantMass(LVector* v1, const size_t N, const size_t local_size) {
      Scalar* invMasses = new Scalar[N];

      // Allocate the device input vector
      LVector* d_v1 = NULL;
      ERRCHECK(hipMalloc((void**)&d_v1, N * sizeof(LVector)));

      // Allocate the device output vector
      Scalar* d_invMasses = NULL;
      ERRCHECK(hipMalloc((void**)&d_invMasses, N * sizeof(Scalar)));
      ERRCHECK(hipMemcpy(d_v1, v1, N * sizeof(LVector), cudaMemcpyHostToDevice));

      InvariantMassKernel<<<N / local_size + 1, local_size>>>(d_v1, d_invMasses, N);

      ERRCHECK(
          hipMemcpy(invMasses, d_invMasses, N * sizeof(Scalar), cudaMemcpyDeviceToHost));

      ERRCHECK(hipFree(d_v1));
      ERRCHECK(hipFree(d_invMasses));

      return invMasses;
    }

    template <class Scalar, class LVector>
    Scalar* InvariantMass(LVector* v1,
                          const size_t N,
                          hipStream_t stream,
                          const size_t local_size) {
      Scalar* invMasses = new Scalar[N];

      // Allocate the device input vector
      LVector* d_v1 = NULL;
      ERRCHECK(hipMallocAsync((void**)&d_v1, N * sizeof(LVector), stream));

      // Allocate the device output vector
      Scalar* d_invMasses = NULL;
      ERRCHECK(hipMallocAsync((void**)&d_invMasses, N * sizeof(Scalar), stream));
      ERRCHECK(
          hipMemcpyAsync(d_v1, v1, N * sizeof(LVector), cudaMemcpyHostToDevice, stream));

      InvariantMassKernel<<<N / local_size + 1, local_size>>>(d_v1, d_invMasses, N);

      ERRCHECK(hipMemcpyAsync(
          invMasses, d_invMasses, N * sizeof(Scalar), cudaMemcpyDeviceToHost, stream));

      ERRCHECK(hipFreeAsync(d_v1, stream));
      ERRCHECK(hipFreeAsync(d_invMasses, stream));

      return invMasses;
    }

  }  // namespace Experimental
}  // namespace ROOT
