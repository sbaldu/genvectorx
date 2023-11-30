#include "SYCLMath/GenVector/MathUtil.h"
#include <cstdio>
#include <chrono>
#include <iostream>

#if defined(ROOT_MATH_SYCL)
#include <sycl/sycl.hpp>
using mode = sycl::access::mode;

#elif defined(ROOT_MATH_CUDA)
#include <cuda_runtime.h>

#define ERRCHECK(err) __checkCudaErrors((err), __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

#endif

namespace ROOT
{

  namespace Experimental
  {

#if defined(ROOT_MATH_CUDA)

    template <class Boost, class LVector>
    __global__ void ApplyBoostKernel(LVector *lv, LVector *lvb, Boost *bst, size_t N)
    {
      int id = blockDim.x * blockIdx.x + threadIdx.x;
      if (id < N)
      {
        Boost bst_loc = bst[0];               //.operator();
        lvb[id] = bst_loc.operator()(lv[id]); // bst(lv[id]);
      }
    }

    template <class Boost, class LVector>
    LVector *ApplyBoost(LVector *lv, Boost bst, const size_t N,
                        const size_t local_size)
    {

      LVector *lvb = new LVector[N];

      LVector *d_lv = NULL;
      ERRCHECK(cudaMalloc((void **)&d_lv, N * sizeof(LVector)));

      // Allocate device input vector
      LVector *d_lvb = NULL;
      ERRCHECK(cudaMalloc((void **)&d_lvb, N * sizeof(LVector)));

      // Allocate the device output vector
      Boost *d_bst = NULL;
      ERRCHECK(cudaMalloc((void **)&d_bst, sizeof(Boost)));

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif

      cudaMemcpy(d_lv, lv, N * sizeof(LVector), cudaMemcpyHostToDevice);
      cudaMemcpy(d_bst, &bst, sizeof(Boost), cudaMemcpyHostToDevice);

      ApplyBoostKernel<<<N / local_size + 1, local_size>>>(d_lv, d_lvb, d_bst, N);

      ERRCHECK(cudaMemcpy(lvb, d_lvb, N * sizeof(LVector), cudaMemcpyDeviceToHost));

#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "cuda time " << duration << " (s)" << std::endl;
#endif
      ERRCHECK(cudaFree(d_lv));
      ERRCHECK(cudaFree(d_lvb));
      ERRCHECK(cudaFree(d_bst));

      return lvb;
    }

    template <class Scalar, class LVector>
    __global__ void InvariantMassKernel(LVector *vec, Scalar *m, size_t N)
    {
      int id = blockDim.x * blockIdx.x + threadIdx.x;
      if (id < N)
      {
        LVector w = vec[id];
        m[id] = w.mass();
      }
    }

    template <class Scalar, class LVector>
    __global__ void InvariantMassesKernel(LVector *v1, LVector *v2, Scalar *m, size_t N)
    {
      int id = blockDim.x * blockIdx.x + threadIdx.x;
      if (id < N)
      {
        LVector w = v1[id] + v2[id];
        m[id] = w.mass();
      }
    }

    template <class Scalar, class LVector>
    Scalar *InvariantMasses(LVector *v1, LVector *v2, const size_t N,
                            const size_t local_size)
    {

      Scalar *invMasses = new Scalar[N];

      // Allocate device input vector
      LVector *d_v1 = NULL;
      ERRCHECK(cudaMalloc((void **)&d_v1, N * sizeof(LVector)));

      // Allocate device input vector
      LVector *d_v2 = NULL;
      ERRCHECK(cudaMalloc((void **)&d_v2, N * sizeof(LVector)));

      // Allocate the device output vector
      Scalar *d_invMasses = NULL;
      ERRCHECK(cudaMalloc((void **)&d_invMasses, N * sizeof(Scalar)));

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif

      cudaMemcpy(d_v1, v1, N * sizeof(LVector), cudaMemcpyHostToDevice);
      cudaMemcpy(d_v2, v2, N * sizeof(LVector), cudaMemcpyHostToDevice);

      InvariantMassesKernel<<<N / local_size + 1, local_size>>>(d_v1, d_v2, d_invMasses, N);

      ERRCHECK(cudaMemcpy(invMasses, d_invMasses, N * sizeof(Scalar), cudaMemcpyDeviceToHost));

#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "cuda time " << duration << " (s)" << std::endl;
#endif
      ERRCHECK(cudaFree(d_v1));
      ERRCHECK(cudaFree(d_v2));
      ERRCHECK(cudaFree(d_invMasses));

      return invMasses;
    }

    template <class Scalar, class LVector>
    Scalar *InvariantMass(LVector *v1, const size_t N, const size_t local_size)
    {

      Scalar *invMasses = new Scalar[N];

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif

      // Allocate the device input vector
      LVector *d_v1 = NULL;
      ERRCHECK(cudaMalloc((void **)&d_v1, N * sizeof(LVector)));

      // Allocate the device output vector
      Scalar *d_invMasses = NULL;
      ERRCHECK(cudaMalloc((void **)&d_invMasses, N * sizeof(Scalar)));
      ERRCHECK(cudaMemcpy(d_v1, v1, N * sizeof(LVector), cudaMemcpyHostToDevice));

      InvariantMassKernel<<<N / local_size + 1, local_size>>>(d_v1, d_invMasses, N);

      ERRCHECK(cudaMemcpy(invMasses, d_invMasses, N * sizeof(Scalar), cudaMemcpyDeviceToHost));

#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "cuda time " << duration << " (s)" << std::endl;
#endif

      ERRCHECK(cudaFree(d_v1));
      ERRCHECK(cudaFree(d_invMasses));

      return invMasses;
    }
#elif defined(ROOT_MATH_SYCL)

    template <class LVector, class Boost>
    LVector *ApplyBoost(LVector *lv, Boost bst, sycl::queue queue, const size_t N,
                        const size_t local_size)
    {

      LVector *lvb = new LVector[N];

      LVector *d_lv = sycl::malloc_device<LVector>(N, queue);
      LVector *d_lvb = sycl::malloc_device<LVector>(N, queue);
      Boost *d_bst = sycl::malloc_device<Boost>(1, queue);

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif

      auto execution_range = sycl::nd_range<1>{
          sycl::range<1>{((N + local_size - 1) / local_size) * local_size},
          sycl::range<1>{local_size}};

      queue.memcpy(d_lv, lv, N * sizeof(LVector));
      queue.memcpy(d_bst, &bst, sizeof(Boost));

      queue.submit([&](sycl::handler &cgh)
                   { cgh.parallel_for(execution_range,
                                      [=](sycl::nd_item<1> item)
                                      {
                                        size_t id = item.get_global_id().get(0);
                                        if (id < N)
                                        {
                                          Boost bst_loc = d_bst[0];                   //.operator();
                                          d_lvb[id] = bst_loc.operator()(d_lv[id]); // bst(lv[id]);
                                        }
                                      }

                     ); });

      queue.memcpy(lvb, d_lvb, N * sizeof(LVector));
      queue.wait();

#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "sycl time " << duration << " (s)" << std::endl;
#endif

      sycl::free(d_lv, queue);
      sycl::free(d_lvb, queue);
      sycl::free(d_bst, queue);
      queue.wait();

      return lvb;
    }

    template <class AccScalar, class AccLVector>
    class InvariantMassKernel
    {
    public:
      InvariantMassKernel(AccLVector acc_vec, AccScalar acc_m, size_t n)
          : vec_acc(acc_vec), m_acc(acc_m), N(n) {}

      void operator()(sycl::nd_item<1> item)
      {
        size_t id = item.get_global_id().get(0);
        if (id < N)
        {
          m_acc[id] = vec_acc[id].mass();
        }
      }

    private:
      AccLVector vec_acc;
      AccScalar m_acc;
      size_t N;
    };

    template <class AccScalar, class AccLVector>
    class InvariantMassesKernel
    {
    public:
      InvariantMassesKernel(AccLVector acc_v1, AccLVector acc_v2, AccScalar acc_m, size_t n)
          : v1_acc(acc_v1), v2_acc(acc_v2), m_acc(acc_m), N(n) {}

      void operator()(sycl::nd_item<1> item) const
      {
        size_t id = item.get_global_id().get(0);
        if (id < N)
        {
          auto w = v1_acc[id] + v2_acc[id];
          m_acc[id] = w.mass();
        }
      }

    private:
      AccLVector v1_acc;
      AccLVector v2_acc;
      AccScalar m_acc;
      size_t N;
    };

    template <class Scalar, class LVector>
    Scalar *InvariantMasses(LVector *v1, LVector *v2, const size_t N,
                            const size_t local_size,
                            sycl::queue queue)
    {

      Scalar *invMasses = new Scalar[N];

      LVector *d_v1 = sycl::malloc_device<LVector>(N, queue);
      LVector *d_v2 = sycl::malloc_device<LVector>(N, queue);
      Scalar *d_invMasses = sycl::malloc_device<Scalar>(N, queue);

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif

      auto execution_range = sycl::nd_range<1>{
          sycl::range<1>{((N + local_size - 1) / local_size) * local_size},
          sycl::range<1>{local_size}};

      queue.memcpy(d_v1, v1, N * sizeof(LVector));
      queue.memcpy(d_v2, v2, N * sizeof(LVector));

      queue.submit([&](sycl::handler &cgh)
                   { cgh.parallel_for(execution_range, InvariantMassesKernel<Scalar *, LVector *>(d_v1, d_v2, d_invMasses, N)); });

      queue.memcpy(invMasses, d_invMasses, N * sizeof(Scalar));
      queue.wait();

#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "sycl time " << duration << " (s)" << std::endl;
#endif

      sycl::free(d_v1, queue);
      sycl::free(d_v2, queue);
      sycl::free(d_invMasses, queue);
      queue.wait();

      return invMasses;
    }

#else

    template <class Scalar, class LVector>
    Scalar *InvariantMasses(const LVector *v1, const LVector *v2, const size_t N)
    {
      Scalar *invMasses = new Scalar[N];
      LVector w;

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif

      for (size_t i = 0; i < N; i++)
      {
        w = v1[i] + v2[i];
        invMasses[i] = w.mass();
      }
#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "cpu time " << duration << " (s)" << std::endl;
#endif
      return invMasses;
    }

    template <class Boost, class LVector>
    LVector *ApplyBoost(LVector *lv, Boost bst, const size_t N)
    {

      LVector *lvb = new LVector[N];

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif
      for (size_t i = 0; i < N; i++)
      {
        lvb[i] = bst(lv[i]);
      }
#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "cpu time " << duration << " (s)" << std::endl;
#endif
      return lvb;
    }

#endif

  } // namespace Experimental
} // namespace ROOT
