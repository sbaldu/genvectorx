#include "SYCLMath/Boost.h"
#include "SYCLMath/Vector4D.h"
#include <chrono>
#include <sycl/sycl.hpp>
#include <vector>

#ifdef SINGLE_PRECISION
using Scalar = float;
#else
using Scalar = double;
#endif

using LVector =
    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Scalar>>;
using Boost = ROOT::Experimental::Boost;

template <class T> using Vector = std::vector<T>; // ROOT::RVec<T>;

LVector *ApplyBoost(LVector *lv, Boost bst, sycl::queue queue, const size_t N,
                    const size_t local_size) {

  Vector<Scalar> invMasses(N);

  LVector *lvb = new LVector[N];

#ifdef ROOT_MEAS_TIMING
  auto start = std::chrono::system_clock::now();
#endif

  { // Start of scope, ensures data copied back to host
    // Create device buffers. The memory is managed by SYCL so we should NOT
    // access these buffers directly.
    auto execution_range = sycl::nd_range<1>{
        sycl::range<1>{((N + local_size - 1) / local_size) * local_size},
        sycl::range<1>{local_size}};

    sycl::buffer<LVector, 1> lv_sycl(lv, sycl::range<1>(N));
    sycl::buffer<LVector, 1> lvb_sycl(lvb, sycl::range<1>(N));
    sycl::buffer<Boost, 1> bst_sycl(&bst, sycl::range<1>(1));

    queue.submit([&](sycl::handler &cgh) {
      // Get handles to SYCL buffers.
      sycl::accessor lv_acc{lv_sycl, cgh, sycl::range<1>(N), sycl::read_only};
      sycl::accessor lvb_acc{lvb_sycl, cgh, sycl::range<1>(N),
                             sycl::write_only};
      sycl::accessor bst_acc{bst_sycl, cgh, sycl::range<1>(1),
                             sycl::read_write};
      // auto v1_acc = v1_sycl.get_access<mode::read>(cgh);
      // auto v2_acc = v2_sycl.get_access<mode::read>(cgh);
      // auto m_acc = m_sycl.get_access<mode::write>(cgh);

      cgh.parallel_for(execution_range,
                       [=](sycl::nd_item<1> item) {
                         size_t id = item.get_global_id().get(0);
                         if (id < N) {
                           lvb[id] = bst(lv[id]);
                         }
                       }

      );
    });
  } // end of scope, ensures data copied back to host
  queue.wait();

#ifdef ROOT_MEAS_TIMING
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "sycl time " << duration << " (s)" << std::endl;
#endif

  return lvb;
}

LVector *GenVectors(int n) {

  LVector *vectors = new LVector[n];

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

  LVector *lv = GenVectors(N);

  Boost bst(0.3, 0.4, 0.5);

  static sycl::queue queue{sycl::default_selector_v};

  std::cout << "sycl::queue check - selected device:\n"
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  LVector *lvb = ApplyBoost(lv, bst, queue, N, local_size);

  // for (size_t i=0; i<N; i++)
  //   assert(print_if_false((std::abs(masses[i] - 2.) <= 1e-5), i) );

  delete[] lv;
  delete[] lvb;
  return 0;
}
