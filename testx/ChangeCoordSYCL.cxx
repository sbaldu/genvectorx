#include "SYCLMath/Vector4D.h"
#include <chrono>
#include <vector>
#include <sycl/sycl.hpp>

#ifdef SINGLE_PRECISION
using arithmetic_type = float;
#else
using arithmetic_type = double;
#endif

using LVectorI =
    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<arithmetic_type>>;
using LVectorO =
    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<arithmetic_type>>;

template <class T> using Vector = std::vector<T>; // ROOT::RVec<T>;

LVectorO *ChangeCoord(LVectorI *lvi, sycl::queue queue, const size_t N,  const size_t local_size) {

  LVectorO *lvo = new LVectorO[N];

  auto start = std::chrono::system_clock::now();

 { // Start of scope, ensures data copied back to host
    // Create device buffers. The memory is managed by SYCL so we should NOT
    // access these buffers directly.
    auto execution_range = sycl::nd_range<1>{
        sycl::range<1>{((N + local_size - 1) / local_size) * local_size},
        sycl::range<1>{local_size}};

    sycl::buffer<LVectorI, 1> lvi_sycl(lvi, sycl::range<1>(N));
    sycl::buffer<LVectorO, 1> lvo_sycl(lvo, sycl::range<1>(N));

    queue.submit([&](sycl::handler &cgh) {
      // Get handles to SYCL buffers.
      sycl::accessor lvi_acc{lvi_sycl, cgh, sycl::range<1>(N), sycl::read_only};
      sycl::accessor lvo_acc{lvo_sycl, cgh, sycl::range<1>(N), sycl::write_only};

      cgh.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
                         size_t id = item.get_global_id().get(0);
                         if (id < N) {
                           lvo_acc[id] = lvi_acc[id];//bst(lv[id]);
                         }
                       }
      );
    });
  } // end of scope, ensures data copied back to host
  queue.wait();

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "sycl time " << duration << " (s)" << std::endl;

  return lvo;
}

LVectorI *GenVectors(int n) {

  LVectorI *vectors = new LVectorI[n];

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
  static sycl::queue queue{sycl::default_selector_v};

  std::cout << "sycl::queue check - selected device:\n"
                << queue.get_device().get_info<sycl::info::device::name>()
                << std::endl;

  LVectorI *lvi = GenVectors(N);
  LVectorO *lvo = ChangeCoord(lvi, queue, N, local_size);

  for (size_t i=0; i<N; i++)
    std::cout << lvi[i] << ", " << lvo[i] << std::endl;

  delete[] lvi;
  delete[] lvo;
  return 0;
}
