
add_executable(Boost_bench Boost.cpp)
target_link_libraries(Boost_bench PRIVATE benchmark::benchmark)
add_executable(InvariantMass_bench InvariantMass.cpp)
target_link_libraries(InvariantMass_bench PRIVATE benchmark::benchmark)

if (cuda)
  add_executable(BoostCuda_bench Boost.cu)
  set_target_properties(BoostCuda_bench PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(BoostCuda_bench PUBLIC GenVectorX SYCLMath VecOps)
  target_compile_options(BoostCuda_bench PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(BoostCuda_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(InvariantMassCuda_bench InvariantMass.cu)
  set_target_properties(InvariantMassCuda_bench PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(InvariantMassCuda_bench PUBLIC GenVectorX SYCLMath VecOps)
  target_compile_options(InvariantMassCuda_bench PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(InvariantMassCuda_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

