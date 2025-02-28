
add_executable(Boost_bench Boost.cpp)
target_link_libraries(Boost_bench PRIVATE benchmark::benchmark)
add_executable(InvariantMasses_bench InvariantMasses.cpp)
target_link_libraries(InvariantMasses_bench PRIVATE benchmark::benchmark)

if (oneapi)

  add_executable(InvariantMassesSYCL_bench InvariantMassesSYCL.cpp)
  target_link_options(InvariantMassesSYCL_bench PUBLIC -E)
  add_sycl_to_root_target_exe(TARGET InvariantMassesSYCL_bench SOURCES InvariantMassesSYCL.cpp 
                              DEPENDENCIES GenVectorX SYCLMath VecOps benchmark
                              INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                              COMPILE_DEFINITIONS ${sycl_compile_list})

  add_executable(BoostSYCL_bench BoostSYCL.cpp)
  target_link_options(BoostSYCL_bench PUBLIC -E)
  add_sycl_to_root_target_exe(TARGET BoostSYCL_bench SOURCES BoostSYCL.cpp 
                              DEPENDENCIES GenVectorX SYCLMath VecOps benchmark
                              INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                              COMPILE_DEFINITIONS ${sycl_compile_list})

  if (single_precision)
    add_executable(SInvariantMassesSYCL_bench InvariantMassesSYCL.cpp)
    target_link_options(SInvariantMassesSYCL_bench PUBLIC -E)
    add_sycl_to_root_target_exe(TARGET SInvariantMassesSYCL_bench SOURCES InvariantMassesSYCL.cpp 
                                DEPENDENCIES GenVectorX SYCLMath VecOps benchmark
                                INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                                COMPILE_DEFINITIONS ${sycl_compile_list_single})

  
    add_executable(SBoostSYCL_bench BoostSYCL.cpp)
    target_link_options(SBoostSYCL_bench PUBLIC -E)
    add_sycl_to_root_target_exe(TARGET SBoostSYCL_bench SOURCES BoostSYCL.cpp 
                                DEPENDENCIES GenVectorX SYCLMath VecOps benchmark
                                INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                                COMPILE_DEFINITIONS ${sycl_compile_list_single})
  
    endif()

endif()

if (adaptivecpp)
  add_executable(InvariantMassesSYCL_bench InvariantMassesSYCL.cpp)
  add_sycl_to_root_target(TARGET InvariantMassesSYCL_bench SOURCES InvariantMassesSYCL.cpp 
                          DEPENDENCIES GenVectorX SYCLMath VecOps benchmark
                          INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                          COMPILE_DEFINITIONS ${sycl_compile_list})
 

  add_executable(BoostSYCL_bench BoostSYCL.cpp)
  add_sycl_to_root_target(TARGET BoostSYCL_bench SOURCES BoostSYCL.cpp 
                          DEPENDENCIES GenVectorX SYCLMath VecOps benchmark
                          INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                          COMPILE_DEFINITIONS ${sycl_compile_list})
  

  if (single_precision)
    add_executable(SInvariantMassesSYCL_bench InvariantMassesSYCL.cpp)
    add_sycl_to_root_target(TARGET SInvariantMassesSYCL_bench SOURCES InvariantMassesSYCL.cpp 
                            DEPENDENCIES GenVectorX SYCLMath VecOps benchmark
                            INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                            COMPILE_DEFINITIONS ${sycl_compile_list_single}) 


    add_executable(SBoostSYCL_bench BoostSYCL.cpp)
    add_sycl_to_root_target(TARGET SBoostSYCL_bench SOURCES BoostSYCL.cpp 
                            DEPENDENCIES GenVectorX SYCLMath VecOps benchmark
                            INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                            COMPILE_DEFINITIONS ${sycl_compile_list_single})

  endif()

endif()


if (cuda)
  add_executable(BoostCUDA_bench BoostCUDA.cu)
  set_target_properties(BoostCUDA_bench PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(BoostCUDA_bench PUBLIC GenVectorX SYCLMath VecOps)
  target_compile_options(BoostCUDA_bench PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(BoostCUDA_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(BoostCUDA_bench PRIVATE benchmark::benchmark)

  add_executable(BoostCUDAStreamed_bench BoostCUDAStreamed.cu)
  set_target_properties(BoostCUDAStreamed_bench PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(BoostCUDAStreamed_bench PUBLIC GenVectorX SYCLMath VecOps)
  target_compile_options(BoostCUDAStreamed_bench PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(BoostCUDAStreamed_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(BoostCUDAStreamed_bench PRIVATE benchmark::benchmark)

  add_executable(InvariantMassesCUDA_bench InvariantMassesCUDA.cu)
  set_target_properties(InvariantMassesCUDA_bench PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(InvariantMassesCUDA_bench PUBLIC GenVectorX SYCLMath VecOps)
  target_compile_options(InvariantMassesCUDA_bench PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(InvariantMassesCUDA_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(InvariantMassesCUDA_bench PRIVATE benchmark::benchmark)

  add_executable(InvariantMassesCUDAStreamed_bench InvariantMassesCUDAStreamed.cu)
  set_target_properties(InvariantMassesCUDAStreamed_bench PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(InvariantMassesCUDAStreamed_bench PUBLIC GenVectorX SYCLMath VecOps)
  target_compile_options(InvariantMassesCUDAStreamed_bench PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(InvariantMassesCUDAStreamed_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(InvariantMassesCUDAStreamed_bench PRIVATE benchmark::benchmark)
endif()

