
add_executable(Boost_bench Boost.cpp)
target_link_libraries(Boost_bench PRIVATE benchmark::benchmark)
add_executable(InvariantMasses_bench InvariantMasses.cpp)
target_link_libraries(InvariantMasses_bench PRIVATE benchmark::benchmark)

if (oneapi)

  add_executable(InvariantMassesSYCL_bench InvariantMassesSYCL.cpp)
  target_link_options(InvariantMassesSYCL_bench PUBLIC -E)
  add_sycl_to_root_target_exe(TARGET InvariantMassesSYCL_bench SOURCES InvariantMassesSYCL.cpp 
                              DEPENDENCIES GenVectorX SYCLMath VecOps 
                              INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                              COMPILE_DEFINITIONS ${sycl_compile_list})

  add_executable(BoostSYCL_bench BoostSYCL.cpp)
  target_link_options(BoostSYCL_bench PUBLIC -E)
  add_sycl_to_root_target_exe(TARGET BoostSYCL_bench SOURCES BoostSYCL.cpp 
                              DEPENDENCIES GenVectorX SYCLMath VecOps 
                              INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                              COMPILE_DEFINITIONS ${sycl_compile_list})

  if (single_precision)
    add_executable(SInvariantMassesSYCL_bench InvariantMassesSYCL.cpp)
    target_link_options(SInvariantMassesSYCL_bench PUBLIC -E)
    add_sycl_to_root_target_exe(TARGET SInvariantMassesSYCL_bench SOURCES InvariantMassesSYCL.cpp 
                                DEPENDENCIES GenVectorX SYCLMath VecOps 
                                INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                                COMPILE_DEFINITIONS ${sycl_compile_list_single})

  
    add_executable(SBoostSYCL_bench BoostSYCL.cpp)
    target_link_options(SBoostSYCL_bench PUBLIC -E)
    add_sycl_to_root_target_exe(TARGET SBoostSYCL_bench SOURCES BoostSYCL.cpp 
                                DEPENDENCIES GenVectorX SYCLMath VecOps 
                                INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                                COMPILE_DEFINITIONS ${sycl_compile_list_single})
  
    endif()

endif()

if (adaptivecpp)
  add_executable(InvariantMassesSYCL_bench InvariantMassesSYCL.cxx)
  add_sycl_to_root_target(TARGET InvariantMassesSYCL_bench SOURCES InvariantMassesSYCL.cxx 
                          DEPENDENCIES GenVectorX SYCLMath VecOps 
                          INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                          COMPILE_DEFINITIONS ${sycl_compile_list})
 

  add_executable(BoostSYCL_bench BoostSYCL.cxx)
  add_sycl_to_root_target(TARGET BoostSYCL_bench SOURCES BoostSYCL.cxx 
                          DEPENDENCIES GenVectorX SYCLMath VecOps 
                          INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                          COMPILE_DEFINITIONS ${sycl_compile_list})
  

  if (single_precision)
    add_executable(SInvariantMassesSYCL_bench InvariantMassesSYCL.cxx)
    add_sycl_to_root_target(TARGET SInvariantMassesSYCL_bench SOURCES InvariantMassesSYCL.cxx 
                            DEPENDENCIES GenVectorX SYCLMath VecOps 
                            INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                            COMPILE_DEFINITIONS ${sycl_compile_list_single}) 


    add_executable(SBoostSYCL_bench BoostSYCL.cxx)
    add_sycl_to_root_target(TARGET SBoostSYCL_bench SOURCES BoostSYCL.cxx 
                            DEPENDENCIES GenVectorX SYCLMath VecOps 
                            INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc 
                            COMPILE_DEFINITIONS ${sycl_compile_list_single})

  endif()

endif()


if (cuda)
  add_executable(BoostCuda_bench Boost.cu)
  set_target_properties(BoostCuda_bench PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(BoostCuda_bench PUBLIC GenVectorX SYCLMath VecOps)
  target_compile_options(BoostCuda_bench PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(BoostCuda_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(BoostCuda_bench PRIVATE benchmark::benchmark)

  add_executable(InvariantMassesCuda_bench InvariantMasses.cu)
  set_target_properties(InvariantMassesCuda_bench PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(InvariantMassesCuda_bench PUBLIC GenVectorX SYCLMath VecOps)
  target_compile_options(InvariantMassesCuda_bench PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(InvariantMassesCuda_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(InvariantMassesCuda_bench PRIVATE benchmark::benchmark)
endif()

