
add_executable(Boost_bench Boost.cpp)
target_link_libraries(Boost_bench PRIVATE benchmark::benchmark)
add_executable(InvariantMass_bench InvariantMass.cpp)
target_link_libraries(InvariantMass_bench PRIVATE benchmark::benchmark)

