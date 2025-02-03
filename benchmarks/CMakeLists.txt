
add_executable(Boost_bench Boost.cpp)
target_link_libraries(Boost_bench PRIVATE benchmark::benchmark)

