#

./build/benchmarks/Boost_bench --benchmark_out=Boost_bench.json 			   \
							   --benchmark_out_format=json			           \
							   --benchmark_report_aggregates_only=true         \
							   --benchmark_repetitions=3
./build/benchmarks/InvariantMass_bench --benchmark_out=InvMass_bench.json			   \
									   --benchmark_out_format=json		               \
									   --benchmark_report_aggregates_only=true         \
									   --benchmark_repetitions=3

./build/benchmarks/BoostCuda_bench --benchmark_out=Boost_bench.json			   \
							   --benchmark_out_format=json		               \
							   --benchmark_report_aggregates_only=true         \
							   --benchmark_repetitions=3
./build/benchmarks/InvariantMassCuda_bench --benchmark_out=InvMass_bench.json			   \
									   --benchmark_out_format=json		                   \
									   --benchmark_report_aggregates_only=true             \
									   --benchmark_repetitions=3

./build/benchmarks/BoostCudaStreamed_bench --benchmark_out=BoostCudaStreamed_bench.json			   \
							   --benchmark_out_format=json		                                   \
							   --benchmark_report_aggregates_only=true                             \
							   --benchmark_repetitions=3
./build/benchmarks/InvariantMassCudaStreamed_bench --benchmark_out=InvMassCudaStreamed_bench.json	    \
                                                   --benchmark_out_format=json		                    \
                                                   --benchmark_report_aggregates_only=true              \
                                                   --benchmark_repetitions=3
