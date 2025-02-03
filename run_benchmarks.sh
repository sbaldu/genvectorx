#

./build/benchmarks/Bench_boost --benchmark_out=test.csv 			   \
							   --benchmark_out_format=csv			   \
							   --benchmark_report_aggregates_only=true \
							   --benchmark_repetitions=5
./build/benchmarks/InvariantMass_boost --benchmark_out=test.csv 			   \
									   --benchmark_out_format=csv			   \
									   --benchmark_report_aggregates_only=true \
									   --benchmark_repetitions=5
