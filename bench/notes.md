# Notes on benchmarking the simulation

Map: 200, 200
Steps: 2000
Optimization: -O3, Release

Base time: 1578 ms
After changing to AoS instead of SoA: 845 ms
Changing vector<bool> to vector<uint8_t>: 746 ms
Swapped for loops (it also changed behaviour): 207 ms

## Notes Lenovo 

./bench/simulation/bench_simulation 2000 200

With branching:
Simulation completed in 1178 ms
Steps/second: 1697.79
Map checksum: 0x71e893f46ffae581

Without branching:
Simulation completed in 3358 ms
Steps/second: 595.593
Map checksum: 0x71e893f46ffae581


Eigen:
Simulation completed in 4292 ms
Steps/second: 465.983
Map checksum: 0x71e893f46ffae581

Eigen with pre-allocated buffers:
Simulation completed in 1297 ms
Steps/second: 1542.02
Map checksum: 0x71e893f46ffae581

Eigen optimized, floats:
Simulation completed in 607 ms
Steps/second: 3294.89
Map checksum: 0xa8cbbbd03ca37680

- the previous, with Intel Core i5-14600KF:
Simulation completed in 242 ms
Steps/second: 8264.46
Map checksum: 0xa8cbbbd03ca37680

GPU backend (NVIDIA GeForce GTX 1660 SUPER):
Simulation completed in 49 ms
Steps/second: 40816.3
Map checksum: 0xa8cbbbd03ca37680

GPU - 100000 steps, 200 grid
Simulation completed in 1961 ms
Steps/second: 50994.4
Map checksum: 0xa8cbbbd03ca37680

CPU - 100000 steps, 200 grid
Simulation completed in 13872 ms
Steps/second: 7208.77
Map checksum: 0xa8cbbbd03ca37680


## Different map (maybe)

- only resource transfer

$ ./build/bin/bench_sim_cuda 2000 200
Simulation completed in 20 ms
Steps/second: 100000
Map checksum: 0xa8cbbbd03ca37680

$ ./build/bin/bench_sim_cuda 100000 200
Simulation completed in 732 ms
Steps/second: 136612
Map checksum: 0xa8cbbbd03ca37680

## Benchmark: resource transfer + cell replication

### CPU

$ ./build/bin/bench_sim_cpu 2000 200
Simulation completed in 1978 ms
Steps/second: 1011.12
Map checksum: 0xa228610c3da694b5

After using expression templates and materializing sooner to avoid repeated calculations:

$ ./build/bin/bench_sim_cpu 2000 200
Simulation completed in 1668 ms
Steps/second: 1199.04
Map checksum: 0xa228610c3da694b5

### GPU

