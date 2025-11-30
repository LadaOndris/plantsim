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