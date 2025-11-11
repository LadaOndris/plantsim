# Notes on benchmarking the simulation

Map: 200, 200
Steps: 2000
Optimization: -O3, Release

Base time: 1578 ms
After changing to AoS instead of SoA: 845 ms
Changing vector<bool> to vector<uint8_t>: 746 ms


