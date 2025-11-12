
# How to run bench apps

```bash
cmake -DBUILD_MAIN=OFF -DBUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```


For SYCL:

```
cmake -S ~/plantsim/ -B ~/plantsim/build   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_CXX_COMPILER=dpcpp   -DBUILD_BENCH=ON   -DBUILD_MAIN=OF
```

```bash
cmake --build . --target bench_simulation
```

```bash
./bench/simulation/bench_simulation
```


```txt
$ ./bench/sycltest/sycl_test 
Found platform: Intel(R) OpenCL
 Device Intel(R) Corporation 11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz
   Global memory size 3.69748 GiB
   Local memory type global
   Local memory size 256 KiB
   Maximum work items dimesions 3
   Maximum work group size 8192
```