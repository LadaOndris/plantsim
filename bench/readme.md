
# How to run bench apps

```bash
cmake -DBUILD_MAIN=OFF -DBUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```


For SYCL:

```
cmake -S ~/plantsim/ -B ~/plantsim/build   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_CXX_COMPILER=dpcpp   -DBUILD_BENCH=ON   -DBUILD_MAIN=OF
```