
# Profiling

```sh
perf record -F 999 -o /tmp/plantsim.perf.data -- timeout 10 ./build/release/bin/cpu/plantsim 2>&1 | tail -10

perf report --stdio --no-children -i /tmp/plantsim.perf.data 2>&1 | head -80
```
