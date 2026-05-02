# plantsim

Plantsim is a plant growth simulator with one core constraint: plant shapes are never prescribed—they must **emerge** from environmental physics. Each cell follows purely local rules (absorb water, photosynthesize under light, pay maintenance, reproduce into empty neighbors), and the resulting global structure is a consequence of those pressures compounding, not of any hard-coded directional growth logic.

***This is a work in progress.*** The currently implemented stages of the simulation don't allow for real-plant-like shapes. Instead, the living organism appears as a blob covering the surface of the soil.

![Plant growing from soil with real-time parameter panel](docs/screenshots/example-of-initial-growth-2.png)

## How it works

The simulation runs on a hex grid. Each tick executes a pipeline of stages:

1. **Light** — top-down column walk; intensity attenuates through plant, dead, and soil cells
2. **Soil diffusion** — water and minerals regenerate toward targets and diffuse laterally through soil
3. **Soil absorption** — plant cells absorb resources from soil at the same tile
4. **Photosynthesis** — `sugar += f(light, water)` with Michaelis-Menten saturation; water is consumed as reagent
5. **Resource transfer** — sugar, water, and minerals diffuse through the connected plant network
6. **Maintenance & death** — cells pay sugar and water costs proportional to light (transpiration analog); health degrades on deficit, regenerates when fed; cells at health ≤ 0 die
7. **Dead decay** — dead matter slowly releases back into adjacent soil; empty dead tiles become air
8. **Reproduction** — eligible cells (enough sugar, empty neighbor exists) expand into a neighbor

## Architecture

```text
src/
  simulation/
    cpu/          # CPU backend
    cuda/         # GPU CUDA backend
  rendering/      # Real-time visualization of the simulation
docs/             # Design documents, notes, descriptions
```

## Tech stack

| Layer | Technology |
| --- | --- |
| Language | C++23 |
| Build | CMake |
| Compute (CPU) | Eigen (vectorization) |
| Compute (GPU) | CUDA |
| Rendering | OpenGL 4, GLFW |
| UI | ImGui |
| Testing | GTest |

## How to run

The build is driven by [Conan 2](https://conan.io/). Dependencies are fetched per profile, so a CPU-only build does not require CUDA or oneAPI to be installed.

Install Conan once:
```bash
pip install conan
```

Build the main application (CPU backend):
```bash
conan install . -pr conan/profiles/cpu-release --build=missing
conan build . -pr conan/profiles/cpu-release
./build/Release/bin/cpu/plantsim
```

### Available profiles

Profiles live in [`conan/profiles/`](conan/profiles/). Pick the one that matches what you want to build:

| Profile | Backends | Build type | Extras |
| --- | --- | --- | --- |
| `cpu-release` / `cpu-debug` | CPU | Release / Debug | — |
| `cuda-release` / `cuda-debug` | CUDA | Release / Debug | — |
| `all-release` / `all-debug` | CPU + CUDA | Release / Debug | — |
| `cpu-dev-release` / `cpu-dev-debug` | CPU | Release / Debug | tests + benchmarks |
| `cuda-dev-release` / `cuda-dev-debug` | CUDA | Release / Debug | tests + benchmarks |
| `cpu-release-profiling` | CPU | Release | gprof/perf flags |

### System prerequisites per backend

Conan handles the "normal" libraries (Eigen, glfw, glm, imgui, freetype, gtest, glad). Compute toolchains stay system-installed and are only required for the matching backend:

- **CPU profiles** (`cpu-*`, `all-*`): C++23-capable compiler (gcc 13+ recommended). Nothing else.
- **CUDA profiles** (`cuda-*`, `all-*`): CUDA Toolkit ≥ 12 with `nvcc` on `PATH`, plus a compatible NVIDIA driver.
- **SYCL profiles**: Intel oneAPI Base Toolkit installed with `INTEL_ONEAPI_ROOT` set in the environment.


## Challenges

**1. Modeling plant biology at the right level of abstraction.** Plant structures can be approximated at many levels. At one extreme, explicit conditional rules that directly encode branching behavior, would be fast to implement, but it would prescribe the outcome rather than produce it. At the other extreme, molecular dynamics, gene expression, and intra-cell biochemistry is biologically faithful, but computationally intractable. The challenge is finding the level in between where environmental fields and local cell interactions are rich enough to produce realistic morphology, without hard-coding it.

**2. Finding parameters that allow realistic structures to emerge.** Even with the right mechanisms in place, the parameter space is large and the relationship between parameters and emergent shape is non-linear. Small changes to maintenance costs, resource diffusion rates, or reproduction thresholds can collapse the plant or prevent it from growing at all.
