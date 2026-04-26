# Implementation Status

Compares the C++ implementation in [src/simulation/](../src/simulation/) against the
Python reference in [overall_idea.md](overall_idea.md). Serves as a TODO list and a
quick reference for what each implemented stage actually does.

> **Spec note.** [overall_idea.md](overall_idea.md) is the working specification.
> This file tracks how close the C++ code is to it, and where it intentionally diverges.

---

## Backend status at a glance

| Backend | State              | What it actually runs                                                                                  |
| ------- | ------------------ | ------------------------------------------------------------------------------------------------------ |
| CPU     | Active development | All stages from the python `step()` except gravity-fall and utility-based growth (uses random neighbor) |
| CUDA    | Stale              | Only `ResourceTransfer` + `RandomNeighborReproduction`, written against an older `State` with a single `resources` field. Does not compile against the current [State.h](../src/simulation/State.h). Will be re-synced once stages are confirmed in CPU. |
| SYCL    | Stub               | `step()` is a `// TODO: implement` ([SyclSimulator.h:21](../src/simulation/sycl/SyclSimulator.h#L21))   |

---

## Stage-by-stage comparison

Stages in the order they appear in the python `PlantSim.step()`.
"CPU" / "CUDA" columns: ✅ implemented for the current `State`, ⚠️ to be revisited,
❌ missing, 🟡 implemented for an old State, not currently wired in.

| # | Stage (python)             | CPU | CUDA | Notes                                                                                                         |
| - | -------------------------- | :-: | :--: | ------------------------------------------------------------------------------------------------------------- |
| 1 | `_soil_regen`              | ✅  | ❌   | Bundled inside `SoilDiffusion::step` (Phase 1).                                                               |
| 2 | `_soil_diffuse`            | ✅  | ❌   | `SoilDiffusion`. Hex 6-neighbor diffusion, soil-only.                                                         |
| 3 | `_compute_light`           | ✅  | ❌   | `LightComputation`. Top-down per logical column.                                                              |
| 4 | `_soil_uptake_into_plants` | ✅  | ❌   | `SoilAbsorption`. **Divergent (intentional):** plant absorbs from soil at the *same* tile (overlap), not from adjacent soil neighbors. |
| 5 | `_photosynthesis`          | ✅  | ❌   | `Photosynthesis`. Adds water cost per sugar produced; capped by available water.                              |
| 6 | `_internal_transport`      | ✅  | 🟡   | `ResourceTransfer` (CPU). CUDA has a `ResourceTransfer` but operates on a single `resources` field on an older `State` — not equivalent.                                  |
| 7 | `_maintenance_and_death`   | ✅  | ❌   | `MaintenanceAndDeath`. Adds **health regeneration when no deficit** (extra vs. python).                       |
| 8 | `_dead_decay`              | ✅  | ❌   | `DeadDecay`. Also flips dead → air when both pools are exhausted (extra vs. python).                          |
| 9 | `_gravity_fall`            | ❌  | ❌   | Not implemented. Python uses single-direction "down" sweep.                                                   |
| 10 | `_growth`                 | ⚠️  | 🟡   | CPU has `RandomNeighborReproduction` (predates the python utility-based design — see [design.md §1](design.md)). CUDA has the same algorithm against the old `State`. **Divergent:** no utility scoring (light, soil-contact, crowding, instability), no `grow_attempt_prob`. May be revisited. |

Pipeline order in CPU is in [CpuSimulator.cpp:51-63](../src/simulation/cpu/CpuSimulator.cpp#L51-L63).
Each stage is gated by an `enableXxx` flag in [Options.h](../src/simulation/Options.h).

---

## How each implemented CPU stage works

Short summaries of the actual behavior in [src/simulation/cpu/](../src/simulation/cpu/).
All stages are vectorized with Eigen `RowMajor` matrices over the **storage grid**
(rectangular padded layout for axial hex coordinates — see [GridTopology.h](../src/simulation/GridTopology.h)).
A precomputed `validityMask` excludes padding cells; a `GridShiftHelper` provides
6-direction outgoing/incoming block shifts.

### Light — [`LightComputation`](../src/simulation/cpu/LightComputation.cpp)

Single static method, runs first each step.

* For each logical column `c`, walk rows top → bottom (row indexing: row `H-1` is sky, row `0` is ground).
* Start with `intensity = options.lightTopIntensity`. Write `state.light[idx] = intensity` at each tile.
* After writing, attenuate: `intensity *= (1 - absorb[type])`. Absorption coefficients from `Options`: `plantLightAbsorb`, `soilLightAbsorb`, `deadLightAbsorb`. Air does not attenuate.
* Result: a full `light` field used by photosynthesis and by maintenance (transpiration term).

### Soil regeneration + diffusion — [`SoilDiffusion`](../src/simulation/cpu/SoilDiffusion.cpp)

One stage that does both python steps. Operates on `soilWater` and `soilMineral` separately, same algorithm.

* **Soil mask** is precomputed once at construction time as the bottom `soilLayerHeight` rows (via the `Initializers` DSL — see [PolicyApplication.h](../src/simulation/initializers/PolicyApplication.h)). It does not adapt if soil tiles change at runtime.
* **Regeneration:** `field += dt * regenRate * (target - field) * soilMask`. Pulls water/mineral toward `soilWaterTarget` / `soilMineralTarget`.
* **Diffusion:** masked hex average. For each of the 6 directions, accumulate `(field * soilMask)` shifted in, and accumulate `soilMask` shifted in to count valid neighbors. `avg = sum / count` (with fallback to self where `count==0`). Then `field += dt * D * (avg - field) * soilMask`.
* Clamp to ≥ 0. Swap front/back buffers.

### Soil absorption (uptake) — [`SoilAbsorption`](../src/simulation/cpu/SoilAbsorption.cpp)

**Diverges from python on purpose:** a plant cell absorbs from the soil *at its own tile*, not from adjacent soil.

* Build a `plantMask` (validity × `cellTypes == Cell`).
* `desiredUptake = plantMask * uptakeRate * dt`.
* `actualUptake = min(desiredUptake, soilResource)` (limited by what's there at the same index).
* `nextSoilResource = soilResource - actualUptake`, `nextPlantResource = plantResource + actualUptake`.
* Run independently for water and mineral. No edge-splitting, no neighbor counting.

### Photosynthesis — [`Photosynthesis`](../src/simulation/cpu/Photosynthesis.cpp)

Adds an explicit water cost per sugar produced — extra vs. python, which only used water as a saturation modifier.

* Mask plant cells.
* Saturation terms: `lightTerm = L / (L + lightHalfSat)`, `waterTerm = W / (W + waterHalfSat)`.
* Potential: `potentialSugar = dt * photoMaxRate * lightTerm * waterTerm * isPlant`.
* Cap by stoichiometry: `maxFromWater = water / waterPerSugar`. Final: `sugarProduced = min(potential, maxFromWater)`.
* Apply: `sugar += sugarProduced`, `water -= sugarProduced * waterPerSugar`.

### Internal transport — [`ResourceTransfer`](../src/simulation/cpu/ResourceTransfer.cpp)

Plant-network diffusion — three independent passes for sugar, water, mineral, with separate transport rates.

* `plantMask = validity * (cellTypes == Cell)`.
* For each resource: accumulate `(resource * plantMask)` and `plantMask` itself across the 6 incoming neighbor shifts to get `neighborSum` and `neighborCount`.
* `avg = (count > 0) ? sum/count : resource` (self-fallback prevents NaN).
* `nextResource = resource + plantMask * dt * T * (avg - resource)`.
* Swap buffers.

### Maintenance & death — [`MaintenanceAndDeath`](../src/simulation/cpu/MaintenanceAndDeath.cpp)

Plant pays per-tick costs in sugar and water. Adds a **health regeneration** path that the python prototype lacks.

* Sugar demand: `dt * sugarMaintCost` (constant per plant tile).
* Water demand: `dt * (waterMaintCost + waterLightLoss * lightTerm)` — transpiration scales with light saturation.
* Deduct demands clamped at 0; deficit = how much demand was unmet.
* `damage = sugarDeficitDamage * sugarDeficit + waterDeficitDamage * waterDeficit`.
* If both deficits ≈ 0: regenerate health toward 1.0 at rate `healthRegenRate * dt` (extra vs. python).
* New health is clamped to `[0, 1]` on plant tiles.
* **Death:** plant tiles with `health <= 0` flip to `Dead`; their `water` and `mineral` move into `deadWater` / `deadMineral`; `sugar`, `water`, `mineral`, `health` are zeroed on the dead tile.

### Dead decay — [`DeadDecay`](../src/simulation/cpu/DeadDecay.cpp)

Releases dead pools into adjacent soil neighbors, then garbage-collects empty dead tiles.

* Build `deadMask` and `soilMask`. Count soil neighbors per dead tile via 6 outgoing shifts.
* `released = min(deadDecayRate * dt * deadPool, deadPool) * deadMask`. Subtract from dead pools.
* Per-direction send: `share = released / soilNeighborCount` (0 where no soil), `mineralShare *= deadToSoilBias`. For each direction, only send to neighbors that are soil (multiply by direction-specific `dirSoilMask`), then shift-accumulate into the soil tile.
* `nextSoilWater += income * soilMask`; same for mineral.
* **Cleanup (extra vs. python):** dead tiles with both pools below `1e-6` flip to `Air`.

### Reproduction (current "growth") — [`RandomNeighborReproduction`](../src/simulation/cpu/RandomNeighborReproduction.cpp)

Three-phase vectorized algorithm. **Diverges from python's `_growth`** — no utility scoring, no instability/crowding penalties, no per-tile attempt probability. Predates the python design; see [design.md §1](design.md).

* **Intention:**
  - `emptyMask` = valid AND (Air OR Soil).
  - For each of the 6 directions, `directionAvailable[d] = shifted(emptyMask)` and `emptyNeighborCount += directionAvailable[d]`.
  - `eligibleMask` = valid AND `cellTypes == Cell` AND `sugar >= reproductionThreshold` AND `emptyNeighborCount > 0`.
  - Pick a uniform random integer in `[0, emptyNeighborCount)` per cell. Walk the 6 directions accumulating availability; the direction whose running count equals the random index is "chosen". Stored as 6 one-hot `directionChosen[d]` matrices.
* **Resolution:** for each direction `d`, shift `directionChosen[d]` to its target. First-claim wins (`tempBuffer *= emptyMask * (1 - childMask)`). Add to `childMask`; shift back to mark the parent in `parentCost`.
* **Application:** `nextSugar -= parentCost * reproductionCost`. For child cells: `cellTypes := Cell`, `sugar := childInitialResources`, `water := childInitialWater`, `health := childInitialHealth`. Swap buffers.

> Note: the eligibility check uses `Air OR Soil` for "empty" — which means cells *can* reproduce into soil tiles. Worth confirming if intentional.

---

## Missing / divergent behaviors (the actual TODO list)

* **Gravity-fall** — no implementation in any backend. Python: bottom-up sweep, falls one tile in the gravity direction if neighbor is air, moving plant or dead state with all its stores.
* **Utility-based growth** — current reproduction is random-neighbor; python scores candidates by `w_light * light + w_soil_contact * contact - w_crowd * crowd - w_unstable * instability + noise` and gates by `grow_attempt_prob`. Decision pending — random-neighbor may stay as the baseline.
* **Adjacency-based soil uptake** — current `SoilAbsorption` uses overlap (kept on purpose). Python uses adjacency with edge-splitting.
* **Dynamic soil mask** — `SoilDiffusion` precomputes the soil region from `soilLayerHeight` once at construction; if soil tiles ever change at runtime (e.g., dead → soil conversion later), the mask will be wrong.
* **CUDA backend re-sync** — bring CUDA up to the current `State` (split sugar/water/mineral/health/dead pools/light, all 8 stages). Currently only the old single-resource flavor of transfer + reproduction exists.
* **SYCL backend** — `step()` body is empty.
* **Bench out of date** — [bench/simulation/run.cpp:78](../bench/simulation/run.cpp#L78) calls `SimulatorFactory::create(initialState)` without the `Options` arg the factory now requires.

---

## Free-form ideas / open design questions

(Section for the user to extend.)

* (e.g.) "Cells should sense local resource gradients before reproducing" — design notes go here.
