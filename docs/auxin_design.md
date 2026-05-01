# Auxin Dynamics — Design Document

A unified hormone-based mechanism for emergent plant morphology. Adds a
single hormone field (`auxin`) and a per-cell age field that, together with
the existing light field and minor changes to reproduction, produce
phototropism, apical dominance, and emergent branching from one mechanism.

This document covers the *implementation* of the design discussed in
[emergence_proposals.md §2](emergence_proposals.md). It is intended to be
read together with [implementation_status.md](implementation_status.md) for
context on the existing CPU pipeline.

---

## 1. What this feature is

A new pipeline stage `AuxinDynamics` plus a modified reproduction direction
selection. The mechanism is a discrete-CA analogue of the
**Cholodny–Went hypothesis** (auxin redistributes asymmetrically in
response to environmental cues) plus **Sachs canalization** (cells become
growth points when they net-export auxin from their neighborhood).

### 1.1 Per-cell state additions

Two new arrays in [State.h](../src/simulation/State.h), allocated like
existing fields over the storage grid:

* `plantAge: vector<float>` — simulation time since the cell was born.
  Meaningful only on plant cells. Initialized to 0 on cell creation.
  Increments by `dt` each tick.
* `auxin: vector<float>` — hormone concentration. Meaningful on all cells:
  plant cells synthesize and re-emit; air cells receive incoming flux from
  plant neighbors and let it decay (they do not re-emit). Soil and dead
  cells are excluded from both production and reception.

### 1.2 Per-step processes (the new `AuxinDynamics` stage)

Three sub-passes per `step()`:

**Synthesis.** Auxin is produced by young plant cells. Production rate
decays with age; this is the only emergent definition of "meristem" that
the simulation needs.

```
auxin += dt * synthRate * youngTerm * plantMask
youngTerm = 1 / (1 + age / ageHalfLife)
```

No light/water gate on synthesis — released lateral branches start out
shaded and would otherwise never produce enough auxin to establish
dominance. Maintenance death already removes cells too starved to function.

**Polar transport.** Each direction `d` gets a per-cell *polarity* `P_d`,
combining the local light gradient (Cholodny–Went analogue of PIN
relocalization) and a constant downward bias (statolith analogue of
basipetal flow). Auxin is exported from each plant cell to its neighbors
in proportion to normalized polarities.

```
P_d[C] = max(L[C] - L[neighbor_d(C)], 0) + gravityBias[d]
fraction_d[C] = P_d[C] / sum_d P_d[C]
flux_d[C] = auxin[C] * fraction_d[C] * transportRate * dt * plantMask[C]
auxin -= sum_d flux_d                  (each cell loses what it exports)
auxin += sum_d shifted(flux_d, +d)     (each cell gains what its neighbors exported toward it)
```

`flux_d` is gated by `plantMask` on the *source* but not on the
destination — air cells accumulate auxin where plant cells push it. This
gives a meaningful "auxin at air target" signal that the reproduction
stage uses for tropism.

Two scalar by-products are captured during this pass for use in
reproduction:

* `totalExport[C] = sum_d flux_d[C]`
* `totalInflux[C] = sum_d shifted(flux_d, +d) at C`

**Decay.** First-order degradation, applied to all cells.

```
auxin *= (1 - decayRate * dt)
```

Together, transport rate and decay rate set the spatial reach of any
single auxin source: characteristic length `L ≈ sqrt(transportRate /
decayRate)`. This is the knob that controls "how far behind a growth
point are buds suppressed".

### 1.3 Modified reproduction direction selection

`RandomNeighborReproduction` is patched in two places:

**Reproduction propensity** uses `exportDom = totalExport / (totalExport +
totalInflux + ε)`. A cell that is a net drain of auxin (apex-like) has
`exportDom ≈ 1`; a cell sitting in the basipetal stream from a stronger
source (lateral bud) has `exportDom ≈ 0` and is suppressed. This is what
gives apical dominance.

```
reproductionPropensity = exportDom / (exportDom + exportDomHalfSat) * eligibleMask
```

**Direction weight** combines the brightest accessible air target
(apical-extension signal) with auxin concentration at that air target
(tropism signal):

```
auxinAtTarget[d] = shift(auxin, outgoingShift_d)        // works because air cells hold auxin
tropismWeight[d] = auxinAtTarget[d] / (auxinAtTarget[d] + auxinHalfSat)

dirWeight[d] = (lightAtTarget[d] + α * tropismWeight[d])
             * directionAvailable[d] + ε
```

Direction is sampled with weight `dirWeight[d]` over the 6 directions,
using the same vectorized inverse-CDF trick the existing
intention phase uses (with float weights instead of integer counts).

### 1.4 Why this gives plant-like emergence

| Phenomenon                        | Mechanism                                                                                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Apical dominance                  | Lateral cells in the basipetal stream get high `totalInflux`, low `totalExport` → `exportDom ≈ 0` → suppressed       |
| Bud release on decapitation       | Apex source removed → influx at downstream cells decays → `exportDom` rises → reproduction resumes                   |
| Phototropism                      | Light gradient biases polar transport toward shaded flank; auxin spills into shaded air; tropism weight pulls growth there; cells added on shaded flank → stem bends toward light |
| Apical extension in uniform light | Air directly above the canopy is bright with ~0 auxin (transport is sideways/down); `lightAtTarget` carries growth upward |
| Branching                         | Cells far from the apex see decayed influx; their own synthesis dominates locally; `exportDom` rises; they become new growth points |
| Senescence                        | Cells aged past `ageHalfLife` stop producing auxin; once all cells are old enough, no more reproduction              |

None of these phenomena is special-cased in the code. They all come from
the three sub-passes plus the two-line reproduction patch.

---

## 2. Integration with the existing CPU simulator

### 2.1 Pipeline placement

In [CpuSimulator.cpp](../src/simulation/cpu/CpuSimulator.cpp), insert
`AuxinDynamics` between `ResourceTransfer` (which propagates sugar/water/
mineral through the plant) and `MaintenanceAndDeath` (which ages cells —
see §2.2):

```cpp
void CpuSimulator::step(const Options &options) {
    LightComputation::compute(state, options);
    soilDiffusion.step(state, backBuffer, options);
    soilAbsorption.step(state, backBuffer, options);

    photosynthesis.apply(state, options);
    resourceTransfer.step(state, backBuffer, options);

    auxinDynamics.step(state, backBuffer, options);    // NEW

    maintenanceAndDeath.step(state, backBuffer, options);
    deadDecay.step(state, backBuffer, options);

    reproduction.step(state, backBuffer, options,
                      auxinDynamics.getFlux(),         // NEW: per-direction flux read by reproduction
                      auxinDynamics.getTotalExport(),
                      auxinDynamics.getTotalInflux());
}
```

`AuxinDynamics` exposes its per-direction `flux[d]`, `totalExport`,
`totalInflux` as `const` references for the reproduction stage to read.
No buffer copies; reproduction holds a reference.

### 2.2 Where age is updated

Two tiny edits, no new stage:

* In `MaintenanceAndDeath::step`, after the existing nextAge ← age copy,
  increment by `dt * plantMask`. On death (`deadMask`), set age to 0
  (matches what already happens to other plant fields).
* In `RandomNeighborReproduction::applicationPhase`, when a child is
  created, set `nextAge = (childMask > 0.5) ? 0 : nextAge`.

This avoids creating a separate `AgeUpdate` stage just to do a one-line
increment.

### 2.3 New `Options` knobs

Added to [Options.h](../src/simulation/Options.h):

```cpp
// Auxin
bool   enableAuxinDynamics    = false;
float  auxinSynthRate         = 0.05f;   // per young-plant cell per tick at age 0
float  ageHalfLife            = 20.0f;   // ticks; synthesis halves at this age
float  auxinTransportRate     = 0.30f;   // fraction of auxin exported per tick (max)
float  auxinDecayRate         = 0.05f;   // per tick
float  auxinGravityBias       = 0.20f;   // per "down" direction component
float  auxinDirectionGain     = 1.0f;    // α: weight of tropism vs apical-extension
float  auxinHalfSat           = 0.5f;    // for tropism weight saturation
float  exportDomHalfSat       = 0.3f;    // for reproduction propensity saturation
```

When `enableAuxinDynamics == false` the stage is a no-op and reproduction
falls back to its existing uniform-direction algorithm (`exportDom` and
`auxinAtTarget` are treated as 1 and 0 respectively, recovering the
existing behavior). This preserves the benchmark and lets us A/B test.

### 2.4 New files

```
src/simulation/cpu/stages/AuxinDynamics.h
src/simulation/cpu/stages/AuxinDynamics.cpp
test/simulation/stages/AuxinDynamicsTests.cpp
test/simulation/stages/AuxinReproductionIntegrationTests.cpp
test/simulation/stages/AuxinPipelineIntegrationTests.cpp
```

`State.h` and `Options.h` get fields added; `CpuSimulator.{h,cpp}` get the
new stage member and the call site update; `RandomNeighborReproduction.cpp`
gets the propensity and direction-weight changes.

### 2.5 CUDA / SYCL

Out of scope for v1. CUDA backend is already stale (per
[implementation_status.md](implementation_status.md)) and will be
re-synced in one pass after the CPU version is producing recognizable
plants.

---

## 3. Implementation plan

Six stages, each independently buildable and testable. Stages land in
sequence; each one leaves the simulation in a working state.

### Stage 1 — Add `plantAge` field

* Add `vector<float> plantAge` to [State.h](../src/simulation/State.h),
  zero-initialize in the constructor.
* Wire the increment into `MaintenanceAndDeath` (one line).
* Wire the reset-to-zero into `RandomNeighborReproduction::applicationPhase`
  (one line).
* No behavioral change yet — the field is recorded but unused.

### Stage 2 — Add `auxin` field + `AuxinDynamics::synthesis`

* Add `vector<float> auxin` to State.h, zero-initialized.
* Create `AuxinDynamics` skeleton with only the synthesis pass.
* Wire into `CpuSimulator::step` (gated by `enableAuxinDynamics`).
* Verify: with synthesis only and no transport, auxin accumulates in
  young plant cells and only there.

### Stage 3 — Add polar transport pass

* Polarity computation per direction (light gradient + gravity bias).
* Flux generation gated by `plantMask` on source.
* Apply outflow + inflow accumulation.
* Compute and store `totalExport` and `totalInflux` as members.
* Verify: auxin moves through plant body following the polarity field;
  conservation under zero decay (sum is constant per tick modulo
  synthesis).

### Stage 4 — Add decay pass

* `auxin *= (1 - decayRate * dt)`, max with 0.
* Verify: with no synthesis, auxin → 0 exponentially with the correct
  half-life.

### Stage 5 — Patch reproduction propensity

* In `RandomNeighborReproduction`, accept references to `totalExport` /
  `totalInflux` from `AuxinDynamics`.
* Compute `exportDom` and use it to scale eligibility.
* Verify: a single isolated tip cell still reproduces; a cell receiving
  large influx with no export does not.

### Stage 6 — Patch reproduction direction

* Compute `lightAtTarget[d]` and `auxinAtTarget[d]` per direction.
* Replace the uniform-direction sampler with the float-weighted inverse
  CDF using `dirWeight[d]`.
* Verify end-to-end: a single seed in uniform overhead light grows
  upward (not into a hemispherical blob).

After Stage 6 the feature is complete for v1. Each stage is self-contained
and can be tested before the next one starts.

---

## 4. Tests

Tests live in `test/simulation/stages/` following the existing convention
([SimulationTestHelper.h](../test/simulation/SimulationTestHelper.h)) and
the gtest patterns from
[PhotosynthesisTests.cpp](../test/simulation/stages/PhotosynthesisTests.cpp).

`SimulationTestHelper` will need new accessors: `setAuxin/getAuxin`,
`setPlantAge/getPlantAge`, plus a helper to call `AuxinDynamics::step`
once.

### 4.1 Unit tests — `AuxinDynamicsTests.cpp`

**Synthesis pass**

* `SynthesisProducesAuxinInPlantCells` — single plant cell, age=0, after
  one step `auxin == dt * synthRate`.
* `SynthesisIsZeroForNonPlant` — air, soil, dead cells produce no auxin.
* `SynthesisDecaysWithAge` — parameterized over ages; expect `auxin` after
  one step `≈ dt * synthRate / (1 + age/halfLife)`.
* `SynthesisIgnoresLightAndWater` — verify the activity gate was
  intentionally dropped: same auxin produced regardless of `light` and
  `plantWater`.

**Transport polarity**

* `PolarityZeroInUniformLight` — flat light field, gravity bias = 0:
  polarity is zero in all directions, no auxin moves.
* `PolarityFollowsLightGradient` — set up a light step (left lit, right
  dark): for a cell on the boundary, polarity is highest in the dark
  direction, zero in the lit direction.
* `GravityBiasMovesAuxinDown` — uniform light, nonzero gravity bias:
  auxin migrates to the gravity-down neighbors.
* `AirCellsDoNotReEmit` — seed auxin into an air cell directly; after a
  step it does not export to its neighbors (only decays).

**Transport mass conservation**

* `TransportConservesMassWithoutDecay` — disable synthesis and decay; sum
  of `auxin` over all cells is constant before/after `step`. This catches
  off-by-one shift errors and asymmetric polarity bugs.
* `TotalExportEqualsTotalInfluxGlobally` — sum over all cells of
  `totalExport` equals sum over all cells of `totalInflux` (every emitted
  unit lands somewhere within the grid; padding cells are excluded by the
  validity mask).

**Decay**

* `DecayHalvesAuxinAtHalfLife` — single isolated cell with seeded auxin,
  no synthesis, no transport: after `t = ln(2)/decayRate` ticks,
  `auxin ≈ initial / 2`.
* `DecayPreservesNonNegativity` — large `decayRate * dt` near 1.0 doesn't
  produce negative values.

**Combined three-pass**

* `SteadyStateAtSingleSource` — single isolated young plant cell, all
  three passes enabled: auxin reaches steady state at `synthRate /
  decayRate`.
* `BasipetalGradientFromApex` — vertical line of young plant cells with
  light from above and gravity bias on: auxin concentration is a
  monotonically decreasing function of distance from the top cell.

### 4.2 Unit tests — `RandomNeighborReproductionTests.cpp` (additions)

Existing reproduction tests stay; add:

* `ExportDomZeroSuppresses` — set up a plant cell with `totalInflux >> totalExport`, verify reproduction propensity is ≈ 0.
* `ExportDomOneAllowsReproduction` — pure source cell (no influx, has
  export) reproduces normally.
* `DirectionWeightFollowsLightAtTarget` — uniform auxin field (so tropism
  weight is constant), light asymmetry: reproduction direction is
  weighted toward the brighter air neighbor.
* `DirectionWeightFollowsAuxinAtTarget` — uniform light, but seed auxin in
  one specific air neighbor: reproduction direction is weighted toward
  that neighbor.
* `DirectionWeightCombinesAdditively` — both signals present in opposite
  directions; verify the chosen direction depends on α as expected.
* `FallbackWhenAuxinDisabled` — `enableAuxinDynamics == false`: existing
  uniform behavior is preserved (regression test for the current
  behavior).

### 4.3 Integration tests — `AuxinReproductionIntegrationTests.cpp`

Multi-step scenarios verifying that the phenomena emerge.

* `ApicalDominanceSuppressesLateralCells` — a vertical column of plant
  cells with a young top and older middle/bottom; run `N` steps; assert
  that lateral expansion happens only at the top, not from the middle.
  Specifically: count `Cell` cells in side columns adjacent to the middle
  vs adjacent to the top; top should win by a clear margin.
* `BudReleaseOnDecapitation` — same setup, but kill the top cell after
  the system has reached pseudo-steady. Run `N` more steps; verify that
  cells previously suppressed (just below the new top) become reproductive
  (count of `Cell` cells in lateral positions increases).
* `PhototropicBendingTowardSideLight` — a stem with light artificially
  forced to come from the side (modify `state.light` before the step);
  verify that after `N` ticks the canopy's centroid has shifted toward
  the lit side.
* `ApicalExtensionInUniformLight` — single seed at the soil top, uniform
  overhead light: after `N` ticks the height of the tallest plant cell
  exceeds the lateral spread by some margin (silhouette is taller than
  wide).
* `BranchingFromDistantBuds` — long thin stem; verify that when the
  stem is taller than the auxin reach `L = sqrt(T/λ)`, lateral branches
  emerge from cells near the soil (where influx has decayed below
  threshold).

### 4.4 Integration tests — `AuxinPipelineIntegrationTests.cpp`

Whole-pipeline tests that exercise `AuxinDynamics` together with the
existing stages.

* `FullPipelineRunsWithoutCrash` — enable all stages including auxin, run
  1000 steps on a small grid, verify no NaN/inf in any field.
* `AuxinDoesNotPerturbExistingFieldsWhenDisabled` — with
  `enableAuxinDynamics == false`, the simulation must produce
  byte-identical results to the pre-feature checksum (regression guard).
* `MassConservationOfSugarUnderAuxin` — sugar accounting still balances
  (photosynthesis input − maintenance + reproduction cost = stored sugar)
  with auxin enabled.
* `DeathClearsAuxinAndAge` — when a plant cell dies via maintenance, both
  `auxin` and `plantAge` are reset to 0.

### 4.5 Performance / sanity guards

Bench is currently broken (`SimulatorFactory::create` signature mismatch
per [implementation_status.md](implementation_status.md)), but once
fixed:

* `AuxinDynamics` should add ≤ 50% overhead to a single step on a 200×200
  grid (it does ~6 shifts + a few elementwise ops, comparable to
  `ResourceTransfer`).
* No allocation per step — all matrices preallocated in the constructor.

---

## 5. Implementation stages mapped to verification

| Stage | What lands                                       | Tests that must pass before merge                                                                                                                                                |
| :---: | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `plantAge` field, increment, reset on birth      | `MaintenanceAndDeathTests::AgeIncrementsForPlantCells`, `RandomNeighborReproductionTests::ChildAgeIsZero`                                                                        |
| 2     | `auxin` field, `AuxinDynamics::synthesis` only   | All four `Synthesis*` unit tests; `FallbackWhenAuxinDisabled`                                                                                                                  |
| 3     | Polar transport pass + by-product matrices       | All `Polarity*` and `Transport*` unit tests; `TotalExportEqualsTotalInfluxGlobally`                                                                                              |
| 4     | Decay pass                                       | `DecayHalvesAuxinAtHalfLife`, `DecayPreservesNonNegativity`, `SteadyStateAtSingleSource`, `BasipetalGradientFromApex`                                                            |
| 5     | Reproduction propensity using `exportDom`        | `ExportDomZeroSuppresses`, `ExportDomOneAllowsReproduction`; `ApicalDominanceSuppressesLateralCells`, `BudReleaseOnDecapitation`                                                |
| 6     | Reproduction direction using light + auxin       | `DirectionWeight*` unit tests; `PhototropicBendingTowardSideLight`, `ApicalExtensionInUniformLight`, `BranchingFromDistantBuds`; full `AuxinPipelineIntegrationTests`           |

Each stage has a clear "done" criterion. A stage that breaks an earlier
stage's test gets fixed before the next stage starts. The disabled-flag
behavior (`enableAuxinDynamics == false` reverts to current behavior)
must hold at every stage.

---

## 6. Future extensions

Listed in rough order of expected payoff. None are needed for v1; each
can be added independently if v1 doesn't produce satisfying behavior.

### 6.1 Sachs canalization (positive feedback on transport)

Add a per-direction conductance field `conductance[d]` per cell that
grows under sustained flux and decays slowly otherwise:

```
conductance[d] += dt * (β * flux[d] - λ * conductance[d])
flux[d] *= (1 + γ * conductance[d])    // multiplicative gain
```

This sharpens the basipetal stream into discrete vascular-like channels.
Adds 6 floats per cell and one extra pass per step. **Exit criterion
from v1:** add this if the basipetal stream is observed to leak too
diffusely and apical dominance is not crisp within a few cells of the
apex.

### 6.2 Multi-hormone signaling

Real plants use cytokinins (antagonist to auxin in apical dominance) and
strigolactones (mediator of bud release). Adding a second field that
follows similar dynamics with opposite sign would let us model:

* Cytokinin from roots (soil-adjacent cells synthesize it; it diffuses
  acropetally) → balances auxin in deciding bud fate.
* Strigolactone from stress (cells low on resources synthesize) →
  reinforces apical dominance under nutrient stress.

This is a generalization rather than a fundamental change; each new
hormone is another instance of synthesis + transport + decay.

### 6.3 Light gradient at higher resolution

Current `LightComputation` walks rows top-down per logical column,
producing a vertical light field with no lateral variation beyond what
canopy occlusion creates. For testing tropism scenarios beyond
self-shading, add support for off-axis light sources (a directional sun)
or scattered ambient — these would create real horizontal light gradients
that drive the polarity term.

### 6.4 Per-cell PIN polarity memory

In real plants, PIN proteins persist on a cell face for hours-to-days,
giving the cell *memory* of its preferred export direction. Currently
polarity is recomputed from scratch each tick. A memory term:

```
polarityMemory[d] = (1 - τ) * polarityMemory[d] + τ * P_d_instantaneous
```

would make tropism less twitchy and more biologically faithful (real
phototropic responses take minutes to hours). Adds 6 floats per cell and
trivial update logic.

### 6.5 Coupling auxin to maintenance / death

Currently auxin only affects reproduction. Real plants use auxin as a
cell-survival signal: cells without auxin (e.g., a branch cut off from
the main stem) abscise. Coupling `health -= damage * (1 - auxinTerm)`
in maintenance would let the system prune disconnected fragments
naturally without an explicit "fall and die" rule.

### 6.6 Genetics integration

The original [README.md](../README.md) vision is plants whose DNA encodes
behavior. Once `src/genetics/` is reactivated, per-plant variation in
`auxinSynthRate`, `ageHalfLife`, `auxinDecayRate`, `auxinDirectionGain`,
`auxinGravityBias` would directly produce morphological diversity:
short-half-life plants are bushy, long-half-life plants are tree-like,
high-gravity-bias plants are vertical, low-gravity-bias plants spread
laterally. Selection pressure (multi-seed competition, [emergence_proposals.md
§E1](emergence_proposals.md)) can then evolve plant phenotypes.

### 6.7 CUDA backend re-sync

Once the CPU pipeline is producing recognizable plants, port the entire
new pipeline (auxin + reproduction patches + age field) to CUDA in one
pass alongside re-syncing the existing stages against the modern `State`.
The auxin stage maps cleanly to GPU: synthesis is element-wise, transport
is 6 stencil ops, decay is element-wise. Reproduction's float-weighted
sampler is a small modification of the existing CUDA reproduction kernel.

---

## 7. Open questions (to revisit during implementation)

* **`auxinDirectionGain` (α) magnitude.** Tropism vs apical-extension
  balance. Suspect parameter-sensitive; expose as Options knob and run
  side-by-side comparisons during Stage 6 tests.
* **Gravity bias direction.** In our offset-row hex layout, which 2-3 of
  the 6 directions count as "down"? The python reference picks SW for
  even rows and SW for odd rows (single-direction gravity). Decide
  whether to follow that or use a smoother distribution over multiple
  "down-ish" directions.
* **Initial auxin reset on birth.** Should new cells inherit some auxin
  from the parent (continuity of basipetal stream) or start at 0
  (clean reset)? Default to 0; revisit if tests show parent-stream
  collapse on each division.
* **Whether `plantAge` should also affect maintenance cost.** Cheap
  source of senescence: older cells pay slightly more maintenance. Not
  required for any v1 test, listed for future consideration.
