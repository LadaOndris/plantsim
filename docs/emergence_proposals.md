# Emergence Proposals — Making the Blob Look Like a Plant

Forward-looking design document. The current CPU pipeline produces a hemispherical
blob on top of the soil — there are no forces that select for vertical structure or
branching. This doc enumerates candidate mechanisms (each grounded in real plant
biology), with pros, cons, and expected impact on plant lifetime.

> Goal: plants should *emerge* from the environment, not be hard-coded.
> Each proposal below adds an environmental pressure that local cells respond to —
> the global shape is a consequence of stacked pressures, not a rule.

---

## 1. Why we get a blob today

Diagnosing the missing forces, given what's already implemented:

* **Reproduction direction is uniform random.** [`RandomNeighborReproduction`](../src/simulation/cpu/RandomNeighborReproduction.cpp) picks any empty neighbor with equal probability → isotropic spread.
* **Light reaches the whole canopy roughly equally** — at the canopy frontier, the topmost cells are all directly exposed, so there's no sub-pixel bias toward "up" vs. "out".
* **Self-shading penalty is absent.** Lower cells get less light → less sugar → could die from maintenance, but currently `RandomNeighborReproduction` keeps making new cells faster than the bottom dies, and the top cells don't preferentially extend upward.
* **No structural cost.** A cell hanging in air with no support is just as cheap as a cell with a stem under it.
* **No cell differentiation.** Every plant cell tries to reproduce. There's no "tip vs. trunk" distinction — so growth is omnidirectional rather than axial.
* **No competition between plants.** A single seed has no reason to grow tall. (Real plants grow tall *to outcompete neighbors for light*.)
* **No hydraulic / vascular cost.** A cell 50 tiles above the soil pays the same maintenance as a cell touching soil.

Any one of these in isolation will not give plant-like shapes. Several need to stack.

---

## 2. Proposals

Grouped by category. Each proposal is independent — most are designed to combine.

### Category A — Local growth utility (modify the reproduction policy)

The cheapest changes: change *where* a reproducing cell sends its child.

#### A1. Phototropism (light-biased growth)

**Bio:** auxin redistributes to the shaded side of a stem, causing differential
elongation that bends the tip toward light. In our discrete CA, the analog is
choosing the candidate direction with the highest `light` value at the target.

**Mechanism:** in the intention phase, instead of uniform-random over empty neighbors,
weight each direction by `light[target]^k` (or by softmax of light values).
`k` controls how greedy the choice is.

* **Pros:** trivially cheap (we already have `light`). Direct vertical bias.
* **Cons:** alone, produces thin spaghetti spires racing upward. No branching, no
  self-organization, no robustness to noise.
* **Lifetime impact:** plants reach the top quickly, then have nowhere to go and
  starve once they overgrow their water budget. Net lifetime is *shorter* without
  a counter-force.

#### A2. Crowding penalty

**Bio:** real plants compete for personal space (canopy gap dynamics). Buds in
densely-occupied positions have lower marginal benefit.

**Mechanism:** in the utility, subtract `w_crowd * (#plant neighbors at target / 6)`.

* **Pros:** spreads growth fronts; discourages re-filling already-grown areas.
* **Cons:** without other forces, just produces sparse dendritic shapes that still
  grow outward.
* **Lifetime impact:** marginal — slightly slower expansion, slightly less self-shading.

#### A3. Gravitropism / "up bias"

**Bio:** statoliths (starch granules in cells) sense gravity; shoots grow against,
roots grow with. This is the fundamental driver of vertical organization.

**Mechanism:** static directional weighting — directions pointing up get a multiplier.

* **Pros:** very simple, gives the simulation an unambiguous "up". Stems become vertical.
* **Cons:** *too* deterministic. Real plants integrate gravity *plus* light; pure
  gravitropism in a dark cave gives the same result as in sunlight.
* **Lifetime impact:** depends on whether maintenance can be paid up there.

#### A4. Stability requirement

**Bio:** unsupported lateral overhangs would snap under their own weight; only
supported tiles persist.

**Mechanism:** a candidate is *invalid* (or heavily penalized) unless it has a
plant or soil tile in the "down" direction *or* has at least one supporting plant
neighbor that is itself supported. The "is supported" property could diffuse from
soil-touching cells.

* **Pros:** makes vertical columns the only viable structure. With phototropism,
  produces stem-like growth.
* **Cons:** computing transitive support is a graph operation — needs either an
  iterative diffusion (compute "support distance from soil") or a simpler local
  proxy.
* **Lifetime impact:** caps growth shape; plants stop growing upward when the
  trunk becomes too tall to support new top tiles given sugar/water budget.

> **Recommended baseline:** A1 + A2 + A4 combined, using a utility function in
> the intention phase: `score = w_light*L − w_crowd*C + w_support*S − w_unstable`.
> This is essentially the python `_growth` utility, ported.

---

### Category B — Cell differentiation / aging

Cell-level state that distinguishes "growing tip" from "structural body".

#### B1. Apical meristem (only tips reproduce)

**Bio:** in real plants, only undifferentiated meristematic cells divide. Mature
xylem/phloem cells do not. The shoot apex is the active growth zone.

**Mechanism:** add an `age` (or `isMeristem`) field per plant tile. Reproduction
eligibility: `age < maturityAge` *or* `isMeristem`. New children inherit
meristem status; after a few ticks they "mature" and stop reproducing.

* **Pros:** localizes growth to the frontier — no more in-fill of interior space.
  Naturally produces a sharp growth boundary that looks like a real growing tip.
* **Cons:** adds one field; needs initialization for seed cells.
* **Lifetime impact:** produces clear life stages. Once tips mature without
  spawning new tips, the plant *stops growing* — it becomes a finite-lifetime
  organism. The interplay with maintenance death is what gives it an expiration date.

#### B2. Apical dominance (auxin-like inhibition)

**Bio:** the shoot tip releases auxin which is transported basipetally (downward)
through the plant; auxin concentration decays with distance. Lateral buds
within the high-auxin zone are *inhibited* from sprouting; buds far from any
active tip are *released* and become new branches. This is the canonical
biological mechanism for branching architecture
([Wikipedia: Apical dominance](https://en.wikipedia.org/wiki/Apical_dominance)).

**Mechanism:** add a scalar field `auxin`. Active tips (newly-spawned cells, or
`isMeristem` cells) emit auxin at a constant rate. Auxin diffuses through the
plant network (similar to existing internal transport) and decays per tick.
Reproduction is suppressed where `auxin > threshold`.

* **Pros:** branching becomes *emergent* — you don't pick where branches go,
  the topology of the plant decides for you. When the apex is destroyed
  (e.g., the tip cell dies), nearby suppressed buds become eligible and grow.
* **Cons:** adds a new diffusion field; one more parameter knob (decay rate).
  Requires careful tuning so that small plants are dominated by a single apex
  but larger plants release lateral branches.
* **Lifetime impact:** very rich. Plants exhibit youth (single stem),
  adolescence (lateral branches release), and senescence (apex starves and
  dies, releasing a flush of secondary growth). Closely matches real arboreal
  lifecycles.

#### B3. Cell typing (root vs. shoot vs. leaf)

**Bio:** plants are differentiated into specialized organs. Roots maximize soil
contact; leaves maximize light surface; stems transport.

**Mechanism:** at division, the parent assigns a type to the child based on
context — soil-adjacent cells become "root"; lit cells become "leaf";
intermediate ones become "stem". Each type has different costs/benefits
(roots: high mineral uptake, no photosynthesis; leaves: high photosynthesis,
brittle; stems: cheap maintenance, support structural).

* **Pros:** rich morphology; allows asymmetric architecture (one trunk, many
  leaves, many roots).
* **Cons:** significant complexity; many new parameters.
* **Lifetime impact:** plants become more efficient (specialized cells), so
  longer lifetimes; allows for things like "drop your leaves in winter".

---

### Category C — Resource flow asymmetry

The current internal transport is symmetric diffusion. Real plants have
*directional* flows (xylem up, phloem down) driven by sources and sinks.

#### C1. Hydraulic / vascular distance penalty

**Bio:** delivering water to the canopy requires a continuous water column; tall
plants face increasing transport cost (cavitation risk, friction). Trees
have a height limit governed by this.

**Mechanism:** compute (or approximate via diffusion) `distanceFromSoil` for
each plant cell. Maintenance cost or transport efficiency degrades with this
distance. Alternative: model it as a `transport efficiency` field that decays
with plant-graph distance from soil.

* **Pros:** caps maximum tree height naturally — plants find an optimal height
  given their soil access. Couples canopy size to root system size.
* **Cons:** computing graph distance every step is more expensive than masks.
  An iterative approximation (a few diffusion passes per step) is feasible.
* **Lifetime impact:** plants grow toward an equilibrium height and stay there.
  Death comes from environmental change (neighbor overshadowing, soil
  depletion), not from runaway growth.

#### C2. Sugar source/sink directional transport

**Bio:** phloem actively pumps sugar from sources (lit leaves) to sinks (roots,
new growth). It's not diffusion — it's pressure-driven flow.

**Mechanism:** replace symmetric diffusion in `ResourceTransfer` with a
gradient-following step: `flow_ij = T * (X_i - X_j)` only when positive
(active transport). Sugar accumulates at growing tips.

* **Pros:** new tips have a built-in resource advantage → stronger growth at
  apex than at static body.
* **Cons:** needs careful handling to keep the system stable (no negative
  resources). More expensive per step.
* **Lifetime impact:** more focused growth → individual plants are more
  efficient. Lifetime depends mostly on environmental constraints rather than
  internal resource imbalance.

#### C3. Long-distance signaling field

**Bio:** plants use mobile molecules (cytokinins, strigolactones) to coordinate
across the body — e.g., roots tell shoots how much water is available.

**Mechanism:** one or more scalar fields that propagate through the plant
network and modulate local decisions (reproduction probability, maintenance
demand). Could be: `rootSignal`, `tipSignal`, `stressSignal`.

* **Pros:** allows whole-plant coordination from purely local rules.
* **Cons:** abstract; needs a clear story for what each signal means.
* **Lifetime impact:** plants can adapt — e.g., reduce growth when stressed,
  prolonging lifetime through droughts.

---

### Category D — Structural mechanics

Treat the plant body as a physical object that can fail.

#### D1. Gravity fall (per python)

**Bio:** unsupported cells fall.

**Mechanism:** as in [overall_idea.md](overall_idea.md) — bottom-up sweep,
move plant/dead state down one tile if its "down" neighbor is air.

* **Pros:** simple. Punishes lateral overhangs without explicit instability
  scoring.
* **Cons:** single-direction sweep is a hack; can produce visible "rain" of
  cells from collapsing structures.
* **Lifetime impact:** sudden death events when a structure collapses;
  fragments may regrow as new plants if any tip survives.

#### D2. Cantilever / moment limit

**Bio:** a horizontal branch can only extend so far before its weight overcomes
the holding strength of its base.

**Mechanism:** for each plant cell, accumulate a "moment" = sum of supported
mass × horizontal distance. If a cell's moment exceeds threshold, it breaks
(becomes dead, stuff above falls).

* **Pros:** realistic tree-like silhouettes; explains why real branches don't
  extend infinitely.
* **Cons:** needs graph traversal or iterative pressure propagation. Hardest
  to vectorize cleanly.
* **Lifetime impact:** old plants gradually self-prune; gives lifetime a
  natural ceiling.

---

### Category E — Environmental selection

Don't change individual plants — change the environment so multiple plants
*compete*, and the survivors look like plants.

#### E1. Multi-seed competition

**Bio:** in nature, no plant grows alone — they evolve to outcompete neighbors
for sky.

**Mechanism:** initialize with N seeds spread across the soil. Each plant's
DNA (when [src/genetics/](../src/genetics/) is reactivated) determines its
parameters; light competition selects for tall over short.

* **Pros:** restores the original GA vision (per [README.md](../README.md)).
  Lateral spread becomes wasteful (a competitor next door will overshadow you);
  going up is the only way to win.
* **Cons:** needs the genetics layer to be re-wired. Without it, all plants
  are identical and competition is just stochastic.
* **Lifetime impact:** turns the simulation from "one plant grows" to "an
  ecosystem evolves". Individual plant lifetimes become determined by where
  they sprouted and who their neighbors are.

#### E2. Stochastic disturbance

**Bio:** wind, fire, herbivory — periodic destruction events that reset
ecological succession.

**Mechanism:** with low probability per tick, kill a random patch of plant
cells, or apply a "wind force" that breaks unstable tiles.

* **Pros:** keeps the system from settling into a single attractor; favors
  resilience.
* **Cons:** arbitrary; tuning is taste.
* **Lifetime impact:** caps individual plant lifetime regardless of resources;
  produces succession dynamics.

---

## 3. Recommended stacks

Three combinations, in increasing complexity:

### Stack 1 — Minimal "looks like a plant"
**A1 (phototropism) + A2 (crowding) + A4 (stability) + D1 (gravity fall).**

This is essentially the python `_growth` + `_gravity_fall` ported. Cheapest
viable upgrade. Produces stems that lean toward light, don't fill themselves
in, and don't float in mid-air. Branching is implicit (when the tip is
crowded, a side direction wins).

* Cost: ~2 weeks of CPU work.
* Result: identifiable "stem with leaves at top" silhouette. Plants live until
  light or water budget is exhausted.

### Stack 2 — Branching architecture
**Stack 1 + B1 (meristem aging) + B2 (apical dominance).**

Adds explicit growth-zone localization. Now plants have a clear *life stage*:
juvenile single stem → adolescent lateral branches → mature canopy → senescent.

* Cost: adds two new fields (`age`, `auxin`) and one diffusion-style step.
* Result: tree-like branching that matures and dies. This is where the
  simulation starts to *look like real plants* rather than just oriented blobs.

### Stack 3 — Ecosystem
**Stack 2 + C1 (vascular distance) + E1 (multi-seed competition).**

Adds the long-range coupling: tall plants pay a transport cost; competitive
neighbors create selection pressure. Combined with the genetics layer, this is
the original vision in [README.md](../README.md).

* Cost: requires re-wiring [src/genetics/](../src/genetics/); adds an
  iterative diffusion for `distanceFromSoil`.
* Result: an evolving ecosystem with measurable plant lifetimes, succession,
  and DNA-driven morphology.

---

## 4. Implementation order suggestion

1. **Bring `RandomNeighborReproduction` toward Stack 1 first.** Add the utility
   scoring (A1+A2+A4) as opt-in `Options` flags. Keep the existing
   random-neighbor as a fallback for benchmarking. This is incremental; each
   force can be A/B'd individually.
2. **Add gravity (D1).** Once you have unstable tiles, you need a way to
   resolve them.
3. **Add meristem aging (B1).** Tiny change, big behavioral impact —
   localizes growth to a frontier.
4. **Add apical dominance (B2).** Pays for itself the moment the plant gets
   big enough for branches to matter.
5. **Vascular distance (C1)** and **multi-seed (E1)** come last — they require
   more infrastructure (graph distance / genetics).

---

## 5. Open questions for the user

* Do you want plants to be **mortal by design** (lifetime-bounded) or just
  resource-limited? Affects whether B1 (aging) is core or optional.
* Should branching be **driven by suppression** (B2 auxin) or by **explicit
  structural rules** (e.g., "every Nth tile spawns a side branch")?
  Suppression is biologically real but harder to tune.
* Will the simulation eventually run **multiple plant species** in one world?
  If yes, all per-cell state needs a `speciesId`, which changes the data layout.
