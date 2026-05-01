# Emergence Proposals — Making the Blob Look Like a Plant

Forward-looking design document. The current CPU pipeline produces a hemispherical
blob on top of the soil — there are no forces that select for vertical structure or
branching. This doc enumerates candidate mechanisms (each grounded in real plant
biology), with pros, cons, and expected impact on plant lifetime.

> Goal: plants should *emerge* from the environment, not be hard-coded.
> Each proposal below adds an environmental pressure that local cells respond to —
> the global shape is a consequence of stacked pressures, not a rule.
>
> **Emergence rule.** No `if-else` behavioral rules of the form "if condition X then
> do Y" that directly produce the desired plant-like outcome. Mechanisms must be
> environmental forces, scalar fields, or local cell-state dynamics whose
> interaction *produces* the outcome as a side effect. Anything below that fails
> this test is filtered out in §3.

---

## 1. Why we get a blob today

Diagnosing the missing forces, given what's already implemented:

* **Reproduction direction is uniform random.** [`RandomNeighborReproduction`](../src/simulation/cpu/stages/RandomNeighborReproduction.cpp) picks any empty neighbor with equal probability → isotropic spread.
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

## 3. Filtered out (not emergent)

These were considered but violate the emergence rule. Recorded so we don't
relitigate them.

* **A3 — Gravitropism / "up bias".** A constant directional preference unrelated to any sensed quantity. Real plants sense gravity via statoliths *and* integrate it with light gradients; the upward orientation is itself emergent from those, not a primitive. Use **A1** (responds to a real field) instead.
* **A4 — Stability requirement.** "Must have support to grow" is a hard if-else. Replaced by **D1** (gravity-fall): unsupported tiles physically drop, so supported topology persists by selection rather than by rule.
* **B3 — Explicit root/shoot/leaf typing.** A label that prescribes behavior. Functional differentiation is already emergent: a cell on soil naturally absorbs minerals, a cell in light naturally photosynthesizes. No type field needed.
* **E2 — Stochastic disturbance.** Externally imposed; doesn't arise from the environment's dynamics. May be useful as a stress-test later, not part of the core emergence story.

---

## 4. Prioritization

Ordered by **how much each mechanism breaks the blob**, given that earlier
mechanisms are already in place. "Impact" is the change in silhouette;
"foundational?" marks whether later items depend on it.

| Priority | Mechanism                       | Impact | Foundational?  | Effort  |
| :------: | ------------------------------- | :----: | :------------: | :-----: |
| **P0**   | **A1** Phototropism             | huge   | —              | small   |
| **P0**   | **B1** Apical meristem aging    | huge   | needed for B2  | small   |
| **P0**   | **B2** Apical dominance (auxin) | huge   | depends on B1  | medium  |
| **P1**   | **D1** Gravity fall             | medium | —              | small   |
| **P1**   | **C1** Vascular distance        | medium | —              | medium  |
| **P2**   | **E1** Multi-seed competition   | huge*  | needs genetics | large   |
| **P3**   | **A2** Crowding penalty         | small  | —              | trivial |
| **P3**   | **C2** Source/sink transport    | small  | —              | medium  |
| **P3**   | **C3** Long-distance signaling  | small  | —              | medium  |
| **P3**   | **D2** Cantilever / moment      | medium | depends on D1  | large   |

\* E1 has huge impact only over many simulation steps (selection pressure
accumulates); single-step impact is small.

**Impact, in one line each:**

* **A1** blob → upright (light becomes the growth-direction signal)
* **B1** omnidirectional → frontier-only (only young cells reproduce)
* **B2** spire → branched (auxin suppression releases lateral buds far from the apex)
* **D1** kills overhangs (unsupported tiles fall)
* **C1** caps tree height, balances root ↔ canopy (transport cost grows with distance)
* **E1** selection pressure for height (neighbors shading neighbors)
* **A2** smooths growth fronts
* **C2** focuses growth at tips (sink-driven sugar flow)
* **C3** plant-wide coordination via signal fields
* **D2** self-pruning of old branches (moment exceeds support)

### Why this order

#### P0 — the three forces that turn a blob into a plant

These three mechanisms together cover all three properties the current
simulation lacks (vertical structure, branching topology, mortality). All three
are responses to real physical fields or local cell-state dynamics — none of
them is a hardcoded rule about "where plants should grow".

1. **A1 Phototropism** — direction selection becomes proportional to the
   `light` field at each candidate. *Why this is emergent:* nothing in the rule
   says "grow up". Plants only grow toward light because, in the current world,
   light comes from above. Place a light source on the side and the plants
   bend sideways. Place the world in a cave and they don't grow up at all.
   This is the smallest patch with the largest visual impact.
2. **B1 Apical meristem aging** — add an `age` field per plant cell.
   Reproduction probability decays with age (or: a stochastic mature event
   makes a cell stop reproducing). *Why this is emergent:* every cell still
   follows the same rule; the localization of growth to the frontier emerges
   from the fact that newly-spawned cells are the only young ones. Without
   this, every interior cell competes for division and the blob refills itself
   faster than maintenance can prune it.
3. **B2 Apical dominance** — add an `auxin` scalar field. Young/meristematic
   cells emit auxin; auxin diffuses through the plant network with decay;
   reproduction probability is multiplied by `1 / (1 + auxin/k)`. *Why this is
   emergent:* this is *the* canonical biological mechanism for branching
   architecture. Lateral buds far from any active tip naturally fall below the
   suppression threshold and become new growth zones; the geometry of the
   plant (and not a rule) decides where branches form. Removing the apex
   releases suppression and triggers regrowth — exactly as in real plants.

After P0, plants should already look recognizably plant-like: a stem with a
clear growth tip, branches when the plant is large enough, and a finite
lifecycle that ends when no auxin source remains.

#### P1 — physical realism

4. **D1 Gravity fall** — bottom-up sweep moves unsupported plant/dead tiles
   one cell down. *Why this is emergent:* gravity is a real environmental force
   in this world (light comes from up; gravity goes down). It punishes
   horizontal overgrowth without an "is this stable?" rule.
5. **C1 Vascular distance penalty** — compute (or approximate via diffusion) a
   `distanceFromSoil` field on the plant network; maintenance cost scales with
   it (cavitation/friction analog). *Why this is emergent:* tall plants
   biologically pay more to push water up. With this, the equilibrium height
   of a plant is a function of its root system size, not a constant. Trees
   stop growing taller because growth no longer pays for itself, not because
   of a hardcoded cap.

#### P2 — ecosystem

6. **E1 Multi-seed competition** — initialize N seeds with parameter variation
   (when [src/genetics/](../src/genetics/) is back online). *Why this is
   emergent:* the original `README.md` vision. With many seeds, the selection
   pressure for vertical growth comes from neighbors-shading-neighbors rather
   than from any individual plant's behavior. This is also where DNA can
   finally bias things like "phototropism strength" or "auxin decay rate" and
   evolve plants that look different.

#### P3 — refinements

The remaining options are small wins after P0–P2 are in place. Implement only
if the silhouette still has a specific deficiency that one of them addresses.

---

## 5. Lifetime impact summary

How plant lifetime evolves as each priority tier is added on top of the
previous ones:

* **Today (blob)** — Plants grow until resources run out, then die back. Uniform random reproduction outpaces maintenance death.
* **+ A1** — *Shorter* lifetime. Plants overgrow their water budget and die from unbounded upward expansion with no internal limit.
* **+ A1 + B1** — *Finite* lifetime. Plant stops growing once tips mature; slowly senesces from accumulated maintenance death. Aging caps total mass produced.
* **+ A1 + B1 + B2** — *Rich lifecycle*: juvenile (single stem) → adolescent (branches release) → mature → senescent. Apex starves and dies → secondary growth → eventually whole plant becomes unsupported.
* **+ D1** — Baseline lifetime unchanged, but punctuated by structural collapse events. Gravity removes overhangs and may fragment plants into surviving pieces.
* **+ C1** — *Bounded equilibrium height*. Plants persist longer at a stable size; tall plants stop expanding when transport cost exceeds photosynthesis gain.
* **+ E1** — Individual lifetimes become *contextual* — they depend on neighbors. Succession dynamics emerge; older plants outcompeted as the canopy fills.

---

## 6. Open questions for the user

* Should branching emerge from **B2 auxin suppression** or from **stochastic
  meristem revival** (a small probability per tick that an aged cell reverts
  to meristematic)? Both are emergent; the auxin route is biologically the
  textbook answer, the revival route is simpler to tune.
* For **C1 vascular distance**: compute via iterative diffusion (cheap,
  approximate) or via a true graph BFS each step (exact, expensive)?
* When **E1 multi-seed** is wired up, do we want all seeds to share parameters
  (testing competition under uniform conditions) or to draw from a parameter
  distribution (testing whether selection picks a winning phenotype)?
