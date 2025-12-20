
# Plant-growth

## Environment

Step 1 — Encode Physical Directionality (Gravity)

- incurs maintenance cost
- cell dies if it cannot maintain itself

Step 2 — Light as an Occluded Environmental Field


Step 3 — Nutrients as a Diffusive Field

- scalar field with a source region (a lower thick layer - soil), diffusion, depletion, and regeneration
- implemented as a 2D scalar field aligned with the hex grid
- clamped to [0, maxNutrient]
- it is queried by cells and is updated every tick
- nutrients flow from higher concentration to lower concentration between neighboring hex cells
- cells have absorption rate: absorbedAmount = min(absorptionRate, nutrientsAvailable)

Step 4 — Cell Cohesion & Connectivity Constraints

Step 5 — Generic Resource Transport (No Types)

Step 6 — Reproduction as Local Advantage Maximization

Step 7 — Death & Pruning (Essential for Emergence)

