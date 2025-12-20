Below is a compact but complete “version 1” simulator scaffold that matches the plan we discussed: soil provides diffusing water and minerals, light drives photosynthesis into sugar, plants transport internally (gradient-based, not random), they pay maintenance, they die and recycle, gravity makes unsupported tiles fall, and growth spends stores into new tiles using an environment-driven local utility (light + soil contact + stability – crowding).

It’s written to be clear and hackable rather than maximally optimized.

```Py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ----------------------------
# States (one per tile)
# ----------------------------
AIR  = np.uint8(0)
SOIL = np.uint8(1)
PLANT = np.uint8(2)
DEAD = np.uint8(3)


@dataclass
class Params:
    # Grid + time
    dt: float = 1.0

    # Soil resource targets + regeneration (only applied in SOIL)
    soil_water_target: float = 1.0
    soil_mineral_target: float = 1.0
    soil_water_regen_rate: float = 0.02
    soil_mineral_regen_rate: float = 0.005

    # Soil diffusion rates (applied via a state-dependent diffusivity field)
    soil_water_diffusivity: float = 0.18
    soil_mineral_diffusivity: float = 0.10

    # Uptake from soil per soil->plant edge per tick (max pulled along one adjacency)
    water_uptake_rate: float = 0.08
    mineral_uptake_rate: float = 0.04

    # Light model
    light_top_intensity: float = 1.0
    plant_light_absorb: float = 0.45      # fraction removed when passing a PLANT tile
    dead_light_absorb: float = 0.15       # dead litter absorbs a bit
    soil_light_absorb: float = 0.95       # soil blocks most light

    # Photosynthesis
    photo_max_rate: float = 0.08          # max sugar per tick
    light_half_sat: float = 0.4           # half saturation for incident light
    water_half_sat: float = 0.2           # water dependence

    # Internal transport (diffusion on plant graph)
    sugar_transport: float = 0.18
    water_transport: float = 0.10
    mineral_transport: float = 0.08

    # Maintenance (per PLANT tile per tick)
    sugar_maint_cost: float = 0.03
    water_maint_cost: float = 0.01
    water_light_loss: float = 0.01        # extra water cost scaled by light saturation

    # Health damage from deficits (how fast starvation kills)
    sugar_deficit_damage: float = 2.0
    water_deficit_damage: float = 3.0

    # Growth costs (to create one new PLANT tile)
    grow_sugar_cost: float = 0.35
    grow_water_cost: float = 0.20
    grow_mineral_cost: float = 0.15

    # Growth scoring weights
    w_light: float = 1.2
    w_soil_contact: float = 0.8
    w_crowd: float = 0.6
    w_unstable: float = 1.0
    growth_noise: float = 0.05

    # Growth rate control: per-tile probability of attempting growth each tick
    grow_attempt_prob: float = 0.25

    # Gravity / falling
    enable_gravity: bool = True

    # Dead matter decay (recycling into soil fields)
    dead_decay_rate: float = 0.02         # fraction per tick of dead stores released
    dead_to_soil_bias: float = 1.0        # how strongly dead returns minerals vs stays inert


class HexNeighborsOddR:
    """
    Hex grid in "odd-r" horizontal layout (rows offset on odd rows).
    Grid indexed by (r, c), 0<=r<H, 0<=c<W.

    Neighbors depend on row parity (r % 2).
    Directions order: E, W, NE, NW, SE, SW
    """
    EVEN = ((0, +1), (0, -1), (-1, 0), (-1, -1), (+1, 0), (+1, -1))
    ODD  = ((0, +1), (0, -1), (-1, +1), (-1, 0), (+1, +1), (+1, 0))

    @staticmethod
    def shifts_for_row_parity(parity: int):
        return HexNeighborsOddR.ODD if parity else HexNeighborsOddR.EVEN


def sat(x: np.ndarray, k: float) -> np.ndarray:
    """Saturation function x/(x+k) for x>=0."""
    return x / (x + k + 1e-12)


def shift2d(arr: np.ndarray, dr: int, dc: int, fill: float) -> np.ndarray:
    """
    Shift arr by (dr, dc). Out-of-bounds filled with 'fill'. No wrap-around.
    """
    H, W = arr.shape
    out = np.empty_like(arr)

    r_src0 = max(0, -dr)
    r_src1 = min(H, H - dr)   # exclusive
    r_dst0 = max(0, dr)
    r_dst1 = min(H, H + dr)

    c_src0 = max(0, -dc)
    c_src1 = min(W, W - dc)
    c_dst0 = max(0, dc)
    c_dst1 = min(W, W + dc)

    out.fill(fill)
    out[r_dst0:r_dst1, c_dst0:c_dst1] = arr[r_src0:r_src1, c_src0:c_src1]
    return out


class PlantSim:
    """
    Environment-driven plant-like growth on a 2D hex grid.
    """

    def __init__(self, H: int, W: int, params: Params, seed: int = 0):
        self.H, self.W = H, W
        self.p = params
        self.rng = np.random.default_rng(seed)

        # Tile state
        self.state = np.full((H, W), AIR, dtype=np.uint8)

        # Environmental fields
        self.soil_water = np.zeros((H, W), dtype=np.float32)
        self.soil_mineral = np.zeros((H, W), dtype=np.float32)
        self.light = np.zeros((H, W), dtype=np.float32)

        # Plant internal stores (meaningful where state==PLANT; otherwise kept at 0)
        self.w = np.zeros((H, W), dtype=np.float32)
        self.m = np.zeros((H, W), dtype=np.float32)
        self.s = np.zeros((H, W), dtype=np.float32)
        self.health = np.zeros((H, W), dtype=np.float32)

        # Dead matter stores (recycling pool per dead tile)
        self.dead_water = np.zeros((H, W), dtype=np.float32)
        self.dead_mineral = np.zeros((H, W), dtype=np.float32)

        # Choose a gravity direction consistent with odd-r layout:
        # We'll define "down" as the SW neighbor for even rows and SE neighbor for odd rows,
        # which corresponds to moving generally toward increasing row.
        # (This is a single-direction gravity; you can later extend to more physical support.)
        self._gravity_dir_even = (+1, -1)  # SW for even rows
        self._gravity_dir_odd  = (+1, 0)   # SW for odd rows

    # ---- world setup helpers ----

    def fill_soil_layer(self, thickness_rows: int):
        """Make the bottom 'thickness_rows' rows soil, initialize their fields to targets."""
        r0 = max(0, self.H - thickness_rows)
        self.state[r0:, :] = SOIL
        self.soil_water[r0:, :] = self.p.soil_water_target
        self.soil_mineral[r0:, :] = self.p.soil_mineral_target

    def seed_plant(self, r: int, c: int, sugar: float = 0.2, water: float = 0.2, mineral: float = 0.2):
        """Place a single plant tile."""
        if not (0 <= r < self.H and 0 <= c < self.W):
            return
        if self.state[r, c] != AIR:
            return
        self.state[r, c] = PLANT
        self.s[r, c] = sugar
        self.w[r, c] = water
        self.m[r, c] = mineral
        self.health[r, c] = 1.0

    # ---- core simulation step ----

    def step(self):
        dt = self.p.dt

        self._soil_regen(dt)
        self._soil_diffuse(dt)
        self._compute_light()
        self._soil_uptake_into_plants(dt)
        self._photosynthesis(dt)
        self._internal_transport(dt)
        self._maintenance_and_death(dt)
        self._dead_decay(dt)

        if self.p.enable_gravity:
            self._gravity_fall()

        self._growth(dt)

    # ---- sub-systems ----

    def _soil_regen(self, dt: float):
        is_soil = (self.state == SOIL)
        self.soil_water[is_soil] += dt * self.p.soil_water_regen_rate * (self.p.soil_water_target - self.soil_water[is_soil])
        self.soil_mineral[is_soil] += dt * self.p.soil_mineral_regen_rate * (self.p.soil_mineral_target - self.soil_mineral[is_soil])
        np.maximum(self.soil_water, 0.0, out=self.soil_water)
        np.maximum(self.soil_mineral, 0.0, out=self.soil_mineral)

    def _soil_diffuse(self, dt: float):
        # State-dependent diffusivity: soil diffuses, others effectively do not.
        D_w = np.where(self.state == SOIL, self.p.soil_water_diffusivity, 0.0).astype(np.float32)
        D_m = np.where(self.state == SOIL, self.p.soil_mineral_diffusivity, 0.0).astype(np.float32)

        self.soil_water[:] = self._hex_diffuse(self.soil_water, D_w, dt)
        self.soil_mineral[:] = self._hex_diffuse(self.soil_mineral, D_m, dt)

        np.maximum(self.soil_water, 0.0, out=self.soil_water)
        np.maximum(self.soil_mineral, 0.0, out=self.soil_mineral)

    def _hex_diffuse(self, X: np.ndarray, D: np.ndarray, dt: float) -> np.ndarray:
        """
        X += dt*D*(avg_neighbors(X) - X) with hex neighbors.
        Uses odd-r parity shifts; boundary filled with self (Neumann-ish).
        """
        H, W = X.shape
        sum_nb = np.zeros_like(X)
        # We'll treat missing neighbors as self by filling with X; this avoids boundary sinks.
        for r in range(H):
            parity = r & 1
            shifts = HexNeighborsOddR.shifts_for_row_parity(parity)
            for (dr, dc) in shifts:
                sum_nb[r] += shift2d(X, dr, dc, fill=X[r].mean())[r]
        avg_nb = sum_nb / 6.0
        return X + dt * D * (avg_nb - X)

    def _compute_light(self):
        """
        Directional light from top (r=0) to bottom (r=H-1) per column.
        light[r,c] is the incident light reaching that tile.
        Tiles attenuate transmitted light according to their state.
        """
        H, W = self.H, self.W
        L = np.zeros((H, W), dtype=np.float32)

        for c in range(W):
            I = float(self.p.light_top_intensity)
            for r in range(H):
                L[r, c] = I
                st = self.state[r, c]
                if st == PLANT:
                    I *= (1.0 - self.p.plant_light_absorb)
                elif st == DEAD:
                    I *= (1.0 - self.p.dead_light_absorb)
                elif st == SOIL:
                    I *= (1.0 - self.p.soil_light_absorb)
                else:
                    # AIR: small or no attenuation; keep as is
                    pass

        self.light[:] = L

    def _soil_uptake_into_plants(self, dt: float):
        """
        Plants pull water and minerals from adjacent soil.
        Soil gives at most uptake_rate*dt per plant-adjacent edge, but total give from a soil tile is limited by availability.
        Give is split evenly among adjacent plant tiles.
        """
        plant = (self.state == PLANT)
        soil = (self.state == SOIL)

        # Count plant neighbors for each soil tile
        plant_nb_count = np.zeros((self.H, self.W), dtype=np.float32)

        # For each row parity, aggregate neighbor plant masks via shifts
        for r in range(self.H):
            shifts = HexNeighborsOddR.shifts_for_row_parity(r & 1)
            row_count = np.zeros((self.W,), dtype=np.float32)
            for (dr, dc) in shifts:
                shifted = shift2d(plant.astype(np.float32), dr, dc, fill=0.0)[r]
                row_count += shifted
            plant_nb_count[r] = row_count

        # Soil demand and limited give
        water_demand = (self.p.water_uptake_rate * dt) * plant_nb_count
        mineral_demand = (self.p.mineral_uptake_rate * dt) * plant_nb_count

        water_give = np.where(soil, np.minimum(self.soil_water, water_demand), 0.0).astype(np.float32)
        mineral_give = np.where(soil, np.minimum(self.soil_mineral, mineral_demand), 0.0).astype(np.float32)

        # Soil loses what it gives
        self.soil_water -= water_give
        self.soil_mineral -= mineral_give

        # Distribute to plants: each soil tile splits give among its adjacent plant tiles
        # Contribution from soil to plant along each direction is handled by shifting soil_give and dividing by neighbor count.
        # To avoid division by zero:
        share = np.where(plant_nb_count > 0, 1.0 / plant_nb_count, 0.0).astype(np.float32)

        water_share_from_soil = water_give * share
        mineral_share_from_soil = mineral_give * share

        water_in = np.zeros_like(self.w)
        mineral_in = np.zeros_like(self.m)

        for r in range(self.H):
            shifts = HexNeighborsOddR.shifts_for_row_parity(r & 1)
            for (dr, dc) in shifts:
                # Plant receives from neighboring soil tile: shift soil->plant direction
                water_in[r] += shift2d(water_share_from_soil, -dr, -dc, fill=0.0)[r]
                mineral_in[r] += shift2d(mineral_share_from_soil, -dr, -dc, fill=0.0)[r]

        # Only plants store internal resources
        self.w[plant] += water_in[plant]
        self.m[plant] += mineral_in[plant]

        np.maximum(self.soil_water, 0.0, out=self.soil_water)
        np.maximum(self.soil_mineral, 0.0, out=self.soil_mineral)

    def _photosynthesis(self, dt: float):
        plant = (self.state == PLANT)
        if not np.any(plant):
            return

        light_term = sat(self.light, self.p.light_half_sat)
        water_term = sat(self.w, self.p.water_half_sat)

        ds = dt * self.p.photo_max_rate * light_term * water_term
        self.s[plant] += ds[plant]

    def _internal_transport(self, dt: float):
        """
        Diffusion-like transport of internal stores among connected PLANT tiles.
        This is the key “no random sharing” replacement.
        """
        plant = (self.state == PLANT)
        if not np.any(plant):
            return

        self.s[:] = self._plant_network_diffuse(self.s, self.p.sugar_transport, dt)
        self.w[:] = self._plant_network_diffuse(self.w, self.p.water_transport, dt)
        self.m[:] = self._plant_network_diffuse(self.m, self.p.mineral_transport, dt)

        np.maximum(self.s, 0.0, out=self.s)
        np.maximum(self.w, 0.0, out=self.w)
        np.maximum(self.m, 0.0, out=self.m)

    def _plant_network_diffuse(self, X: np.ndarray, T: float, dt: float) -> np.ndarray:
        """
        For PLANT tiles only:
          X += dt*T*(avg_plant_neighbors(X) - X)
        Non-plant tiles are left unchanged.
        """
        st = self.state
        plant = (st == PLANT)
        H, W = X.shape

        sum_nb = np.zeros_like(X)
        cnt_nb = np.zeros_like(X, dtype=np.float32)

        plant_f = plant.astype(np.float32)

        for r in range(H):
            shifts = HexNeighborsOddR.shifts_for_row_parity(r & 1)
            for (dr, dc) in shifts:
                Xn = shift2d(X, dr, dc, fill=0.0)[r]
                Pn = shift2d(plant_f, dr, dc, fill=0.0)[r]
                sum_nb[r] += Xn * Pn
                cnt_nb[r] += Pn

        avg_nb = np.where(cnt_nb > 0, sum_nb / cnt_nb, X)
        X_new = X.copy()
        X_new[plant] = X[plant] + dt * T * (avg_nb[plant] - X[plant])
        return X_new

    def _maintenance_and_death(self, dt: float):
        plant = (self.state == PLANT)
        if not np.any(plant):
            return

        # Maintenance demands
        dS = dt * self.p.sugar_maint_cost
        # Water maintenance includes a light-linked loss term (transpiration-like)
        light_term = sat(self.light, self.p.light_half_sat)
        dW = dt * (self.p.water_maint_cost + self.p.water_light_loss * light_term)

        # Compute deficits before clamping
        s_before = self.s.copy()
        w_before = self.w.copy()

        self.s[plant] = np.maximum(0.0, self.s[plant] - dS)
        self.w[plant] = np.maximum(0.0, self.w[plant] - dW[plant])

        sugar_deficit = np.maximum(0.0, dS - s_before)
        water_deficit = np.maximum(0.0, dW - w_before)

        damage = (self.p.sugar_deficit_damage * sugar_deficit +
                  self.p.water_deficit_damage * water_deficit)

        self.health[plant] -= damage[plant]

        # Death: convert to DEAD and move remaining stores into dead pools
        dead_mask = plant & (self.health <= 0.0)
        if np.any(dead_mask):
            self.state[dead_mask] = DEAD
            self.dead_water[dead_mask] = self.w[dead_mask]
            self.dead_mineral[dead_mask] = self.m[dead_mask]

            self.w[dead_mask] = 0.0
            self.m[dead_mask] = 0.0
            self.s[dead_mask] = 0.0
            self.health[dead_mask] = 0.0

    def _dead_decay(self, dt: float):
        """
        Dead tiles return minerals (and optionally water) back into adjacent soil fields.
        Minerals are the important recyclable here.
        """
        dead = (self.state == DEAD)
        if not np.any(dead):
            return

        # Amount released this tick
        rel_m = self.p.dead_decay_rate * dt * self.dead_mineral
        rel_w = self.p.dead_decay_rate * dt * self.dead_water

        # Clamp release to what's available
        rel_m = np.minimum(rel_m, self.dead_mineral)
        rel_w = np.minimum(rel_w, self.dead_water)

        self.dead_mineral -= rel_m
        self.dead_water -= rel_w

        # Distribute to adjacent SOIL tiles
        soil = (self.state == SOIL)
        soil_count = np.zeros((self.H, self.W), dtype=np.float32)

        soil_f = soil.astype(np.float32)
        for r in range(self.H):
            shifts = HexNeighborsOddR.shifts_for_row_parity(r & 1)
            count_row = np.zeros((self.W,), dtype=np.float32)
            for (dr, dc) in shifts:
                count_row += shift2d(soil_f, dr, dc, fill=0.0)[r]
            soil_count[r] = count_row

        share = np.where(soil_count > 0, 1.0 / soil_count, 0.0).astype(np.float32)

        # Push released amounts from dead tiles into neighboring soil tiles equally
        m_share = rel_m * share * self.p.dead_to_soil_bias
        w_share = rel_w * share

        add_m = np.zeros_like(self.soil_mineral)
        add_w = np.zeros_like(self.soil_water)

        for r in range(self.H):
            shifts = HexNeighborsOddR.shifts_for_row_parity(r & 1)
            for (dr, dc) in shifts:
                add_m[r] += shift2d(m_share, -dr, -dc, fill=0.0)[r]
                add_w[r] += shift2d(w_share, -dr, -dc, fill=0.0)[r]

        self.soil_mineral[soil] += add_m[soil]
        self.soil_water[soil] += add_w[soil]

    def _gravity_fall(self):
        """
        Simple single-direction gravity: if a PLANT or DEAD tile has AIR in its gravity neighbor, it falls.
        This creates a meaningful penalty for unsupported horizontal overgrowth.
        """
        # Sweep bottom-up so falling doesn't cascade multiple steps in one tick.
        for r in range(self.H - 2, -1, -1):
            parity = r & 1
            dr, dc = self._gravity_dir_odd if parity else self._gravity_dir_even
            rr = r + dr

            if not (0 <= rr < self.H):
                continue

            # c mapping differs per tile due to dc; we handle bounds by slicing
            if dc == 0:
                src_slice = (slice(r, r + 1), slice(0, self.W))
                dst_slice = (slice(rr, rr + 1), slice(0, self.W))
            else:
                # dc = -1: destination is shifted left
                src_slice = (slice(r, r + 1), slice(1, self.W))
                dst_slice = (slice(rr, rr + 1), slice(0, self.W - 1))

            src_state = self.state[src_slice]
            dst_state = self.state[dst_slice]

            movable = (src_state == PLANT) | (src_state == DEAD)
            can_fall = movable & (dst_state == AIR)

            if np.any(can_fall):
                # Move state
                dst_state[can_fall] = src_state[can_fall]
                src_state[can_fall] = AIR

                # Move stores for PLANT
                for arr in (self.w, self.m, self.s, self.health):
                    src_arr = arr[src_slice]
                    dst_arr = arr[dst_slice]
                    dst_arr[can_fall] = src_arr[can_fall]
                    src_arr[can_fall] = 0.0

                # Move stores for DEAD
                for arr in (self.dead_water, self.dead_mineral):
                    src_arr = arr[src_slice]
                    dst_arr = arr[dst_slice]
                    dst_arr[can_fall] = src_arr[can_fall]
                    src_arr[can_fall] = 0.0

                # Write back
                self.state[src_slice] = src_state
                self.state[dst_slice] = dst_state

    def _growth(self, dt: float):
        """
        Stochastic local growth: each plant tile attempts growth with some probability.
        Chooses best adjacent AIR tile by a simple utility:
          + light at target
          + soil adjacency at target
          - crowding at target
          - instability (air below in gravity direction)
          + small noise
        and pays sugar/water/mineral costs.
        """
        p = self.p
        plant_positions = np.argwhere(self.state == PLANT)
        if plant_positions.size == 0:
            return

        self.rng.shuffle(plant_positions)

        for (r, c) in plant_positions:
            # Attempt rate
            if self.rng.random() > p.grow_attempt_prob:
                continue

            # Must afford growth
            if (self.s[r, c] < p.grow_sugar_cost or
                self.w[r, c] < p.grow_water_cost or
                self.m[r, c] < p.grow_mineral_cost):
                continue

            # Evaluate 6 neighbors (odd-r parity)
            shifts = HexNeighborsOddR.shifts_for_row_parity(r & 1)
            best_score = -1e9
            best_pos = None

            for (dr, dc) in shifts:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < self.H and 0 <= cc < self.W):
                    continue
                if self.state[rr, cc] != AIR:
                    continue

                # Light value at candidate
                light_score = self.light[rr, cc]

                # Soil contact: count soil neighbors around the candidate
                soil_contact = 0.0
                cand_shifts = HexNeighborsOddR.shifts_for_row_parity(rr & 1)
                for (dr2, dc2) in cand_shifts:
                    r2, c2 = rr + dr2, cc + dc2
                    if 0 <= r2 < self.H and 0 <= c2 < self.W and self.state[r2, c2] == SOIL:
                        soil_contact += 1.0
                soil_contact /= 6.0

                # Crowding: how many plant neighbors candidate would have
                crowd = 0.0
                for (dr2, dc2) in cand_shifts:
                    r2, c2 = rr + dr2, cc + dc2
                    if 0 <= r2 < self.H and 0 <= c2 < self.W and self.state[r2, c2] == PLANT:
                        crowd += 1.0
                crowd /= 6.0

                # Instability: penalize if the cell would have AIR below it (gravity neighbor)
                gdr, gdc = self._gravity_dir_odd if (rr & 1) else self._gravity_dir_even
                rb, cb = rr + gdr, cc + gdc
                unstable = 1.0
                if 0 <= rb < self.H and 0 <= cb < self.W:
                    unstable = 1.0 if self.state[rb, cb] == AIR else 0.0

                noise = p.growth_noise * (self.rng.random() - 0.5)

                score = (p.w_light * light_score +
                         p.w_soil_contact * soil_contact -
                         p.w_crowd * crowd -
                         p.w_unstable * unstable +
                         noise)

                if score > best_score:
                    best_score = score
                    best_pos = (rr, cc)

            if best_pos is None:
                continue

            # Execute growth
            rr, cc = best_pos
            self.s[r, c] -= p.grow_sugar_cost
            self.w[r, c] -= p.grow_water_cost
            self.m[r, c] -= p.grow_mineral_cost

            self.state[rr, cc] = PLANT
            self.health[rr, cc] = 1.0
            # New tile starts with small stores (you can make these depend on parent / costs)
            self.s[rr, cc] = 0.0
            self.w[rr, cc] = 0.0
            self.m[rr, cc] = 0.0


# ----------------------------
# Minimal example usage
# ----------------------------
if __name__ == "__main__":
    p = Params()
    sim = PlantSim(H=80, W=120, params=p, seed=1)

    sim.fill_soil_layer(thickness_rows=20)

    # Seed a few plant tiles just above soil
    sim.seed_plant(r=59, c=40, sugar=0.2, water=0.2, mineral=0.2)
    sim.seed_plant(r=59, c=80, sugar=0.2, water=0.2, mineral=0.2)

    for _ in range(2000):
        sim.step()

    # At this point you can visualize sim.state, sim.light, sim.soil_* etc.
```

What’s included (and why it addresses your “blob/moss” issue)

This scaffold includes the smallest set of coupled pressures that reliably move you away from “cover the entire soil”:

Directional light + occlusion (_compute_light) so plants shade themselves and competitors.

Photosynthesis → sugar (_photosynthesis) so energy is produced where light is.

Non-random internal transport (_internal_transport) so sugar/water/minerals move locally along the plant network rather than being globally well-mixed.

Maintenance paid mainly by sugar + water (_maintenance_and_death) so shaded or poorly supplied tissue dies back.

Growth requires sugar + water + minerals (_growth) so you can’t expand indefinitely just by touching soil.

Minerals are soil-sourced building stock (uptake + growth cost + recycling) so canopy growth depends on connection to soil.

Gravity falling (_gravity_fall) so unsupported lateral carpets tend to drop and reorganize rather than freezing into a flat blanket.

Recycling (_dead_decay) so minerals are conserved-ish and the system doesn’t permanently lock up resources.