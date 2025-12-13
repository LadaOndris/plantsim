# Business Logic Design Design

## Cell reproduction

### Decision axes

There are multiple axes when it comes to decisions on how cell replication works.

#### A. Eligibility

When can a cell reproduce?

* Resource threshold (fixed or dynamic)
* Cooldown / age-based limits
* Environmental constraints (light, nutrients, space)
* Stochastic chance even if conditions are met

#### B. Neighbor selection

Where does the offspring go?

* Uniform random neighbor
* Directionally biased

#### C. Resource economics

* Fixed reproduction cost
* Proportional cost (percentage)
* Split evenly (mitosis-style)
* Parent donates minimal “seed” amount

#### D. Conflict resolution

#### E. Update semantics

When does reproduction “take effect”?

* Immediate (in-place)
* Buffered (like your backBuffer)
* Multi-phase (decide → resolve → apply)

### Reproduction Policy Families

### 1. Random Neighbor Reproduction

**Summary:**  
A cell reproduces into a randomly chosen neighboring empty cell.

**Key idea:**  
No direction or preference—every valid neighbor is equally likely.

**Results in:**  

* Uniform, blob-like growth  
* High randomness, low structure  

**Good for:**  

* Baseline behavior  
* Early prototyping  
* Debugging correctness

---

### 2. Directionally Biased Reproduction

**Summary:**  
A cell prefers certain directions when reproducing.

**Key idea:**  
Growth is biased (e.g., upward, outward, toward light).

**Results in:**  

* Asymmetric shapes  
* Branching or tendrils  
* Plant- or organism-like morphology

**Good for:**  

* Modeling plants, roots, or directional organisms  
* Introducing structure without heavy computation

---

### 3. Priority-Based Neighbor Selection

**Summary:**  
A cell evaluates all neighboring empty cells and chooses the “best” one.

**Key idea:**  
Each neighbor is scored using heuristics (space, environment, distance, etc.).

**Results in:**  

* Intentional-looking growth  
* Controlled expansion patterns  

**Good for:**  

* Strategic or engineered systems  
* Avoiding overcrowding or dead-ends

---

### 4. Probabilistic Reproduction

**Summary:**  
Even if conditions are met, reproduction happens with some probability.

**Key idea:**  
Chance limits growth and introduces variability.

**Results in:**  

* Slower, more organic population changes  
* Natural fluctuations and extinctions  

**Good for:**  

* Ecological simulations  
* Preventing runaway exponential growth

---

### 5. Competitive Reproduction

**Summary:**  
Multiple cells may attempt to reproduce into the same empty cell.

**Key idea:**  
Conflicts are resolved by rules (random, strongest wins, etc.).

**Results in:**  

* Territorial boundaries  
* Fronts and competition zones  

**Good for:**  

* Modeling competition, dominance, or selection pressure

---

### 6. Pressure-Based / Diffusive Growth

**Summary:**  
Cells influence nearby empty spaces; reproduction happens when influence accumulates.

**Key idea:**  
Growth emerges from accumulated “pressure” rather than discrete choices.

**Results in:**  

* Smooth growth fronts  
* Organic, morphogen-like patterns  

**Good for:**  

* Morphogenesis  
* Reaction–diffusion style systems

---

### 7. Rule-Based (Cellular Automaton-Style)

**Summary:**  
Reproduction follows fixed local rules based on neighbor counts and states.

**Key idea:**  
Strict, deterministic rules define when reproduction happens.

**Results in:**  

* Predictable patterns  
* Often complex behavior from simple rules  

**Good for:**  

* Mathematical exploration  
* Classic cellular automata behavior

---

## Resource Handling Variants (Orthogonal to Policy)

These can be combined with **any** reproduction policy:

* **Fixed cost:** Parent pays a fixed resource amount  
* **Proportional split:** Parent and child share resources  
* **Symmetric split:** Parent divides evenly (mitosis-like)  
* **Minimal seed:** Child starts weak and must survive  

---

## Implementation

### Two-Phase Reproduction

1. Intention phase - Cells declare reproduction attempts
2. Resolution phase - Conflicts resolved
3. Application phase - State updated
