import numpy as np
import matplotlib.pyplot as plt
from time import time

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def shift_matrix(matrix, offset, fill_value=0):
    """
    Shifts a 2D numpy array by the given offset (dy, dx).
    Elements shifted from outside are filled with fill_value.
    
    Args:
        matrix: 2D numpy array
        offset: Tuple (dy, dx)
        fill_value: Value to pad with (usually 0 or False)
    """
    dy, dx = offset
    h, w = matrix.shape
    result = np.full_like(matrix, fill_value)
    
    # Calculate slice ranges
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)
    
    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)
    
    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        matrix[src_y_start:src_y_end, src_x_start:src_x_end]
        
    return result

def precompute_topology(validity_map, neighbors):
    """
    Precomputes the neighbor validity masks and counts since the grid topology
    is static.
    
    Returns:
        valid_neighbor_counts: Matrix where each cell contains the count of its valid neighbors.
        neighbor_validity_cache: List of matrices, one per direction, indicating if that neighbor is valid.
    """
    valid_neighbor_counts = np.zeros_like(validity_map, dtype=int)
    neighbor_validity_cache = []

    for dy, dx in neighbors:
        inv_direction = (-dy, -dx)
        # Shift validity map backwards to see if a neighbor exists in this direction
        n_valid = shift_matrix(validity_map, inv_direction, fill_value=0)
        valid_neighbor_counts += n_valid
        neighbor_validity_cache.append(n_valid)
        
    return valid_neighbor_counts, neighbor_validity_cache

def run_step(resources, validity_map, neighbors, valid_neighbor_counts, neighbor_validity_cache):
    """
    Performs one step of resource redistribution using matrix operations.
    Simplified to calculate total outflow directly.
    """
    
    # 1. Identify Active Sources
    # A cell is active if it has > 1 resource and is a valid cell
    active_mask = (resources > 1) & (validity_map == 1)
    
    # 2. Calculate Total Outgoing Flow (Vectorized)
    # We want to send 1.0 to every valid neighbor.
    # Therefore, desired outflow = count of valid neighbors * 1.0.
    # The actual outflow is limited by available resources.
    desired_outflow = valid_neighbor_counts * 1.0
    total_outgoing = np.minimum(resources, desired_outflow) * active_mask
    
    # 3. Calculate Flow Per Neighbor
    # This determines how much mass flows along each specific valid connection.
    # flow_per_neighbor = Total Outflow / Count of Valid Neighbors
    # We use np.divide with a 'where' clause to safely handle cells with 0 neighbors.
    flow_per_neighbor = np.divide(
        total_outgoing, 
        valid_neighbor_counts, 
        out=np.zeros_like(resources), 
        where=valid_neighbor_counts != 0
    )

    # 4. Calculate Incoming Flow
    # We still loop here because 'shift' depends on specific direction.
    total_incoming = np.zeros_like(resources, dtype=float)
    
    for i, (dy, dx) in enumerate(neighbors):
        # Retrieve validity mask for this direction
        neighbor_is_valid = neighbor_validity_cache[i]
        
        # Flow exists in this direction only if the specific neighbor is valid
        flow_out_in_dir = flow_per_neighbor * neighbor_is_valid
        
        # Shift this flow to the neighbor's position to become "incoming"
        flow_in = shift_matrix(flow_out_in_dir, (dy, dx), fill_value=0)
        
        total_incoming += flow_in
        
    # 5. Update State
    new_resources = resources - total_outgoing + total_incoming
    
    # Floating point safety clamp
    return np.maximum(new_resources, 0)

def simulate():
    # --- Setup ---
    H, W = 200, 200
    
    # 1. Define Neighbor Offsets for Axial Coordinates
    # These correspond to the 6 neighbors in a hex grid stored as (r, q)
    axial_neighbors = [
        (0, 1),   # +q
        (0, -1),  # -q
        (1, 0),   # +r
        (-1, 0),  # -r
        (1, -1),  # +r, -q
        (-1, 1)   # -r, +q
    ]
    
    # 2. Create Validity Matrix (0 = Air, 1 = Cell)
    # Let's make a diamond shape or irregular blob
    validity = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            # Arbitrary shape logic for demo
            if 2 <= r <= 7 and 2 <= c <= 7:
                validity[r, c] = 1
    # Punch a hole in the middle (Air)
    validity[4, 4] = 0
    validity[5, 5] = 0

    # 3. Initialize Resources
    resources = np.zeros((H, W), dtype=float)
    # Place a large pile of resources at (3, 3)
    resources[3, 3] = 200.0
    
    # Ensure resources only exist on valid cells
    resources = resources * validity

    print("--- Initial State (Subset) ---")
    print(resources[2:8, 2:8])

    # --- Precompute Topology ---
    # This step is now done once before the loop
    valid_neighbor_counts, neighbor_validity_cache = precompute_topology(validity, axial_neighbors)

    # --- Simulation Loop ---
    steps = 2000
    print(f"\n--- Simulating {steps} steps ---")
    

    @timer_func
    def run_steps(resources, steps):
      for t in range(steps):
          resources = run_step(resources, validity, axial_neighbors, valid_neighbor_counts, neighbor_validity_cache)
      return resources
        # Optional: Print sum to check conservation 
        # (Mass is conserved unless it flows off the validity map edge)
        #print(f"Step {t+1}: Total Mass = {np.sum(resources):.1f}")
    resources = run_steps(resources, steps)

    print("\n--- Final State (Subset) ---")
    print(np.round(resources[2:8, 2:8], 1))
    
    # Visualizing logic (ASCII)
    # print("\n--- Visual Map (Values > 0) ---")
    # rows, cols = resources.shape
    # for r in range(rows):
    #     line = ""
    #     for c in range(cols):
    #         if validity[r, c] == 0:
    #             line += " .  " # Air
    #         elif resources[r, c] == 0:
    #             line += " _  " # Empty Cell
    #         else:
    #             line += f"{resources[r,c]:.1f} "
    #     print(line)

if __name__ == "__main__":
    simulate()