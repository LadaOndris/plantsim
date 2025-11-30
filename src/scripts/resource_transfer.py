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

def run_step(resources, validity_map, cell_type_map, neighbors):
    """
    Performs one step of resource redistribution.
    Calculates neighbor connectivity dynamically based on current cell types.
    
    Logic adapted from pseudocode:
    1. canSendResources = isValidCell * isCellType * hasResources
    2. desiredOutflow = neighborsCanReceiveCount * 1.0
    3. availableOutflow = resources * canSendResources
    4. totalOutgoing = min(availableOutflow, desiredOutflow)
    5. Distribute totalOutgoing to valid neighbors
    """
    
    # --- 0. Calculate Dynamic Connectivity ---
    # A cell can receive if it is valid structurally and of the right type.
    # Since cell_type_map can change, we compute this every step.
    receiver_mask = (validity_map == 1) & (cell_type_map == 1)
    
    valid_neighbor_counts = np.zeros_like(validity_map, dtype=int)
    neighbor_receiver_cache = []

    for dy, dx in neighbors:
        inv_direction = (-dy, -dx)
        
        # Shift the receiver mask backwards to see if a valid receiver exists in this direction
        # (i.e., looking from current cell TO the neighbor)
        neighbor_is_receiver = shift_matrix(receiver_mask, inv_direction, fill_value=0)
        
        valid_neighbor_counts += neighbor_is_receiver
        neighbor_receiver_cache.append(neighbor_is_receiver)

    # --- 1. Determine Sender Capability ---
    # isValidCell * isCellType * hasResources (where hasResources is > 1.0 for redistribution)
    # We maintain the > 1.0 threshold for "excess" redistribution.
    can_send_mask = (validity_map == 1) & (cell_type_map == 1) & (resources > 1)
    
    # --- 2. Calculate Desired Outflow ---
    # neighborsCanReceiveCount * outflowPerNeighbor (1.0)
    outflow_per_neighbor_target = 1.0
    desired_outflow = valid_neighbor_counts * outflow_per_neighbor_target
    
    # --- 3. Calculate Available and Total Outgoing ---
    # availableOutflow = resources * canSendResources (implicitly handled by masking later)
    # totalOutgoing = min(availableOutflow, desiredOutflow)
    
    # Calculate potential outgoing limited by resources
    potential_outgoing = np.minimum(resources, desired_outflow)
    
    # Apply the sender mask (canSendResources)
    total_outgoing = potential_outgoing * can_send_mask
    
    # --- 4. Distribute Flow (Calculate Flow Per Neighbor) ---
    # If a cell sends X amount, and has N neighbors, each gets X/N.
    # We use np.divide to handle the division safely.
    flow_per_neighbor = np.divide(
        total_outgoing, 
        valid_neighbor_counts, 
        out=np.zeros_like(resources), 
        where=valid_neighbor_counts != 0
    )

    # --- 5. Calculate Incoming Flow ---
    total_incoming = np.zeros_like(resources, dtype=float)
    
    for i, (dy, dx) in enumerate(neighbors):
        # Retrieve precomputed mask: "Is the neighbor in this direction a valid receiver?"
        # Corresponds to: isValidCellShifted * isCellTypeShifted
        neighbor_is_receiver = neighbor_receiver_cache[i]
        
        # Calculate flow specifically for this direction
        # flowInDirection (outgoing) = flow_per_neighbor * neighbor_mask
        flow_out_in_dir = flow_per_neighbor * neighbor_is_receiver
        
        # Shift this flow to the neighbor's position to become "incoming" for them
        # totalIncoming += flowInDirection shifted
        flow_in = shift_matrix(flow_out_in_dir, (dy, dx), fill_value=0)
        
        total_incoming += flow_in
        
    # --- 6. Update State ---
    new_resources = resources - total_outgoing + total_incoming
    
    # Floating point safety clamp
    return np.maximum(new_resources, 0)

def simulate():
    # --- Setup ---
    H, W = 200, 200

    # 1. Define Neighbor Offsets (Axial)
    axial_neighbors = [
        (0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)
    ]
    
    # 2. Create Validity Matrix (0 = Air, 1 = Grid)
    validity = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            if 2 <= r <= 7 and 2 <= c <= 7:
                validity[r, c] = 1
    validity[4, 4] = 0 # Hole
    
    # 3. Create Cell Type Matrix (0 = Passive/Wall, 1 = Active/Conduit)
    # For this demo, let's make most cells type 1, but make a specific row type 0 (blocked)
    cell_type = np.ones((H, W), dtype=int)
    cell_type[6, :] = 0 # Row 6 acts as a "type barrier" even if valid grid
    
    # 4. Initialize Resources
    resources = np.zeros((H, W), dtype=float)
    resources[3, 3] = 20.0
    
    # Clean initial state
    resources = resources * validity * cell_type

    print("--- Initial State (Subset) ---")
    print(resources[2:8, 2:8])
    # --- Simulation Loop ---
    steps = 2000
    print(f"\n--- Simulating {steps} steps ---")
    

    @timer_func
    def run_steps(resources, steps):
      for t in range(steps):
          resources = run_step(resources, validity, cell_type, axial_neighbors)
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