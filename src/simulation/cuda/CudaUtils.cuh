#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * @brief Macro for checking CUDA API calls and throwing on error.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

/**
 * @brief Number of neighbors in a hexagonal grid.
 */
constexpr int HEX_NEIGHBOR_COUNT = 6;

/**
 * @brief Default block size for 2D kernels.
 */
constexpr int DEFAULT_BLOCK_SIZE = 16;

/**
 * @brief Compute linear index from 2D coordinates.
 * @param x Column coordinate
 * @param y Row coordinate
 * @param width Width of the grid (stride)
 * @return Linear index into 1D storage
 */
__device__ __forceinline__ int toLinearIndex(int x, int y, int width) {
    return y * width + x;
}

/**
 * @brief Extract X coordinate from linear index.
 * @param idx Linear index
 * @param width Width of the grid
 * @return X (column) coordinate
 */
__device__ __forceinline__ int toX(int idx, int width) {
    return idx % width;
}

/**
 * @brief Extract Y coordinate from linear index.
 * @param idx Linear index
 * @param width Width of the grid
 * @return Y (row) coordinate
 */
__device__ __forceinline__ int toY(int idx, int width) {
    return idx / width;
}

/**
 * @brief Get the thread's 2D position in the grid.
 * @param[out] x Output X coordinate
 * @param[out] y Output Y coordinate
 */
__device__ __forceinline__ void getThreadPosition(int& x, int& y) {
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
}

/**
 * @brief Check if coordinates are within bounds.
 * @param x X coordinate
 * @param y Y coordinate
 * @param width Width bound
 * @param height Height bound
 * @return true if within bounds
 */
__device__ __forceinline__ bool isInBounds(int x, int y, int width, int height) {
    return x >= 0 && x < width && y >= 0 && y < height;
}

/**
 * @brief Checks if storage coordinates represent a valid hex cell.
 * 
 * This handles the offset coordinate system where not all storage cells
 * map to valid hex grid positions.
 * 
 * @param x Storage X coordinate
 * @param y Storage Y coordinate
 * @param width Logical hex grid width
 * @param height Logical hex grid height
 * @param storageWidth Storage array width
 * @param storageHeight Storage array height
 * @return true if this is a valid hex cell
 */
__device__ __forceinline__ bool isValidHexCell(
    int x, int y,
    int width, int height,
    int storageWidth, int storageHeight
) {
    if (x < 0 || x >= storageWidth || y < 0 || y >= storageHeight) {
        return false;
    }
    
    int offset = (height - 1) / 2;
    int parity = y & 1;
    int minStorageX = offset - ((y - parity) / 2);
    int maxStorageX = minStorageX + width - 1;
    
    return x >= minStorageX && x <= maxStorageX;
}

/**
 * @brief Neighbor iterator for hexagonal grids.
 * 
 * Provides inline iteration over the 6 neighbors of a hex cell.
 * Neighbor offsets are for axial coordinates (q, r):
 *   0: Right       (+1,  0)
 *   1: TopRight    (+1, -1)
 *   2: TopLeft     ( 0, -1)
 *   3: Left        (-1,  0)
 *   4: BottomLeft  (-1, +1)
 *   5: BottomRight ( 0, +1)
 */
struct HexNeighborOffsets {
    static constexpr int dq[HEX_NEIGHBOR_COUNT] = { +1, +1,  0, -1, -1,  0 };
    static constexpr int dr[HEX_NEIGHBOR_COUNT] = {  0, -1, -1,  0, +1, +1 };
};

/**
 * @brief Get the offset for a specific neighbor direction.
 * @param direction Neighbor direction (0-5)
 * @param[out] dq Delta Q offset
 * @param[out] dr Delta R offset
 */
__device__ __forceinline__ void getNeighborOffset(int direction, int& dq, int& dr) {
    // Hardcoded for performance (avoids array access overhead)
    switch (direction) {
        case 0: dq = +1; dr =  0; break;  // Right
        case 1: dq = +1; dr = -1; break;  // TopRight
        case 2: dq =  0; dr = -1; break;  // TopLeft
        case 3: dq = -1; dr =  0; break;  // Left
        case 4: dq = -1; dr = +1; break;  // BottomLeft
        case 5: dq =  0; dr = +1; break;  // BottomRight
        default: dq = 0; dr = 0; break;
    }
}

/**
 * @brief Calculate neighbor coordinates from current position.
 * @param x Current X coordinate
 * @param y Current Y coordinate
 * @param direction Neighbor direction (0-5)
 * @param[out] nx Neighbor X coordinate
 * @param[out] ny Neighbor Y coordinate
 */
__device__ __forceinline__ void getNeighborCoords(
    int x, int y, int direction,
    int& nx, int& ny
) {
    int dq, dr;
    getNeighborOffset(direction, dq, dr);
    nx = x + dq;
    ny = y + dr;
}

/**
 * @brief Calculate neighbor index if valid, otherwise return -1.
 * @param x Current X coordinate
 * @param y Current Y coordinate
 * @param direction Neighbor direction (0-5)
 * @param storageWidth Width of storage array
 * @param storageHeight Height of storage array
 * @return Neighbor linear index, or -1 if out of bounds
 */
__device__ __forceinline__ int getNeighborIndex(
    int x, int y, int direction,
    int storageWidth, int storageHeight
) {
    int nx, ny;
    getNeighborCoords(x, y, direction, nx, ny);
    
    if (isInBounds(nx, ny, storageWidth, storageHeight)) {
        return toLinearIndex(nx, ny, storageWidth);
    }
    return -1;
}

/**
 * @brief Macro for iterating over all hex neighbors with bounds checking.
 * 
 * Usage:
 *   FOR_EACH_NEIGHBOR(x, y, storageWidth, storageHeight, neighborIdx, {
 *       // Use neighborIdx here
 *       value += data[neighborIdx];
 *   });
 */
#define FOR_EACH_NEIGHBOR(x, y, storageWidth, storageHeight, neighborIdxVar, body) \
    do { \
        for (int _dir = 0; _dir < HEX_NEIGHBOR_COUNT; _dir++) { \
            int neighborIdxVar = getNeighborIndex(x, y, _dir, storageWidth, storageHeight); \
            if (neighborIdxVar >= 0) { \
                body \
            } \
        } \
    } while (0)

/**
 * @brief Macro for iterating over neighbors with access to direction.
 * 
 * Usage:
 *   FOR_EACH_NEIGHBOR_WITH_DIR(x, y, storageWidth, storageHeight, dir, nidx, {
 *       // Use dir and nidx here
 *   });
 */
#define FOR_EACH_NEIGHBOR_WITH_DIR(x, y, storageWidth, storageHeight, dirVar, neighborIdxVar, body) \
    do { \
        for (int dirVar = 0; dirVar < HEX_NEIGHBOR_COUNT; dirVar++) { \
            int neighborIdxVar = getNeighborIndex(x, y, dirVar, storageWidth, storageHeight); \
            if (neighborIdxVar >= 0) { \
                body \
            } \
        } \
    } while (0)

/**
 * @brief Compute optimal grid dimensions for a 2D kernel.
 * @param width Grid width
 * @param height Grid height
 * @param blockSize Block size (default 16x16)
 * @return Pair of (gridSize, blockSize) dim3 values
 */
inline std::pair<dim3, dim3> computeGridDimensions(
    int width, int height,
    int blockSize = DEFAULT_BLOCK_SIZE
) {
    dim3 block(blockSize, blockSize);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    return {grid, block};
}

/**
 * @brief Helper struct for kernel launch configuration.
 */
struct KernelConfig {
    dim3 gridSize;
    dim3 blockSize;
    
    KernelConfig(int width, int height, int blockDim = DEFAULT_BLOCK_SIZE) {
        auto [grid, block] = computeGridDimensions(width, height, blockDim);
        gridSize = grid;
        blockSize = block;
    }
};

/**
 * @brief Standard 2D kernel entry check - returns if thread is out of bounds.
 * 
 * Usage at kernel start:
 *   KERNEL_2D_GUARD(x, y, storageWidth, storageHeight);
 */
#define KERNEL_2D_GUARD(xVar, yVar, width, height) \
    int xVar, yVar; \
    getThreadPosition(xVar, yVar); \
    if (xVar >= (width) || yVar >= (height)) return

/**
 * @brief Standard 2D kernel entry with index computation.
 * 
 * Usage at kernel start:
 *   KERNEL_2D_SETUP(x, y, idx, storageWidth, storageHeight);
 */
#define KERNEL_2D_SETUP(xVar, yVar, idxVar, width, height) \
    KERNEL_2D_GUARD(xVar, yVar, width, height); \
    int idxVar = toLinearIndex(xVar, yVar, width)

// =============================================================================
// Shared Memory Tile Utilities
// =============================================================================

/**
 * @brief Halo size for hex neighbor access (max offset is 1 in any direction).
 */
constexpr int TILE_HALO = 1;

/**
 * @brief Tile dimensions including halo for shared memory.
 */
constexpr int TILE_WIDTH = DEFAULT_BLOCK_SIZE + 2 * TILE_HALO;   // 18
constexpr int TILE_HEIGHT = DEFAULT_BLOCK_SIZE + 2 * TILE_HALO;  // 18

/**
 * @brief Load a tile with halo into shared memory.
 * 
 * Each thread loads its own cell, plus some threads load halo cells.
 * This handles boundary conditions by loading 0.0f for out-of-bounds cells.
 * 
 * @param globalData Source data in global memory
 * @param tile Destination shared memory tile [TILE_HEIGHT][TILE_WIDTH]
 * @param tileStartX Global X coordinate of tile's top-left corner (excluding halo)
 * @param tileStartY Global Y coordinate of tile's top-left corner (excluding halo)
 * @param storageWidth Width of global storage
 * @param storageHeight Height of global storage
 */
__device__ __forceinline__ void loadTileWithHalo(
    const float* __restrict__ globalData,
    float tile[TILE_HEIGHT][TILE_WIDTH],
    int tileStartX, int tileStartY,
    int storageWidth, int storageHeight
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Each thread loads multiple cells to cover the tile + halo
    // Tile covers [tileStartX - HALO, tileStartX + BLOCK_SIZE + HALO) in X
    // and similar in Y.
    for (int localY = ty; localY < TILE_HEIGHT; localY += blockDim.y) {
        for (int localX = tx; localX < TILE_WIDTH; localX += blockDim.x) {
            int globalX = tileStartX + localX - TILE_HALO;
            int globalY = tileStartY + localY - TILE_HALO;
            
            float value = 0.0f;
            if (isInBounds(globalX, globalY, storageWidth, storageHeight)) {
                value = globalData[toLinearIndex(globalX, globalY, storageWidth)];
            }
            tile[localY][localX] = value;
        }
    }
}

/**
 * @brief Get value from shared memory tile for a neighbor.
 * 
 * @param tile Shared memory tile
 * @param localX Thread's local X in tile (threadIdx.x + TILE_HALO)
 * @param localY Thread's local Y in tile (threadIdx.y + TILE_HALO)
 * @param direction Hex neighbor direction (0-5)
 * @return Neighbor's value from tile
 */
__device__ __forceinline__ float getTileNeighbor(
    const float tile[TILE_HEIGHT][TILE_WIDTH],
    int localX, int localY, int direction
) {
    int dq, dr;
    getNeighborOffset(direction, dq, dr);
    return tile[localY + dr][localX + dq];
}

/**
 * @brief Get value from shared memory tile for an opposite-direction neighbor.
 * 
 * Used for incoming flow where we look at neighbors in the opposite direction.
 * 
 * @param tile Shared memory tile
 * @param localX Thread's local X in tile (threadIdx.x + TILE_HALO)
 * @param localY Thread's local Y in tile (threadIdx.y + TILE_HALO)
 * @param direction Hex neighbor direction (0-5), will access opposite
 * @return Opposite neighbor's value from tile
 */
__device__ __forceinline__ float getTileOppositeNeighbor(
    const float tile[TILE_HEIGHT][TILE_WIDTH],
    int localX, int localY, int direction
) {
    int dq, dr;
    getNeighborOffset(direction, dq, dr);
    // Opposite direction: subtract instead of add
    return tile[localY - dr][localX - dq];
}
