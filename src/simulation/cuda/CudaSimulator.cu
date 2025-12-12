#include "CudaSimulator.h"
#include "CudaUtils.cuh"
#include <cstring>

/**
 * @brief Compute receiver mask value on-the-fly.
 * Returns 1.0f if the cell is a valid hex cell AND is a Cell type, 0.0f otherwise.
 */
__device__ __forceinline__ float computeReceiverMask(
    const int* __restrict__ cellTypes,
    int x, int y, int idx,
    int width, int height,
    int storageWidth, int storageHeight
) {
    bool isValid = isValidHexCell(x, y, width, height, storageWidth, storageHeight);
    bool isCell = (cellTypes[idx] == 1);  // CellState::Type::Cell
    return (isValid && isCell) ? 1.0f : 0.0f;
}

/**
 * @brief Compute receiver mask for a neighbor cell.
 */
__device__ __forceinline__ float computeNeighborReceiverMask(
    const int* __restrict__ cellTypes,
    int nx, int ny,
    int width, int height,
    int storageWidth, int storageHeight
) {
    if (!isInBounds(nx, ny, storageWidth, storageHeight)) {
        return 0.0f;
    }
    int nidx = toLinearIndex(nx, ny, storageWidth);
    return computeReceiverMask(cellTypes, nx, ny, nidx, width, height, storageWidth, storageHeight);
}

/**
 * @brief Fused kernel: count neighbors + compute outgoing flow.
 * Computes receiver mask on-the-fly, eliminating d_receiverMask buffer.
 */
__global__ void computeOutgoingFlowKernel(
    const float* __restrict__ resources,
    const int* __restrict__ cellTypes,
    float* __restrict__ totalOutgoing,
    float* __restrict__ flowPerNeighbor,
    int width, int height,
    int storageWidth, int storageHeight
) {
    KERNEL_2D_SETUP(x, y, idx, storageWidth, storageHeight);
    
    // Count neighbors that can receive
    float neighborCount = 0.0f;
    for (int dir = 0; dir < HEX_NEIGHBOR_COUNT; dir++) {
        int nx, ny;
        getNeighborCoords(x, y, dir, nx, ny);
        neighborCount += computeNeighborReceiverMask(cellTypes, nx, ny, width, height, storageWidth, storageHeight);
    }
    
    // Compute receiver mask for current cell
    float mask = computeReceiverMask(cellTypes, x, y, idx, width, height, storageWidth, storageHeight);
    
    float availableOutflow = resources[idx] * mask;
    float outgoing = fminf(availableOutflow, neighborCount);
    totalOutgoing[idx] = outgoing;
    
    // Flow per neighbor (avoid division by zero)
    flowPerNeighbor[idx] = (neighborCount > 0.0f) ? (outgoing / neighborCount) : 0.0f;
}

/**
 * @brief Compute neighbor index in opposite direction (for incoming flow).
 */
__device__ __forceinline__ int getOppositeNeighborIndex(
    int x, int y, int direction,
    int storageWidth, int storageHeight
) {
    int dq, dr;
    getNeighborOffset(direction, dq, dr);
    int nx = x - dq;  // Opposite direction
    int ny = y - dr;
    
    if (isInBounds(nx, ny, storageWidth, storageHeight)) {
        return toLinearIndex(nx, ny, storageWidth);
    }
    return -1;
}

/**
 * @brief Fused kernel: compute incoming flow + update resources.
 * Computes receiver mask on-the-fly, eliminating d_receiverMask buffer.
 */
__global__ void updateResourcesKernel(
    const float* __restrict__ resources,
    const float* __restrict__ flowPerNeighbor,
    const int* __restrict__ cellTypes,
    const float* __restrict__ totalOutgoing,
    float* __restrict__ nextResources,
    int width, int height,
    int storageWidth, int storageHeight
) {
    KERNEL_2D_SETUP(x, y, idx, storageWidth, storageHeight);
    
    float incoming = 0.0f;
    
    // Sum incoming flow from all neighbors (opposite direction)
    for (int dir = 0; dir < HEX_NEIGHBOR_COUNT; dir++) {
        int nidx = getOppositeNeighborIndex(x, y, dir, storageWidth, storageHeight);
        if (nidx >= 0) {
            incoming += flowPerNeighbor[nidx];
        }
    }
    
    // Compute receiver mask for current cell
    float mask = computeReceiverMask(cellTypes, x, y, idx, width, height, storageWidth, storageHeight);
    incoming *= mask;  // Only cells that can receive get incoming
    
    nextResources[idx] = resources[idx] - totalOutgoing[idx] + incoming;
}

CudaSimulator::CudaSimulator(State initialState)
    : state(std::move(initialState))
{
    const int height = state.height;
    int additionalWidth = (height - 1) / 2;
    storageWidth = state.width + additionalWidth;
    storageHeight = height;
    totalStorageCells = static_cast<size_t>(storageWidth) * storageHeight;
    
    allocateDeviceMemory();
    copyStateToDevice();
}

CudaSimulator::~CudaSimulator() {
    freeDeviceMemory();
}

CudaSimulator::CudaSimulator(CudaSimulator&& other) noexcept
    : state(std::move(other.state))
    , storageWidth(other.storageWidth)
    , storageHeight(other.storageHeight)
    , totalStorageCells(other.totalStorageCells)
    , d_resources(other.d_resources)
    , d_nextResources(other.d_nextResources)
    , d_cellTypes(other.d_cellTypes)
    , d_totalOutgoing(other.d_totalOutgoing)
    , d_flowPerNeighbor(other.d_flowPerNeighbor)
{
    // Null out the other's pointers to prevent double-free
    other.d_resources = nullptr;
    other.d_nextResources = nullptr;
    other.d_cellTypes = nullptr;
    other.d_totalOutgoing = nullptr;
    other.d_flowPerNeighbor = nullptr;
    other.totalStorageCells = 0;
}

CudaSimulator& CudaSimulator::operator=(CudaSimulator&& other) noexcept {
    if (this != &other) {
        freeDeviceMemory();
        
        state = std::move(other.state);
        storageWidth = other.storageWidth;
        storageHeight = other.storageHeight;
        totalStorageCells = other.totalStorageCells;
        d_resources = other.d_resources;
        d_nextResources = other.d_nextResources;
        d_cellTypes = other.d_cellTypes;
        d_totalOutgoing = other.d_totalOutgoing;
        d_flowPerNeighbor = other.d_flowPerNeighbor;
        
        // Null out the other's pointers
        other.d_resources = nullptr;
        other.d_nextResources = nullptr;
        other.d_cellTypes = nullptr;
        other.d_totalOutgoing = nullptr;
        other.d_flowPerNeighbor = nullptr;
        other.totalStorageCells = 0;
    }
    return *this;
}


void CudaSimulator::allocateDeviceMemory() {
    size_t floatSize = totalStorageCells * sizeof(float);
    size_t intSize = totalStorageCells * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_resources, floatSize));
    CUDA_CHECK(cudaMalloc(&d_nextResources, floatSize));
    CUDA_CHECK(cudaMalloc(&d_cellTypes, intSize));
    CUDA_CHECK(cudaMalloc(&d_totalOutgoing, floatSize));
    CUDA_CHECK(cudaMalloc(&d_flowPerNeighbor, floatSize));
}

void CudaSimulator::freeDeviceMemory() {
    if (d_resources) cudaFree(d_resources);
    if (d_nextResources) cudaFree(d_nextResources);
    if (d_cellTypes) cudaFree(d_cellTypes);
    if (d_totalOutgoing) cudaFree(d_totalOutgoing);
    if (d_flowPerNeighbor) cudaFree(d_flowPerNeighbor);
}

void CudaSimulator::copyStateToDevice() {
    size_t floatSize = totalStorageCells * sizeof(float);
    size_t intSize = totalStorageCells * sizeof(int);
    
    CUDA_CHECK(cudaMemcpy(d_resources, state.resources.data(), floatSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellTypes, state.cellTypes.data(), intSize, cudaMemcpyHostToDevice));
}

void CudaSimulator::copyStateFromDevice() {
    size_t floatSize = totalStorageCells * sizeof(float);
    CUDA_CHECK(cudaMemcpy(state.resources.data(), d_resources, floatSize, cudaMemcpyDeviceToHost));
}

const State& CudaSimulator::getState() const {
    // Need to copy from device first (const_cast needed since getState is const)
    const_cast<CudaSimulator*>(this)->copyStateFromDevice();
    return state;
}

void CudaSimulator::step(const Options& options) {
    if (options.enableResourceTransfer) {
        transferResources();
    }
}

void CudaSimulator::transferResources() {
    // Configure grid and block dimensions
    KernelConfig config(storageWidth, storageHeight);
    
    // Step 1: Count neighbors + compute outgoing flow
    computeOutgoingFlowKernel<<<config.gridSize, config.blockSize>>>(
        d_resources, d_cellTypes,
        d_totalOutgoing, d_flowPerNeighbor,
        state.width, state.height,
        storageWidth, storageHeight
    );
    
    // Step 2: Compute incoming flow + update resources
    updateResourcesKernel<<<config.gridSize, config.blockSize>>>(
        d_resources, d_flowPerNeighbor, d_cellTypes, d_totalOutgoing,
        d_nextResources,
        state.width, state.height,
        storageWidth, storageHeight
    );
    
    // Swap buffers
    std::swap(d_resources, d_nextResources);
}
