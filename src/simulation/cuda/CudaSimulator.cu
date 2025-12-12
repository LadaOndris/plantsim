#include "CudaSimulator.h"
#include "CudaUtils.cuh"
#include <cstring>

/**
 * @brief Kernel to compute the receiver mask (valid cells that can receive resources).
 */
__global__ void computeReceiverMaskKernel(
    const int* __restrict__ cellTypes,
    float* __restrict__ receiverMask,
    int width, int height,
    int storageWidth, int storageHeight
) {
    KERNEL_2D_SETUP(x, y, idx, storageWidth, storageHeight);
    
    bool isValid = isValidHexCell(x, y, width, height, storageWidth, storageHeight);
    bool isCell = (cellTypes[idx] == 1);  // CellState::Type::Cell
    
    receiverMask[idx] = (isValid && isCell) ? 1.0f : 0.0f;
}

/**
 * @brief Kernel to count how many valid neighbors can receive from each cell.
 */
__global__ void countNeighborsCanReceiveKernel(
    const float* __restrict__ receiverMask,
    float* __restrict__ neighborsCount,
    int width, int height,
    int storageWidth, int storageHeight
) {
    KERNEL_2D_SETUP(x, y, idx, storageWidth, storageHeight);
    
    float count = 0.0f;
    
    // For each neighbor direction, check if the neighbor can receive
    FOR_EACH_NEIGHBOR(x, y, storageWidth, storageHeight, nidx, {
        count += receiverMask[nidx];
    });
    
    neighborsCount[idx] = count;
}

/**
 * @brief Kernel to compute outgoing flow and flow per neighbor.
 */
__global__ void computeOutgoingFlowKernel(
    const float* __restrict__ resources,
    const float* __restrict__ receiverMask,
    const float* __restrict__ neighborsCount,
    float* __restrict__ totalOutgoing,
    float* __restrict__ flowPerNeighbor,
    int storageWidth, int storageHeight
) {
    KERNEL_2D_SETUP(x, y, idx, storageWidth, storageHeight);
    
    float availableOutflow = resources[idx] * receiverMask[idx];
    float neighborCount = neighborsCount[idx];
    
    // Total outgoing is minimum of available and neighbor count
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
 * @brief Kernel to compute total incoming flow for each cell.
 */
__global__ void computeIncomingFlowKernel(
    const float* __restrict__ flowPerNeighbor,
    const float* __restrict__ receiverMask,
    float* __restrict__ totalIncoming,
    int width, int height,
    int storageWidth, int storageHeight
) {
    KERNEL_2D_SETUP(x, y, idx, storageWidth, storageHeight);
    
    float incoming = 0.0f;
    
    // Sum incoming flow from all neighbors
    // Incoming flow comes from neighbors that are sending to us (opposite direction)
    for (int dir = 0; dir < HEX_NEIGHBOR_COUNT; dir++) {
        int nidx = getOppositeNeighborIndex(x, y, dir, storageWidth, storageHeight);
        if (nidx >= 0) {
            incoming += flowPerNeighbor[nidx];
        }
    }
    
    // Only cells that can receive will get incoming flow
    totalIncoming[idx] = incoming * receiverMask[idx];
}

/**
 * @brief Kernel to update resources: new = old - outgoing + incoming.
 */
__global__ void updateResourcesKernel(
    const float* __restrict__ resources,
    const float* __restrict__ totalOutgoing,
    const float* __restrict__ totalIncoming,
    float* __restrict__ nextResources,
    int storageWidth, int storageHeight
) {
    KERNEL_2D_SETUP(x, y, idx, storageWidth, storageHeight);
    
    nextResources[idx] = resources[idx] - totalOutgoing[idx] + totalIncoming[idx];
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
    , d_receiverMask(other.d_receiverMask)
    , d_neighborsCount(other.d_neighborsCount)
    , d_totalOutgoing(other.d_totalOutgoing)
    , d_flowPerNeighbor(other.d_flowPerNeighbor)
    , d_totalIncoming(other.d_totalIncoming)
{
    // Null out the other's pointers to prevent double-free
    other.d_resources = nullptr;
    other.d_nextResources = nullptr;
    other.d_cellTypes = nullptr;
    other.d_receiverMask = nullptr;
    other.d_neighborsCount = nullptr;
    other.d_totalOutgoing = nullptr;
    other.d_flowPerNeighbor = nullptr;
    other.d_totalIncoming = nullptr;
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
        d_receiverMask = other.d_receiverMask;
        d_neighborsCount = other.d_neighborsCount;
        d_totalOutgoing = other.d_totalOutgoing;
        d_flowPerNeighbor = other.d_flowPerNeighbor;
        d_totalIncoming = other.d_totalIncoming;
        
        // Null out the other's pointers
        other.d_resources = nullptr;
        other.d_nextResources = nullptr;
        other.d_cellTypes = nullptr;
        other.d_receiverMask = nullptr;
        other.d_neighborsCount = nullptr;
        other.d_totalOutgoing = nullptr;
        other.d_flowPerNeighbor = nullptr;
        other.d_totalIncoming = nullptr;
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
    CUDA_CHECK(cudaMalloc(&d_receiverMask, floatSize));
    CUDA_CHECK(cudaMalloc(&d_neighborsCount, floatSize));
    CUDA_CHECK(cudaMalloc(&d_totalOutgoing, floatSize));
    CUDA_CHECK(cudaMalloc(&d_flowPerNeighbor, floatSize));
    CUDA_CHECK(cudaMalloc(&d_totalIncoming, floatSize));
}

void CudaSimulator::freeDeviceMemory() {
    if (d_resources) cudaFree(d_resources);
    if (d_nextResources) cudaFree(d_nextResources);
    if (d_cellTypes) cudaFree(d_cellTypes);
    if (d_receiverMask) cudaFree(d_receiverMask);
    if (d_neighborsCount) cudaFree(d_neighborsCount);
    if (d_totalOutgoing) cudaFree(d_totalOutgoing);
    if (d_flowPerNeighbor) cudaFree(d_flowPerNeighbor);
    if (d_totalIncoming) cudaFree(d_totalIncoming);
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
    
    // Step 1: Compute receiver mask
    computeReceiverMaskKernel<<<config.gridSize, config.blockSize>>>(
        d_cellTypes, d_receiverMask,
        state.width, state.height,
        storageWidth, storageHeight
    );
    
    // Step 2: Count neighbors that can receive
    countNeighborsCanReceiveKernel<<<config.gridSize, config.blockSize>>>(
        d_receiverMask, d_neighborsCount,
        state.width, state.height,
        storageWidth, storageHeight
    );
    
    // Step 3: Compute outgoing flow and flow per neighbor
    computeOutgoingFlowKernel<<<config.gridSize, config.blockSize>>>(
        d_resources, d_receiverMask, d_neighborsCount,
        d_totalOutgoing, d_flowPerNeighbor,
        storageWidth, storageHeight
    );
    
    // Step 4: Compute incoming flow
    computeIncomingFlowKernel<<<config.gridSize, config.blockSize>>>(
        d_flowPerNeighbor, d_receiverMask, d_totalIncoming,
        state.width, state.height,
        storageWidth, storageHeight
    );
    
    // Step 5: Update resources
    updateResourcesKernel<<<config.gridSize, config.blockSize>>>(
        d_resources, d_totalOutgoing, d_totalIncoming, d_nextResources,
        storageWidth, storageHeight
    );
    
    // Swap buffers
    std::swap(d_resources, d_nextResources);
    
    // Synchronize to ensure all operations complete
    CUDA_CHECK(cudaDeviceSynchronize());
}
