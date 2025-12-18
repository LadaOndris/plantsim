
#include "CudaState.h"
#include "CudaUtils.cuh"

CudaState::CudaState(StatePtr ptrState) :
    width(ptrState->width),
    height(ptrState->height),
    state(ptrState) {
  int additionalWidth = (height - 1) / 2;
  storageWidth = state->width + additionalWidth;
  storageHeight = height;
  totalStorageCells = static_cast<size_t>(storageWidth) * storageHeight;

  allocateDeviceMemory();
  copyStateToDevice();
}

CudaState::~CudaState() {
  freeDeviceMemory();
}

void CudaState::allocateDeviceMemory() {
  size_t floatSize = totalStorageCells * sizeof(float);
  size_t intSize = totalStorageCells * sizeof(int);

  CUDA_CHECK(cudaMalloc(&d_resources, floatSize));
  CUDA_CHECK(cudaMalloc(&d_cellTypes, intSize));
}

void CudaState::freeDeviceMemory() {
  if (d_resources)
    cudaFree(d_resources);
  if (d_cellTypes)
    cudaFree(d_cellTypes);
}

void CudaState::copyStateToDevice() {
  size_t floatSize = totalStorageCells * sizeof(float);
  size_t intSize = totalStorageCells * sizeof(int);

  CUDA_CHECK(cudaMemcpy(d_resources, state->resources.data(), floatSize,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cellTypes, state->cellTypes.data(), intSize,
                        cudaMemcpyHostToDevice));
}

void CudaState::copyStateFromDevice() {
  size_t floatSize = totalStorageCells * sizeof(float);
  size_t intSize = totalStorageCells * sizeof(int);

  CUDA_CHECK(cudaMemcpy(state->resources.data(), d_resources, floatSize,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(state->cellTypes.data(), d_cellTypes, intSize,
                        cudaMemcpyDeviceToHost));
}

const State &CudaState::getState() const {
  const_cast<CudaState *>(this)->copyStateFromDevice();
  return *state;
}
