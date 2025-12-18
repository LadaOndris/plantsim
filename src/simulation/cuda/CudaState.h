#pragma once

#include "simulation/State.h"
#include <memory>

class CudaState {
public:
  int width = 0;
  int height = 0;
  int storageWidth = 0;
  int storageHeight = 0;
  size_t totalStorageCells = 0;

  float *d_resources = nullptr;
  int *d_cellTypes = nullptr;

public:
  explicit CudaState(StatePtr ptrState);
  ~CudaState();

  void allocateDeviceMemory();
  void freeDeviceMemory();

  void copyStateToDevice();
  void copyStateFromDevice();

  const State &getState() const;

private:
  StatePtr state;
};

using CudaStatePtr = std::shared_ptr<CudaState>;
