#include "CudaSimulator.h"

CudaSimulator::CudaSimulator(State ptrInitialState, const Options& options)
    : state(std::make_shared<State>(ptrInitialState)),
      cudaState(std::make_shared<CudaState>(state)),
      resourceTransfer(cudaState),
      reproduction(cudaState)
{
    // TODO: Initialize soil diffusion for CUDA backend
}

void CudaSimulator::step(const Options& options) {
    if (options.enableResourceTransfer) {
        resourceTransfer.step(options);
    }
    if (options.enableCellMultiplication) {
        reproduction.step(options);
    }
}

const State& CudaSimulator::getState() const {
    return cudaState->getState();
}