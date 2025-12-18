#include "CudaSimulator.h"

CudaSimulator::CudaSimulator(State ptrInitialState)
    : state(std::make_shared<State>(ptrInitialState)),
      cudaState(std::make_shared<CudaState>(state)),
      resourceTransfer(cudaState)
{
}

void CudaSimulator::step(const Options& options) {
    if (options.enableResourceTransfer) {
        resourceTransfer.step(options);
    }
}

const State& CudaSimulator::getState() const {
    return cudaState->getState();
}