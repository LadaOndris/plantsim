
#include "simulation/SimulatorFactory.h"

#if defined(BACKEND_CPU)
    #include "simulation/cpu/CpuSimulator.h"
#elif defined(BACKEND_CUDA)
    #include "simulation/cuda/CudaSimulator.h"
#elif defined(BACKEND_SYCL)
    #include "simulation/sycl/SyclSimulator.h"
#endif

#include <iostream>
#include <memory>


std::unique_ptr<ISimulator> SimulatorFactory::create(State initialState, const Options& options) {
        std::cout << "[INFO] Using " << getBackendName() << " backend" << std::endl;

#if defined(BACKEND_CPU)
        return std::make_unique<CpuSimulator>(std::move(initialState), options);
#elif defined(BACKEND_CUDA)
        return std::make_unique<CudaSimulator>(std::move(initialState), options);
#elif defined(BACKEND_SYCL)
        return std::make_unique<SyclSimulator>(std::move(initialState), options);
#else
        #error "No backend defined. Define BACKEND_CPU, BACKEND_CUDA, or BACKEND_SYCL"
#endif
}

const char* SimulatorFactory::getBackendName() {
#if defined(BACKEND_CPU)
        return "CPU";
#elif defined(BACKEND_CUDA)
        return "CUDA";
#elif defined(BACKEND_SYCL)
        return "SYCL";
#else
        return "Unknown";
#endif
}