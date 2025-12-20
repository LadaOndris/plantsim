#pragma once

#include "ISimulator.h"
#include "State.h"
#include <memory>
#include <iostream>

#if defined(BACKEND_CPU)
    #include "simulation/cpu/CpuSimulator.h"
#elif defined(BACKEND_CUDA)
    #include "simulation/cuda/CudaSimulator.h"
#elif defined(BACKEND_SYCL)
    #include "simulation/sycl/SyclSimulator.h"
#endif

/**
 * @brief Factory for creating simulator instances based on compile-time backend.
 * 
 * Abstracts the backend selection logic and provides a unified interface
 * for simulator creation.
 */
class SimulatorFactory {
public:
    /**
     * @brief Create a simulator with the configured backend.
     * 
     * @param initialState The initial simulation state
     * @return Unique pointer to the created simulator
     */
    static std::unique_ptr<ISimulator> create(State initialState, const Options& options) {
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

    /**
     * @brief Get the name of the configured backend.
     */
    static constexpr const char* getBackendName() {
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
};
