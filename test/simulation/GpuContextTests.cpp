
#include <gtest/gtest.h>
#include "simulation/GpuContext.h"
#include "simulation/CellState.h"
#include <vector>
#include <utility>
#include <numeric>

/**
 * @brief Performs a single simulation step expecting one resource to be transferred.
 *
 * Two points are of type cell with only one having resources.
 * During the simulation step, one resource is transferred to a neighbor.
 */
TEST(ResourcesSimulator, SingleStep) {

    sycl::queue q;
    const int width = 10;
    const int height = 10;
    const int initialResources = 1;
    const size_t totalCells = width * height;
    
    std::vector<int> res_host(totalCells, 0);
    std::vector<int> type_host(totalCells, static_cast<int>(CellState::Type::Air));
    std::vector<std::pair<int, int>> neigh_host = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    // Initial setup: two neighboring cells, one with resources
    const int r = 1;
    const int q_ = 1;
    const int source_idx = r * width + q_;
    
    // Set up source cell
    res_host[source_idx] = initialResources;
    type_host[source_idx] = static_cast<int>(CellState::Type::Cell);
    
    // Set up neighboring cell (right neighbor)
    const int neighbor_q = q_ + 1;
    const int neighbor_r = r;
    const int neighbor_idx = neighbor_r * width + neighbor_q;
    type_host[neighbor_idx] = static_cast<int>(CellState::Type::Cell);

    ResourcesSimulator sim(q, width, height, res_host, type_host, neigh_host);
    sim.step();

    auto res_out = sim.copyResourcesToHost();

    ASSERT_EQ(res_out[source_idx], 0);

    // The logic in Pass1Scatter is deterministic: `(r + q) % numN`
    // const int chosen_neighbor_idx = (r + q_) % neigh_host.size();
    // auto offset = neigh_host[chosen_neighbor_idx];
    // const int nq = q_ + offset.first;
    // const int nr = r + offset.second;
    // const int expected_neighbor_idx = nr * width + nq;

    // One of the neighbors should have 1 resource.
    ASSERT_EQ(res_out[neighbor_idx], initialResources);

    // The sum of all resources should still be 1
    const int total_resources = std::accumulate(res_out.begin(), res_out.end(), 0);
    ASSERT_EQ(total_resources, initialResources);
}
