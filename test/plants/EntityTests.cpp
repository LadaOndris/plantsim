//
// Created by lada on 8/27/21.
//

#include <gtest/gtest.h>
#include <memory>
#include "plants/Entity.h"
#include "plants/Cell.h"
#include "plants/WorldState.h"
#include "../dummies/EmptyProcess.h"

TEST(EntityWithProcess, InvokesProcessForEachCell) {
    // Create processes
    std::shared_ptr<Process> emptyProcess = std::make_shared<EmptyProcess>();
    std::vector<std::shared_ptr<Process>> processes;
    processes.push_back(std::move(emptyProcess));

    // Create a new world state
    WorldState state(10, 10, processes);


    // Create entity
    auto entity = state.getEntity();

    // Add cells to entity
    std::shared_ptr<Cell> cell = std::make_shared<Cell>(1, 2);
    entity->addCell(cell);

    // Invoke processes on the entity
    state.invokeProcesses();
}