
#include <gtest/gtest.h>
#include <memory>
#include <plants/AxialRectangularMap.h>
#include "plants/Entity.h"
#include "plants/Cell.h"
#include "plants/WorldState.h"
#include "../dummies/EmptyProcess.h"

//TEST(EntityWithProcess, InvokesProcessForEachCell) {
//    // Create processes
//    std::shared_ptr<Process> emptyProcess = std::make_shared<EmptyProcess>();
//    std::vector<std::shared_ptr<Process>> processes;
//    processes.push_back(std::move(emptyProcess));
//    auto map = std::make_shared<AxialRectangularMap>(10, 10);
//    WorldState state(map, processes);
//    auto entity = state.getEntity();
//    // Add cells to entity
//    std::shared_ptr<Cell> cell = std::make_shared<Cell>(1, 2);
//    entity->addCell(cell);
//    state.invokeProcesses();
//}