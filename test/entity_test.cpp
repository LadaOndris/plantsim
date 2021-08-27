//
// Created by lada on 8/27/21.
//

#include <gtest/gtest.h>
#include <memory>
#include "Entity.h"
#include "Cell.h"
#include "WorldState.h"

class EmptyProcess : public Process {
private:
    int doGetGenesCount() const override {
        // The process requires two genes to define its operation.
        return 2;
    }

    void doInvoke(WorldState &worldState, std::shared_ptr<Cell> &cell) override {
        std::cout << "Invoking empty process." << std::endl;
    }
};

TEST(EntityWithProcess, InvokesProcessForEachCell) {
    // Create a new world state
    WorldState state(10, 10);

    // Create processes
    std::unique_ptr<Process> emptyProcess = std::make_unique<EmptyProcess>();
    std::vector<std::unique_ptr<Process>> processes;
    processes.push_back(std::move(emptyProcess));

    // Create entity
    Entity entity(processes);

    // Add cells to entity
    std::shared_ptr<Cell> cell(new Cell(1, 2));
    entity.addCell(cell);

    // Invoke processes on the entity
    entity.invokeProcesses(state);
}