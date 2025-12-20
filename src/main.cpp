#include <iostream>
#include <memory>

#include "ApplicationConfig.h"
#include "visualisation/GraphicsContext.h"
#include "visualisation/RenderLoop.h"
#include "visualisation/rendering/RendererRegistry.h"
#include "visualisation/rendering/GuiFrameRenderer.h"
#include "visualisation/rendering/WorldStateRenderer.h"
#include "visualisation/rendering/converters/AxialRectangularMapToMeshConverter.h"

#include "simulation/SimulatorFactory.h"
#include "simulation/GridTopology.h"
#include "simulation/initializers/Initializers.h"

namespace {

/**
 * @brief Create the initial simulation state based on configuration.
 */
State createInitialState(const ApplicationConfig& config) {
    using namespace initializers;

    GridTopology topology{config.gridWidth, config.gridHeight};
    OffsetCoord center{config.gridWidth / 2, 20};

    StateInitializer initializer{
        // Set center cells to Cell type
        PolicyApplication{CircleRegion{center, 1}, SetCellType{CellState::Cell}},
        // Set resources at center cell
        PolicyApplication{
            SingleCell{center},
            SetResource{FixedAmount{0.0f}}
        },
        // Initialize soil layer (bottom 5 rows) with nutrients
        PolicyApplication{
            BottomRowsRegion{5},
            SetNutrient{FixedAmount{100.0f}}
        }
    };

    return initializer.initialize(topology);
}

/**
 * @brief Register all renderers with the registry.
 */
void registerRenderers(
    RendererRegistry& registry,
    const GridTopology& topology,
    std::unique_ptr<ISimulator>& simulatorPtr,
    const ApplicationConfig& config
) {
    // World state renderer (simulation visualization)
    auto worldStateProgram = std::make_shared<ShaderProgram>();
    worldStateProgram->addShader(
        std::make_unique<Shader>(config.vertexShaderPath.c_str(), ShaderType::Vertex)
    );
    worldStateProgram->addShader(
        std::make_unique<Shader>(config.fragmentShaderPath.c_str(), ShaderType::Fragment)
    );

    AxialRectangularMapToMeshConverter mapConverter{};
    auto worldStateRenderer = std::make_shared<WorldStateRenderer>(
        topology, simulatorPtr, mapConverter, worldStateProgram
    );
    registry.add(worldStateRenderer);

    // GUI renderer (ImGui overlay)
    auto guiRenderer = std::make_shared<GuiFrameRenderer>();
    guiRenderer->initializeWithOptions(config.simulationOptions, config.stepsPerFrame);
    registry.add(guiRenderer);
}

} // namespace

int main() {
    std::cout << "Starting the application..." << std::endl;

    ApplicationConfig config{};

    GraphicsContext context;
    if (!context.initialize(config.window)) {
        return EXIT_FAILURE;
    }

    GridTopology topology{config.gridWidth, config.gridHeight};
    State initialState = createInitialState(config);
    std::unique_ptr<ISimulator> simulator = SimulatorFactory::create(std::move(initialState), config.simulationOptions);

    RendererRegistry registry;
    registerRenderers(registry, topology, simulator, config);

    if (!registry.initializeAll()) {
        registry.destroyAll();
        return EXIT_FAILURE;
    }

    // Get the GUI renderer as the options provider
    auto guiRenderer = registry.get<GuiFrameRenderer>();
    if (!guiRenderer) {
        std::cerr << "[ERROR] GuiFrameRenderer not created" << std::endl;
        registry.destroyAll();
        return EXIT_FAILURE;
    }

    // Create state creator lambda for reset functionality
    auto stateCreator = [&config]() -> State {
        return createInitialState(config);
    };

    RenderLoop::run(context, registry, *guiRenderer, simulator, config, stateCreator);

    registry.destroyAll();

    return EXIT_SUCCESS;
}

