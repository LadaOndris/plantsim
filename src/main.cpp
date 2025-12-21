#include <iostream>
#include <memory>

#include "ApplicationConfig.h"
#include "simulation/initializers/actions/SetStateField.h"
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

    int initialCellRow = config.simulationOptions.soilLayerHeight;
    OffsetCoord seedSoil{config.gridWidth / 2, initialCellRow};

    StateInitializer initializer{
        // Set center cells to Cell type with initial water for photosynthesis
        PolicyApplication{
            CircleRegion{seedSoil, 1}, 
            CompositeAction{
                SetCellType{CellState::Cell},
                SetPlantWater(FixedAmount{0.1f})
            }
        },
        // Initialize soil layer with water and minerals
        PolicyApplication{
            BottomRowsRegion{config.simulationOptions.soilLayerHeight},
            CompositeAction{
                SetSoilWater(FixedAmount{config.simulationOptions.soilWaterTarget}),
                SetSoilMineral(FixedAmount{config.simulationOptions.soilMineralTarget})
            }
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

