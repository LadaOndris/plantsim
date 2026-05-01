#include <iostream>
#include <memory>

#include "common/ApplicationConfig.h"
#include "simulation/initializers/actions/SetCellType.h"
#include "simulation/initializers/actions/SetStateField.h"
#include "simulation/initializers/amounts/FixedAmount.h"
#include "visualisation/GraphicsContext.h"
#include "visualisation/RenderLoop.h"
#include "visualisation/rendering/RendererRegistry.h"
#include "visualisation/rendering/GuiFrameRenderer.h"
#include "visualisation/rendering/WorldStateRenderer.h"
#include "visualisation/rendering/converters/AxialRectangularMapToMeshConverter.h"

#include "simulation/SimulatorFactory.h"
#include "simulation/GridTopology.h"
#include "simulation/initializers/Initializers.h"
#include "visualisation/rendering/shaders/ProgramBuilder.h"
#include "visualisation/rendering/shaders/Shader.h"

namespace {

State createInitialState(const ApplicationConfig& config) {
    using namespace initializers;

    GridTopology topology{config.gridWidth, config.gridHeight};

    int initialCellRow = config.simulationOptions.soilLayerHeight;
    OffsetCoord seedSoil{config.gridWidth / 2, initialCellRow};

    StateInitializer initializer{
        PolicyApplication{
            BottomRowsRegion{config.simulationOptions.soilLayerHeight},
            SeedSoil{
                FixedAmount{config.simulationOptions.soilWaterTarget / 4.0f},
                FixedAmount{config.simulationOptions.soilMineralTarget}
            }
        },
        PolicyApplication{
            CircleRegion{seedSoil, 1}, 
            SeedCell{FixedAmount{config.simulationOptions.childInitialHealth}}
        }
    };

    return initializer.initialize(topology);
}

void registerRenderers(
    RendererRegistry& registry,
    const GridTopology& topology,
    std::unique_ptr<ISimulator>& simulatorPtr,
    const ApplicationConfig& config
) {
    ShaderProgram worldStateProgram = ProgramBuilder()
        .addShader(config.vertexShaderPath, ShaderType::Vertex)
        .addShader(config.fragmentShaderPath, ShaderType::Fragment)
        .build();

    AxialRectangularMapToMeshConverter mapConverter{};
    auto worldStateRenderer = std::make_shared<WorldStateRenderer>(
        topology, simulatorPtr, mapConverter, std::move(worldStateProgram)
    );
    registry.add(worldStateRenderer);

    auto guiRenderer = std::make_shared<GuiFrameRenderer>();
    guiRenderer->initializeWithOptions(config.simulationOptions, config.stepsPerFrame);
    registry.add(guiRenderer);
}

void runApplication() {
    ApplicationConfig config{};

    GraphicsContext context;
    if (!context.initialize(config.window)) {
        throw std::runtime_error("Failed to initialize graphics context");
    }

    GridTopology topology{config.gridWidth, config.gridHeight};
    State initialState = createInitialState(config);
    std::unique_ptr<ISimulator> simulator = SimulatorFactory::create(std::move(initialState), config.simulationOptions);

    RendererRegistry registry;
    registerRenderers(registry, topology, simulator, config);

    auto guiRenderer = registry.get<GuiFrameRenderer>();
    if (!guiRenderer) {
        throw std::runtime_error("GuiFrameRenderer not created");
    }

    auto stateCreator = [&config]() -> State {
        return createInitialState(config);
    };

    RenderLoop::run(context, registry, *guiRenderer, simulator, config, stateCreator);
}

} // namespace

int main() {
    std::cout << "Starting the application..." << std::endl;

    try {
        runApplication();
    } catch (const std::exception& ex) {
        std::cerr << "Application error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

