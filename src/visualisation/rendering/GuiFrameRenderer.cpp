
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "GuiFrameRenderer.h"
#include "SliderHelper.h"

void GuiFrameRenderer::initializeWithOptions(const Options& initialOptions, int stepsPerFrame) {
    simulationControl.pendingOptions = initialOptions;
    simulationControl.activeOptions = initialOptions;
    simulationControl.stepsPerFrame = stepsPerFrame;
}

void GuiFrameRenderer::render(const WindowDefinition &window, const RenderingOptions &options) {
    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Create main control window
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 450), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Simulation Control", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    renderPlaybackControls();
    ImGui::Separator();
    
    renderVisualizationControls();
    ImGui::Separator();
    
    renderOptionsPanel();
    
    ImGui::End();
    
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GuiFrameRenderer::renderPlaybackControls() {
    ImGui::Text("Playback");
    ImGui::Spacing();
    
    // Play/Pause button
    if (simulationControl.isPaused) {
        if (ImGui::Button("Play", ImVec2(80, 0))) {
            simulationControl.isPaused = false;
        }
    } else {
        if (ImGui::Button("Pause", ImVec2(80, 0))) {
            simulationControl.isPaused = true;
        }
    }
    
    ImGui::SameLine();
    
    // Reset button
    if (ImGui::Button("Reset", ImVec2(80, 0))) {
        simulationControl.shouldReset = true;
    }
    
    // Show pending changes indicator next to reset button
    if (simulationControl.hasPendingChanges()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "(*)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Pending option changes will be applied on Reset");
        }
    }
    
    ImGui::Spacing();
    
    // Speed control
    ImGui::Text("Speed (steps/frame):");
    ImGui::SliderInt("##speed", &simulationControl.stepsPerFrame, 1, 200);
}

void GuiFrameRenderer::renderVisualizationControls() {
    ImGui::Text("Visualization Layers");
    ImGui::Spacing();
    
    // Sugar layer
    ImGui::Checkbox("Sugar", &renderingOptions.showSugar);
    if (renderingOptions.showSugar) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##sugar_opacity", &renderingOptions.sugarOpacity, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
    }
    
    // Cell types layer
    ImGui::Checkbox("Cell Types", &renderingOptions.showCellTypes);
    if (renderingOptions.showCellTypes) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##cell_opacity", &renderingOptions.cellTypesOpacity, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
    }
    
    // Water layer (plant water)
    ImGui::Checkbox("Water", &renderingOptions.showWater);
    if (renderingOptions.showWater) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##water_opacity", &renderingOptions.waterOpacity, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
    }
    
    // Mineral layer (plant mineral)
    ImGui::Checkbox("Mineral", &renderingOptions.showMineral);
    if (renderingOptions.showMineral) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##mineral_opacity", &renderingOptions.mineralOpacity, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
    }
    
    // Health layer
    ImGui::Checkbox("Health", &renderingOptions.showHealth);
    if (renderingOptions.showHealth) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##health_opacity", &renderingOptions.healthOpacity, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
    }
    
    // Light layer
    ImGui::Checkbox("Light", &renderingOptions.showLight);
    if (renderingOptions.showLight) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##light_opacity", &renderingOptions.lightOpacity, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
    }
}

void GuiFrameRenderer::renderOptionsPanel() {
    if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderPendingChangesIndicator();
        
        Options& opts = simulationControl.pendingOptions;
        
        // Reset to Defaults button
        if (ImGui::Button("Reset All to Defaults")) {
            opts = DefaultOptions;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Reset all simulation parameters to their default values");
        }
        
        ImGui::Spacing();
        ImGui::Text("Features:");
        ImGui::Checkbox("Resource Transfer", &opts.enableResourceTransfer);
        ImGui::Checkbox("Cell Multiplication", &opts.enableCellMultiplication);
        ImGui::Checkbox("Soil System", &opts.enableSoilSystem);
        ImGui::Checkbox("Maintenance & Death", &opts.enableMaintenanceAndDeath);
        ImGui::Checkbox("Dead Decay", &opts.enableDeadDecay);
        
        ImGui::Spacing();
        ImGui::Text("Soil Parameters:");
        
        ImGui::PushItemWidth(150);
        
        ImGui::SliderInt("Soil Layer Height", &opts.soilLayerHeight, 1, 30);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Number of bottom rows that act as soil");
        }
        
        ImGui::Separator();
        ImGui::Text("Water:");
        
        NonLinearSlider::SliderFloatWithValue("Water Target", &opts.soilWaterTarget, 
            DefaultOptions.soilWaterTarget, "Equilibrium water level in soil", "%.2f");
        
        NonLinearSlider::SliderFloatWithValue("Water Regen Rate", &opts.soilWaterRegenRate, 
            DefaultOptions.soilWaterRegenRate, "Rate of water regeneration toward target", "%.4f");
        
        NonLinearSlider::SliderFloatWithValue("Water Diffusivity", &opts.soilWaterDiffusivity, 
            DefaultOptions.soilWaterDiffusivity, "Rate of water diffusion between soil tiles", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Water Uptake Rate", &opts.waterUptakeRate, 
            DefaultOptions.waterUptakeRate, "Max water pulled from soil per edge per tick", "%.3f");
        
        ImGui::Separator();
        ImGui::Text("Minerals:");
        
        NonLinearSlider::SliderFloatWithValue("Mineral Target", &opts.soilMineralTarget, 
            DefaultOptions.soilMineralTarget, "Equilibrium mineral level in soil", "%.2f");
        
        NonLinearSlider::SliderFloatWithValue("Mineral Regen Rate", &opts.soilMineralRegenRate, 
            DefaultOptions.soilMineralRegenRate, "Rate of mineral regeneration toward target", "%.5f");
        
        NonLinearSlider::SliderFloatWithValue("Mineral Diffusivity", &opts.soilMineralDiffusivity, 
            DefaultOptions.soilMineralDiffusivity, "Rate of mineral diffusion between soil tiles", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Mineral Uptake Rate", &opts.mineralUptakeRate, 
            DefaultOptions.mineralUptakeRate, "Max mineral pulled from soil per edge per tick", "%.3f");
        
        ImGui::Separator();
        ImGui::Text("Internal Transport:");
        
        NonLinearSlider::SliderFloatWithValue("Sugar Transport", &opts.sugarTransportRate, 
            DefaultOptions.sugarTransportRate, "Rate of sugar diffusion between plant cells", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Water Transport", &opts.waterTransportRate, 
            DefaultOptions.waterTransportRate, "Rate of water diffusion between plant cells", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Mineral Transport", &opts.mineralTransportRate, 
            DefaultOptions.mineralTransportRate, "Rate of mineral diffusion between plant cells", "%.3f");
        
        ImGui::Separator();
        ImGui::Text("Light & Photosynthesis:");
        
        NonLinearSlider::SliderFloatWithValue("Light Intensity", &opts.lightTopIntensity, 
            DefaultOptions.lightTopIntensity, "Intensity of light at top of grid", "%.2f");
        
        // Light absorption values are bounded [0, 1] - use linear slider
        NonLinearSlider::SliderFloatLinear("Plant Light Absorb", &opts.plantLightAbsorb, 0.0f, 1.0f,
            "Fraction of light absorbed by plant cells", "%.2f");
        
        NonLinearSlider::SliderFloatLinear("Dead Light Absorb", &opts.deadLightAbsorb, 0.0f, 1.0f,
            "Fraction of light absorbed by dead cells", "%.2f");
        
        NonLinearSlider::SliderFloatLinear("Soil Light Absorb", &opts.soilLightAbsorb, 0.0f, 1.0f,
            "Fraction of light absorbed by soil", "%.2f");
        
        NonLinearSlider::SliderFloatWithValue("Photo Max Rate", &opts.photoMaxRate, 
            DefaultOptions.photoMaxRate, "Maximum sugar production rate per tick", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Light Half-Sat", &opts.lightHalfSat, 
            DefaultOptions.lightHalfSat, "Half-saturation constant for light", "%.2f");
        
        NonLinearSlider::SliderFloatWithValue("Water Half-Sat", &opts.waterHalfSat, 
            DefaultOptions.waterHalfSat, "Half-saturation constant for water in photosynthesis", "%.2f");
        
        NonLinearSlider::SliderFloatWithValue("Water Per Sugar", &opts.waterPerSugar, 
            DefaultOptions.waterPerSugar, "Water consumed per unit sugar produced", "%.2f");
        
        ImGui::Separator();
        ImGui::Text("Maintenance & Death:");
        
        NonLinearSlider::SliderFloatWithValue("Sugar Maint Cost", &opts.sugarMaintCost, 
            DefaultOptions.sugarMaintCost, "Sugar consumed per plant cell per tick for maintenance", "%.4f");
        
        NonLinearSlider::SliderFloatWithValue("Water Maint Cost", &opts.waterMaintCost, 
            DefaultOptions.waterMaintCost, "Base water consumed per plant cell per tick", "%.4f");
        
        NonLinearSlider::SliderFloatWithValue("Water Light Loss", &opts.waterLightLoss, 
            DefaultOptions.waterLightLoss, "Extra water loss scaled by light (transpiration)", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Sugar Deficit Damage", &opts.sugarDeficitDamage, 
            DefaultOptions.sugarDeficitDamage, "Health damage per unit sugar deficit", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Water Deficit Damage", &opts.waterDeficitDamage, 
            DefaultOptions.waterDeficitDamage, "Health damage per unit water deficit", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Health Regen Rate", &opts.healthRegenRate, 
            DefaultOptions.healthRegenRate, "Health regeneration per tick when resources are sufficient", "%.3f");
        
        ImGui::Separator();
        ImGui::Text("Dead Decay:");
        
        NonLinearSlider::SliderFloatWithValue("Decay Rate", &opts.deadDecayRate, 
            DefaultOptions.deadDecayRate, "Fraction of dead cell resources released per tick", "%.3f");
        
        NonLinearSlider::SliderFloatWithValue("Dead to Soil Bias", &opts.deadToSoilBias, 
            DefaultOptions.deadToSoilBias, "How strongly dead cells return minerals to soil", "%.2f");
        
        ImGui::Separator();
        ImGui::Text("Time Step:");
        
        NonLinearSlider::SliderFloatWithValue("dt", &opts.dt, 
            DefaultOptions.dt, "Time step multiplier for physics", "%.2f");
        
        ImGui::PopItemWidth();
    }
}

void GuiFrameRenderer::renderPendingChangesIndicator() {
    if (simulationControl.hasPendingChanges()) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), 
            "* Changes pending - press Reset to apply");
        ImGui::Spacing();
    }
}

RenderingOptions GuiFrameRenderer::getRenderingOptions() const {
    return renderingOptions;
}

SimulationControl& GuiFrameRenderer::getSimulationControl() {
    return simulationControl;
}

const SimulationControl& GuiFrameRenderer::getSimulationControl() const {
    return simulationControl;
}
