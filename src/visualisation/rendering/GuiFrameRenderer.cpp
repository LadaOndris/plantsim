
#include <iostream>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "GuiFrameRenderer.h"

GuiFrameRenderer::GuiFrameRenderer() = default;

void GuiFrameRenderer::initializeWithOptions(const Options& initialOptions, int stepsPerFrame) {
    simulationControl.pendingOptions = initialOptions;
    simulationControl.activeOptions = initialOptions;
    simulationControl.stepsPerFrame = stepsPerFrame;
}

bool GuiFrameRenderer::initialize() {
    return true;
}

void GuiFrameRenderer::destroy() {
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
        
        ImGui::SliderFloat("Water Target", &opts.soilWaterTarget, 0.0f, 5.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Equilibrium water level in soil");
        }
        
        ImGui::SliderFloat("Water Regen Rate", &opts.soilWaterRegenRate, 0.0f, 0.1f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of water regeneration toward target");
        }
        
        ImGui::SliderFloat("Water Diffusivity", &opts.soilWaterDiffusivity, 0.0f, 0.5f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of water diffusion between soil tiles");
        }
        
        ImGui::Separator();
        ImGui::Text("Minerals:");
        
        ImGui::SliderFloat("Mineral Target", &opts.soilMineralTarget, 0.0f, 5.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Equilibrium mineral level in soil");
        }
        
        ImGui::SliderFloat("Mineral Regen Rate", &opts.soilMineralRegenRate, 0.0f, 0.05f, "%.4f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of mineral regeneration toward target");
        }
        
        ImGui::SliderFloat("Mineral Diffusivity", &opts.soilMineralDiffusivity, 0.0f, 0.5f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of mineral diffusion between soil tiles");
        }
        
        ImGui::Separator();
        ImGui::Text("Internal Transport:");
        
        ImGui::SliderFloat("Sugar Transport", &opts.sugarTransportRate, 0.0f, 0.5f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of sugar diffusion between plant cells");
        }
        
        ImGui::SliderFloat("Water Transport", &opts.waterTransportRate, 0.0f, 0.5f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of water diffusion between plant cells");
        }
        
        ImGui::SliderFloat("Mineral Transport", &opts.mineralTransportRate, 0.0f, 0.5f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of mineral diffusion between plant cells");
        }
        
        ImGui::Separator();
        ImGui::Text("Light & Photosynthesis:");
        
        ImGui::SliderFloat("Light Intensity", &opts.lightTopIntensity, 0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Intensity of light at top of grid");
        }
        
        ImGui::SliderFloat("Plant Light Absorb", &opts.plantLightAbsorb, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fraction of light absorbed by plant cells");
        }
        
        ImGui::SliderFloat("Soil Light Absorb", &opts.soilLightAbsorb, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fraction of light absorbed by soil");
        }
        
        ImGui::SliderFloat("Photo Max Rate", &opts.photoMaxRate, 0.0f, 0.5f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Maximum sugar production rate per tick");
        }
        
        ImGui::SliderFloat("Light Half-Sat", &opts.lightHalfSat, 0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Half-saturation constant for light");
        }
        
        ImGui::SliderFloat("Water Half-Sat", &opts.waterHalfSat, 0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Half-saturation constant for water in photosynthesis");
        }
        
        ImGui::SliderFloat("Water Per Sugar", &opts.waterPerSugar, 0.0f, 5.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Water consumed per unit sugar produced");
        }
        
        ImGui::Separator();
        ImGui::Text("Maintenance & Death:");
        
        ImGui::SliderFloat("Sugar Maint Cost", &opts.sugarMaintCost, 0.0f, 0.2f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Sugar consumed per plant cell per tick for maintenance");
        }
        
        ImGui::SliderFloat("Water Maint Cost", &opts.waterMaintCost, 0.0f, 0.1f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Base water consumed per plant cell per tick");
        }
        
        ImGui::SliderFloat("Water Light Loss", &opts.waterLightLoss, 0.0f, 0.1f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Extra water loss scaled by light (transpiration)");
        }
        
        ImGui::SliderFloat("Sugar Deficit Damage", &opts.sugarDeficitDamage, 0.0f, 10.0f, "%.1f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Health damage per unit sugar deficit");
        }
        
        ImGui::SliderFloat("Water Deficit Damage", &opts.waterDeficitDamage, 0.0f, 10.0f, "%.1f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Health damage per unit water deficit");
        }
        
        ImGui::SliderFloat("Health Regen Rate", &opts.healthRegenRate, 0.0f, 0.2f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Health regeneration per tick when resources are sufficient");
        }
        
        ImGui::Separator();
        ImGui::Text("Dead Decay:");
        
        ImGui::SliderFloat("Decay Rate", &opts.deadDecayRate, 0.0f, 0.2f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fraction of dead cell resources released per tick");
        }
        
        ImGui::SliderFloat("Dead to Soil Bias", &opts.deadToSoilBias, 0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("How strongly dead cells return minerals to soil");
        }
        
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
