
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
    
    // Resources layer
    ImGui::Checkbox("Resources", &renderingOptions.showResources);
    if (renderingOptions.showResources) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##res_opacity", &renderingOptions.resourcesOpacity, 0.0f, 1.0f, "%.2f");
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
    
    // Soil Water layer
    ImGui::Checkbox("Soil Water", &renderingOptions.showSoilWater);
    if (renderingOptions.showSoilWater) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##water_opacity", &renderingOptions.soilWaterOpacity, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
    }
    
    // Soil Mineral layer
    ImGui::Checkbox("Soil Mineral", &renderingOptions.showSoilMineral);
    if (renderingOptions.showSoilMineral) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##mineral_opacity", &renderingOptions.soilMineralOpacity, 0.0f, 1.0f, "%.2f");
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
