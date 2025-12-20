
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
    
    // Nutrients layer
    ImGui::Checkbox("Nutrients", &renderingOptions.showNutrients);
    if (renderingOptions.showNutrients) {
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        ImGui::SliderFloat("##nut_opacity", &renderingOptions.nutrientsOpacity, 0.0f, 1.0f, "%.2f");
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
        ImGui::Checkbox("Nutrients System", &opts.enableNutrients);
        
        ImGui::Spacing();
        ImGui::Text("Nutrient Parameters:");
        
        ImGui::PushItemWidth(150);
        
        ImGui::SliderFloat("Diffusion Rate", &opts.nutrientDiffusionRate, 0.0f, 1.0f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of nutrient diffusion between adjacent cells");
        }
        
        ImGui::SliderFloat("Absorption Rate", &opts.nutrientAbsorptionRate, 0.0f, 2.0f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Maximum nutrients a cell can absorb per tick");
        }
        
        ImGui::SliderFloat("Regeneration Rate", &opts.nutrientRegenerationRate, 0.0f, 1.0f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rate of nutrient regeneration in soil layer");
        }
        
        ImGui::SliderInt("Soil Layer Height", &opts.soilLayerHeight, 1, 20);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Number of bottom rows that act as soil (nutrient sources)");
        }
        
        ImGui::SliderFloat("Max Nutrient", &opts.maxNutrient, 10.0f, 500.0f, "%.1f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Maximum nutrient concentration per cell");
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
