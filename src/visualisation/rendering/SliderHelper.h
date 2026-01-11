#pragma once

#include <imgui.h>
#include <cmath>
#include <algorithm>

/**
 * Non-linear slider using: f(x) = M * x / (x + k * (1 - x))
 * where k = M/d - 1. Default value appears at slider midpoint (x = 0.5).
 */
namespace NonLinearSlider {

    constexpr float DEFAULT_MAX = 10.0f;
    constexpr float DEFAULT_MIN = 0.0f;

    inline float toConfigValue(float sliderPos, float defaultVal, float maxVal = DEFAULT_MAX) {
        sliderPos = std::clamp(sliderPos, 0.0f, 1.0f);
        
        // Degenerate cases: use linear mapping
        if (defaultVal < 1e-6f || std::abs(defaultVal - maxVal) < 1e-6f) {
            return sliderPos * maxVal;
        }
        
        if (sliderPos < 1e-6f) return 0.0f;
        if (sliderPos > 1.0f - 1e-6f) return maxVal;
        
        float k = (maxVal / defaultVal) - 1.0f;
        float denominator = sliderPos + k * (1.0f - sliderPos);
        if (std::abs(denominator) < 1e-9f) return 0.0f;
        
        return (maxVal * sliderPos) / denominator;
    }

    inline float toSliderPos(float configVal, float defaultVal, float maxVal = DEFAULT_MAX) {
        configVal = std::clamp(configVal, 0.0f, maxVal);
        
        // Degenerate cases: use linear mapping
        if (defaultVal < 1e-6f || std::abs(defaultVal - maxVal) < 1e-6f) {
            return configVal / maxVal;
        }
        
        if (configVal < 1e-6f) return 0.0f;
        if (configVal > maxVal - 1e-6f) return 1.0f;
        
        float k = (maxVal / defaultVal) - 1.0f;
        float denominator = maxVal + configVal * (k - 1.0f);
        if (std::abs(denominator) < 1e-9f) return 0.5f;
        
        return (k * configVal) / denominator;
    }

    inline bool SliderFloat(const char* label, float* value, float defaultVal, 
                            const char* tooltip = nullptr, 
                            const char* format = "%.4f",
                            float maxVal = DEFAULT_MAX) {
        float sliderPos = toSliderPos(*value, defaultVal, maxVal);
        
        ImGui::PushID(label);
        bool changed = ImGui::SliderFloat(label, &sliderPos, 0.0f, 1.0f, format);
        
        if (changed) {
            *value = toConfigValue(sliderPos, defaultVal, maxVal);
        }
        
        if (tooltip && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", tooltip);
        }
        
        ImGui::PopID();
        return changed;
    }

    /// Displays actual config value in slider, label to the right
    inline bool SliderFloatWithValue(const char* label, float* value, float defaultVal,
                                     const char* tooltip = nullptr,
                                     const char* format = "%.4f",
                                     float maxVal = DEFAULT_MAX) {
        float sliderPos = toSliderPos(*value, defaultVal, maxVal);
        
        char valueStr[64];
        snprintf(valueStr, sizeof(valueStr), format, *value);
        
        ImGui::PushID(label);
        bool changed = ImGui::SliderFloat("##slider", &sliderPos, 0.0f, 1.0f, valueStr);
        
        if (changed) {
            *value = toConfigValue(sliderPos, defaultVal, maxVal);
        }
        
        ImGui::SameLine();
        ImGui::Text("%s", label);
        
        if (tooltip && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", tooltip);
        }
        
        ImGui::PopID();
        return changed;
    }

    /// Linear slider for bounded values (e.g., [0, 1])
    inline bool SliderFloatLinear(const char* label, float* value, float minVal, float maxVal,
                                  const char* tooltip = nullptr,
                                  const char* format = "%.2f") {
        bool changed = ImGui::SliderFloat(label, value, minVal, maxVal, format);
        
        if (tooltip && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", tooltip);
        }
        
        return changed;
    }

}
