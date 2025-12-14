#pragma once

#include "Renderer.h"
#include <memory>
#include <vector>
#include <typeinfo>
#include <iostream>

/**
 * @brief Registry for managing renderer lifecycle.
 * 
 * Provides centralized management of renderers including initialization,
 * rendering, and cleanup. Renderers are rendered in registration order.
 */
class RendererRegistry {
public:
    RendererRegistry() = default;
    ~RendererRegistry() = default;

    // Movable but not copyable
    RendererRegistry(const RendererRegistry&) = delete;
    RendererRegistry& operator=(const RendererRegistry&) = delete;
    RendererRegistry(RendererRegistry&&) = default;
    RendererRegistry& operator=(RendererRegistry&&) = default;

    /**
     * @brief Add a renderer to the registry.
     * 
     * Renderers are rendered in the order they are added.
     * 
     * @param renderer Shared pointer to the renderer
     */
    void add(std::shared_ptr<Renderer> renderer) {
        renderers.push_back(std::move(renderer));
    }

    /**
     * @brief Initialize all registered renderers.
     * 
     * @return true if all renderers initialized successfully
     */
    bool initializeAll() {
        for (const auto& renderer : renderers) {
            if (!renderer->initialize()) {
                auto& rendererObject = *renderer;
                std::cerr << "[ERROR] Failed to initialize renderer: " 
                          << typeid(rendererObject).name() << std::endl;
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Render all registered renderers.
     * 
     * @param window Current window definition
     * @param options Current rendering options
     */
    void renderAll(const WindowDefinition& window, const RenderingOptions& options) {
        for (const auto& renderer : renderers) {
            renderer->render(window, options);
        }
    }

    /**
     * @brief Destroy all registered renderers.
     * 
     * Should be called before graphics context shutdown.
     */
    void destroyAll() {
        for (const auto& renderer : renderers) {
            renderer->destroy();
        }
        renderers.clear();
    }

    /**
     * @brief Get a renderer by type.
     * 
     * @tparam T The renderer type to find
     * @return Shared pointer to the renderer, or nullptr if not found
     */
    template<typename T>
    std::shared_ptr<T> get() const {
        for (const auto& renderer : renderers) {
            auto casted = std::dynamic_pointer_cast<T>(renderer);
            if (casted) {
                return casted;
            }
        }
        return nullptr;
    }

    /**
     * @brief Get number of registered ren< derers.
     */
    [[nodiscard]] size_t size() const { return renderers.size(); }

private:
    std::vector<std::shared_ptr<Renderer>> renderers;
};
