import os
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


class PlantsimConan(ConanFile):
    name = "plantsim"
    version = "0.1.0"
    settings = "os", "compiler", "build_type", "arch"

    options = {
        "with_cpu": [True, False],
        "with_cuda": [True, False],
        "with_sycl": [True, False],
        "with_tests": [True, False],
        "with_visualization": [True, False],
        "with_benchmarks": [True, False],
        "with_experiments": [True, False],
        "enable_profiling": [True, False],
    }
    default_options = {
        "with_cpu": True,
        "with_cuda": False,
        "with_sycl": False,
        "with_tests": False,
        "with_visualization": True,
        "with_benchmarks": False,
        "with_experiments": False,
        "enable_profiling": False,
        "glad/*:gl_version": "4.6",
        "glad/*:gl_profile": "compatibility",
        "glad/*:spec": "gl",
        "glad/*:extensions": "",
        "glad/*:no_loader": False,
    }

    def requirements(self):
        if self.options.with_cpu:
            self.requires("eigen/3.4.0")
        if self.options.with_visualization:
            self.requires("glfw/3.4")
            self.requires("glm/1.0.1")
            self.requires("imgui/1.91.5")
            self.requires("freetype/2.13.2")
            self.requires("glad/0.1.36")
        if self.options.with_tests:
            self.requires("gtest/1.14.0")

    def validate(self):
        if not (self.options.with_cpu or self.options.with_cuda or self.options.with_sycl):
            raise Exception("At least one of with_cpu, with_cuda, with_sycl must be enabled.")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        if self.options.with_visualization:
            self._copy_imgui_backend_sources()

        tc = CMakeToolchain(self)

        backends = []
        if self.options.with_cpu:
            backends.append("CPU")
        if self.options.with_cuda:
            backends.append("CUDA")
        if self.options.with_sycl:
            backends.append("SYCL")
        tc.cache_variables["BACKEND_LIST"] = ";".join(backends)

        tc.cache_variables["BUILD_VISUALIZATION"] = bool(self.options.with_visualization)
        tc.cache_variables["BUILD_TEST"] = bool(self.options.with_tests)
        tc.cache_variables["BUILD_BENCH"] = bool(self.options.with_benchmarks)
        tc.cache_variables["BUILD_EXPERIMENTS"] = bool(self.options.with_experiments)
        tc.cache_variables["ENABLE_PROFILING"] = bool(self.options.enable_profiling)

        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def _copy_imgui_backend_sources(self):
        src_dir = os.path.join(self.dependencies["imgui"].package_folder, "res", "bindings")
        dst_dir = os.path.join(self.source_folder, "src", "visualisation", "imgui_backends")

        copy(self, "*glfw*", src_dir, dst_dir)
        copy(self, "*opengl3*", src_dir, dst_dir)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
