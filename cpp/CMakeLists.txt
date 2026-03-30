# ============================================================================
# Core static library — no Python dependency
# ============================================================================
add_library(ndiffusion_core STATIC
    src/solver_1d.cpp
    src/solver_2d_structured.cpp
    src/solver_2d_unstructured.cpp
)

target_include_directories(ndiffusion_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(ndiffusion_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2 -Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W4>
)

# Required so the static library can be linked into the shared _core.so extension
set_target_properties(ndiffusion_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Standalone driver executable
# ============================================================================
add_executable(ndiffusion_driver src/main.cpp)
target_link_libraries(ndiffusion_driver PRIVATE ndiffusion_core)

# ============================================================================
# Python extension (_core)
# ============================================================================
if(NDIFFUSION_BUILD_PYTHON)
    pybind11_add_module(_core python/bindings.cpp)
    target_link_libraries(_core PRIVATE ndiffusion_core)
    target_include_directories(_core PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

    # Install the extension into the ndiffusion package directory inside the wheel.
    # scikit-build-core sets CMAKE_INSTALL_PREFIX to the platlib staging dir.
    install(TARGETS _core LIBRARY DESTINATION ndiffusion)
endif()

