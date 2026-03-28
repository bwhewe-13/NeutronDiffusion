cmake_minimum_required(VERSION 3.21)
project(NeutronDiffusion VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(SKBUILD)
  set(_ndiffusion_build_python_default ON)
else()
  set(_ndiffusion_build_python_default OFF)
endif()

option(NDIFFUSION_BUILD_PYTHON "Build the Python extension module" ${_ndiffusion_build_python_default})

if(NDIFFUSION_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

  # pybind11 is injected via CMAKE_PREFIX_PATH by scikit-build-core at install
  # time, and can also be found from a manual install for standalone builds.
  # Fall back to querying the active Python interpreter if pybind11_DIR is unset.
  if(NOT DEFINED pybind11_DIR)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import pybind11; print(pybind11.get_cmake_dir())"
      RESULT_VARIABLE _pybind11_status
      OUTPUT_VARIABLE _pybind11_cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_pybind11_status EQUAL 0 AND _pybind11_cmake_dir)
      set(pybind11_DIR "${_pybind11_cmake_dir}")
    endif()
  endif()

  find_package(pybind11 REQUIRED)
endif()

add_subdirectory(cpp)
