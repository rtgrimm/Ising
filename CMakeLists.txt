cmake_minimum_required(VERSION 3.21)
project(IsingCpp)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_SWIG_FLAGS "-py3")

find_package(SWIG REQUIRED)
include(UseSWIG)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)


set_source_files_properties(Ising.i PROPERTIES
        CPLUSPLUS ON
        INCLUDE_DIRECTORIES ./
        )

swig_add_library(Ising
        TYPE SHARED
        LANGUAGE python
        SOURCES	Ising.i)




include_directories(${SWIG_MODULE_Ising_REAL_NAME}  "./")
target_include_directories(${SWIG_MODULE_Ising_REAL_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS} "./")
target_link_libraries(${SWIG_MODULE_Ising_REAL_NAME} PRIVATE ${PYTHON_LIBRARIES})

add_executable(IsingCpp Native/main.cpp Native/lattice.hpp)
