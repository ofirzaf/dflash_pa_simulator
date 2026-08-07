cmake_minimum_required(VERSION 3.20)

project(dflash_pa_simulator LANGUAGES CXX)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_executable(dflash_pa_simulator src/main.cpp)
target_link_libraries(dflash_pa_simulator PRIVATE openvino::runtime)
target_compile_features(dflash_pa_simulator PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dflash_pa_simulator PRIVATE -Wall -Wextra -Wpedantic)
endif()
