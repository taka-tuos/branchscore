cmake_minimum_required(VERSION 3.20)

project(branchscore VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BRANCHSCORE_BUILD_TESTS "Build branchscore tests" ON)
option(BRANCHSCORE_CUDA "Build the ggml CUDA backend" OFF)
option(BRANCHSCORE_VULKAN "Build the ggml Vulkan backend" OFF)
set(BRANCHSCORE_TEST_MODEL "" CACHE FILEPATH "Gemma 4 GGUF used by tokenizer tests")
set(BRANCHSCORE_TEST_MMPROJ "" CACHE FILEPATH "Gemma 4 mmproj GGUF used by model tests")
set(BRANCHSCORE_TEST_BACKEND "cpu" CACHE STRING "Backend used by model execution tests")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml submodule is missing. Run: git submodule update --init --recursive")
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${BRANCHSCORE_CUDA} CACHE BOOL "" FORCE)
set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${BRANCHSCORE_VULKAN} CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

add_library(branchscore_core
    src/backend_context.cpp
    src/categorical_readout.cpp
    src/gemma4_decision_engine.cpp
    src/gemma4_prompt_renderer.cpp
    src/image_preprocessor.cpp
    src/json.cpp
    src/model_loader.cpp
    src/prefill_engine.cpp
    src/state_cache.cpp
    src/tokenizer.cpp
    src/vision_encoder.cpp
)
add_library(branchscore::core ALIAS branchscore_core)

target_include_directories(branchscore_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(branchscore_core SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/examples)
target_link_libraries(branchscore_core PUBLIC ggml::ggml)
target_compile_features(branchscore_core PUBLIC cxx_std_17)

add_executable(branchscore tools/branchscore.cpp)
target_link_libraries(branchscore PRIVATE branchscore::core)

add_executable(branchscore-tokenize tools/tokenize.cpp)
target_link_libraries(branchscore-tokenize PRIVATE branchscore::core)

add_executable(branchscore-bench tools/branchscore_bench.cpp)
target_link_libraries(branchscore-bench PRIVATE branchscore::core)

if(BRANCHSCORE_BUILD_TESTS)
    enable_testing()
    add_executable(branchscore_backend_test tests/backend_context_test.cpp)
    target_link_libraries(branchscore_backend_test PRIVATE branchscore::core)
    add_test(NAME backend_context COMMAND branchscore_backend_test)

    add_executable(branchscore_gemma4_prompt_renderer_test tests/gemma4_prompt_renderer_test.cpp)
    target_link_libraries(branchscore_gemma4_prompt_renderer_test PRIVATE branchscore::core)
    add_test(NAME gemma4_prompt_renderer COMMAND branchscore_gemma4_prompt_renderer_test)

    add_executable(branchscore_decision_contract_test tests/decision_contract_test.cpp)
    target_link_libraries(branchscore_decision_contract_test PRIVATE branchscore::core)
    add_test(NAME decision_contract COMMAND branchscore_decision_contract_test)

    add_executable(branchscore_json_test tests/json_test.cpp)
    target_link_libraries(branchscore_json_test PRIVATE branchscore::core)
    add_test(NAME json COMMAND branchscore_json_test)

    add_executable(branchscore_gemma4_decision_engine_test tests/gemma4_decision_engine_test.cpp)
    target_link_libraries(branchscore_gemma4_decision_engine_test PRIVATE branchscore::core)
    add_test(
        NAME gemma4_decision_engine
        COMMAND branchscore_gemma4_decision_engine_test
            ${BRANCHSCORE_TEST_MODEL}
            ${BRANCHSCORE_TEST_MMPROJ}
            ${BRANCHSCORE_TEST_BACKEND})
    set_tests_properties(gemma4_decision_engine PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(branchscore_tokenizer_test tests/tokenizer_test.cpp)
    target_link_libraries(branchscore_tokenizer_test PRIVATE branchscore::core)
    add_test(NAME tokenizer COMMAND branchscore_tokenizer_test ${BRANCHSCORE_TEST_MODEL})
    set_tests_properties(tokenizer PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(branchscore_image_preprocessor_test tests/image_preprocessor_test.cpp)
    target_link_libraries(branchscore_image_preprocessor_test PRIVATE branchscore::core)
    add_test(NAME image_preprocessor COMMAND branchscore_image_preprocessor_test)

    add_executable(branchscore_state_cache_test tests/state_cache_test.cpp)
    target_link_libraries(branchscore_state_cache_test PRIVATE branchscore::core)
    add_test(NAME state_cache COMMAND branchscore_state_cache_test)

    add_executable(branchscore_prefill_test tests/prefill_engine_test.cpp)
    target_link_libraries(branchscore_prefill_test PRIVATE branchscore::core)
    add_test(
        NAME prefill
        COMMAND branchscore_prefill_test
            ${BRANCHSCORE_TEST_MODEL}
            ${BRANCHSCORE_TEST_MMPROJ}
            ${BRANCHSCORE_TEST_BACKEND})
    set_tests_properties(prefill PROPERTIES SKIP_RETURN_CODE 77)
endif()
