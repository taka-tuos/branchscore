cmake_minimum_required(VERSION 3.20)

# The generated parser is checked in so the normal build has no Node.js,
# npm, pkg-config, or system llhttp dependency.
add_library(branchscore_llhttp STATIC
    generated/src/llhttp.c
    generated/include/llhttp.h
    upstream/src/native/api.c
    upstream/src/native/api.h
    upstream/src/native/http.c
)
add_library(branchscore::llhttp ALIAS branchscore_llhttp)

target_include_directories(branchscore_llhttp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generated/include>
)
target_compile_features(branchscore_llhttp PUBLIC c_std_99)

# Regeneration is deliberately opt-in. It needs npm and the pinned generator
# dependencies from upstream/package-lock.json, but never affects consumers
# that only build the checked-in generated C source.
add_custom_target(branchscore-llhttp-regenerate
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/regenerate.sh"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Regenerating vendored llhttp 9.3.1 sources"
)
