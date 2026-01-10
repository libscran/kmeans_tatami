include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

option(CODE_COVERAGE "Enable coverage testing for tests" OFF)
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

add_executable(
    libtest
    src/kmeans_tatami.cpp
)

target_link_libraries(
    libtest
    gtest_main
    kmeans_tatami
)

target_compile_options(libtest PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(DO_CODE_COVERAGE)
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

gtest_discover_tests(libtest)
