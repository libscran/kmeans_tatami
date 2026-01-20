cmake_minimum_required(VERSION 3.24)

project(kmeans_tatami
    VERSION 0.1.1
    DESCRIPTION "tatami bindings for the k-means library"
    LANGUAGES CXX)

add_library(kmeans_tatami INTERFACE)
add_library(libscran::kmeans_tatami ALIAS kmeans_tatami)

include(GNUInstallDirs)
target_include_directories(kmeans_tatami INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libscran_kmeans_tatami>)

target_compile_features(kmeans_tatami INTERFACE cxx_std_17)

# Dependencies
option(KMEANS_TATAMI_FETCH_EXTERN "Automatically fetch kmeans_tatami's external dependencies." ON)
if(KMEANS_TATAMI_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(ltla_kmeans 4.0.0 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.1.0 CONFIG REQUIRED)
    find_package(tatami_tatami 4.0.4 CONFIG REQUIRED)
endif()

target_link_libraries(kmeans_tatami INTERFACE ltla::kmeans ltla::sanisizer tatami::tatami)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(KMEANS_TATAMI_TESTS "Build kmeans_tatami's test suite." ON)
else()
    option(KMEANS_TATAMI_TESTS "Build kmeans_tatami's test suite." OFF)
endif()
if(KMEANS_TATAMI_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libscran_kmeans_tatami)

install(TARGETS kmeans_tatami
    EXPORT kmeans_tatamiTargets)

install(EXPORT kmeans_tatamiTargets
    FILE libscran_kmeans_tatamiTargets.cmake
    NAMESPACE libscran::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_kmeans_tatami)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_kmeans_tatamiConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_kmeans_tatami)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_kmeans_tatamiConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libscran_kmeans_tatamiConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_kmeans_tatamiConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_kmeans_tatami)
