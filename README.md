# tatami bindings for k-means

![Unit tests](https://github.com/libscran/kmeans_tatami/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/libscran/kmeans_tatami/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/libscran/kmeans_tatami/branch/master/graph/badge.svg?token=7S231XHC0Q)](https://codecov.io/gh/libscran/kmeans_tatami)

## Overview

This library implements a wrapper class to use [`tatami::Matrix`](https://github.com/tatami-inc/tatami) instances in the [**kmeans** library](https://github.com/libscran/kmeans).
The goal is to support k-means clustering from alternative matrix representations (e.g., sparse, file-backed) without requiring realization into a `kmeans::SimpleMatrix`.

## Quick start

Not much to say, really. 
Just replace the usual `kmeans::SimpleMatrix` with an instance of a `kmeans_tatami::Matrix`.

```cpp
#include "kmeans_tatami/kmeans_tatami.hpp"

// Initialize this with an instance of a concrete tatami subclass.
std::shared_ptr<tatami::Matrix<double, int> > tmat;

kmeans_tatami::Matrix<int, double, double, int> wrapper(std::move(tmat));
auto res = kmeans::compute(
    wrapper,
    kmeans::InitializeKmeanspp<int, double, int, double>(),
    kmeans::RefineHartiganWong<int, double, int, double>(),
    /* k = */ 10
);
```

See the [reference documentation](https://libscran.github.io/kmeans_tatami) for more details.

## Building projects 

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/libscran/kmeans_tatami
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(kmeans)
```

Then you can link to **kmeans** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe libscran::kmeans_tatami)

# For libaries
target_link_libraries(mylib INTERFACE libscran::kmeans_tatami)
```

By default, this will use `FetchContent` to fetch all external dependencies. 
Applications are advised to pin the versions of each dependency for stability - see [`extern/CMakeLists.txt`](extern/CMakeLists.txt) for suggested versions.
If you want to install them manually, use `-DKMEANS_TATAMI_FETCH_EXTERN=OFF`.

### CMake with `find_package()`

To install the library, clone an appropriate version of this repository and run:

```sh
mkdir build && cd build
cmake .. -DKMEANS_TATAMI_TESTS=OFF
cmake --build . --target install
```

Then we can use `find_package()` as usual:

```cmake
find_package(libscran_kmeans_tatami CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE libscran::kmeans_tatami)
```

Again, this will automatically acquire all its dependencies, see recommendations above.

### Manual

If you're not using CMake, the simple approach is to just copy the files in `include/` - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt). 
