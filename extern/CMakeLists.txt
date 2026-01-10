include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG master # ^4.0.4
)

FetchContent_Declare(
  sanisizer
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master # ^0.1.0
)

FetchContent_Declare(
  kmeans
  GIT_REPOSITORY https://github.com/libscran/kmeans
  GIT_TAG master # ^4.0.0
)

FetchContent_MakeAvailable(tatami)
FetchContent_MakeAvailable(sanisizer)
FetchContent_MakeAvailable(kmeans)
