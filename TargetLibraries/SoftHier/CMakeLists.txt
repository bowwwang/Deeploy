file(GLOB_RECURSE SOURCES 
    "src/**"
)

include(cmake/softhier-runtime.cmake)
add_deeploy_library(deeploysofthier STATIC ${SOURCES})

target_include_directories(deeploysofthier 
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/inc
)

target_compile_options(deeploysofthier PRIVATE
  # -DDEEPLOY_GENERIC_PLATFORM 
  -DDEEPLOY_SOFTHIER_PLATFORM 
  -Wno-implicit-function-declaration
  -Wno-implicit-int-conversion
  -Wno-sign-conversion
  -Wno-sign-compare
)

# bowwang: something is missing here
# target_include_directories(deeploysofthier PUBLIC ${SOFTHIER_INCLUDES})

target_link_libraries(deeploysofthier INTERFACE softhier-sdk)
target_sources(deeploysofthier INTERFACE $<TARGET_OBJECTS:softhier-sdk>)