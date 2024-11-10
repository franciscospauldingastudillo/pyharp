# Install utils submodule
#
#   library: libutils_debug.o
#   library: libutils_release.o
#
# Define the following symbols
#
#   utils_debug
#   utils_release

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel utils)
string(TOUPPER ${namel} nameu)

file(GLOB src_files *.cpp)

add_library(${namel}_${buildl}
    OBJECT
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )
