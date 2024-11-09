include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME gtest)
set(REPO_URL "https://github.com/google/googletest")
set(REPO_TAG "v1.13.0")
set(INSTALL_GTEST OFF)

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
include_directories(${gtest_SOURCE_DIR}/include
                    ${gtest_SOURCE_DIR}/googletest/include)

target_compile_definitions(gtest PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(gtest_main PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
