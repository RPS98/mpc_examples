# Google Test tests.
file(GLOB_RECURSE GTEST_SOURCE "*_gtest.cpp")

if(GTEST_SOURCE)
  find_package(GTest REQUIRED)

  foreach(TEST_FILE ${GTEST_SOURCE})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${PROJECT_NAME}_${TEST_NAME} ${TEST_FILE})

    # Link GTest::gtest_main only when the test does not define its own main().
    file(READ ${TEST_FILE} _TEST_CONTENT)
    if(_TEST_CONTENT MATCHES "int[ \t]+main[ \t]*\\(")
      target_link_libraries(${PROJECT_NAME}_${TEST_NAME}
        GTest::gtest
        ${PROJECT_NAME})
    else()
      target_link_libraries(${PROJECT_NAME}_${TEST_NAME}
        GTest::gtest
        GTest::gtest_main
        ${PROJECT_NAME})
    endif()

    add_test(
      NAME ${PROJECT_NAME}_${TEST_NAME}
      COMMAND ${PROJECT_NAME}_${TEST_NAME}
    )
  endforeach()
endif()
