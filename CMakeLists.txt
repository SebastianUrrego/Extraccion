cmake_minimum_required(VERSION 2.8)

project(Taller1Vision)

include_directories("/usr/local/include")
find_package( OpenCV REQUIRED )

add_executable(${PROJECT_NAME} "feature_matching.cpp")
target_link_libraries( ${PROJECT_NAME}  ${OpenCV_LIBS} )
