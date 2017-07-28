# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

cmake_minimum_required(VERSION 3.4.1)

#add_definitions(-Werror)
#add_definitions(-Wall)

if( OPENCL_VENDOR MATCHES Dragon )          #高通 晓龙820 小米5

    include_directories(Adreon/include)
    link_directories( Adreon/libs/${ANDROID_ABI}/ )
    set(CMAKE_C_FLAGS "-DUSING_DRAGON=1  ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-DUSING_DRAGON=1  ${CMAKE_CXX_FLAGS}")

elseif( OPENCL_VENDOR MATCHES MTK )         #MTK MT6797

    include_directories(MTK/include)
    link_directories( MTK/libs/${ANDROID_ABI}/ )
    set(CMAKE_C_FLAGS "-DUSING_MTK=1  ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-DUSING_MTK=1  ${CMAKE_CXX_FLAGS}")
else()                                      #默认 高通平台

    include_directories(Adreon/include)
    link_directories( Adreon/libs/${ANDROID_ABI}/ )
    set(CMAKE_C_FLAGS "-DUSING_DRAGON=1  ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-DUSING_DRAGON=1  ${CMAKE_CXX_FLAGS}")
endif()



add_library(native-lib
             SHARED
             src/main/cpp/native-lib.cpp
             src/main/cpp/ocl.cpp
             src/main/cpp/opencv_ocl.cpp
             src/main/cpp/jni_sobel.cpp
             src/main/cpp/MySobel.cpp
             )

find_library(
              log-lib
              log )

if( OPENCL_VENDOR MATCHES Dragon )
    set(log-lib ${log-lib} libOpenCL.so )
elseif( OPENCL_VENDOR MATCHES MTK )
    set(log-lib ${log-lib} libGLES_mali.so )
else()
    set(log-lib ${log-lib} libOpenCL.so )
endif()

# find_package will search <package>_DIR to find <package>Config.cmake
# <name>Config.cmake or <lower-case-name>-config.cmak
# move to build.gradle
# set(OpenCV_DIR E:\\temp\\opencv-3.2.0-rc\\build\\OpenCV-android-sdk\\sdk\\native\\jni )
# set OpenCV_DIR variable to the absolute path to the directory containing OpenCVConfig.cmake file
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})


target_link_libraries( # Specifies the target library.
                       native-lib
                       ${log-lib}
                       ${OpenCV_LIBS} )