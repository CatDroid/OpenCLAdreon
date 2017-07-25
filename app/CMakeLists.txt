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



target_link_libraries( # Specifies the target library.
                       native-lib
                       ${log-lib} )