//
// Created by rd0394 on 2017/8/2.
//

#ifndef OPENCLADREON_CL_COMMON_H
#define OPENCLADREON_CL_COMMON_H

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <cmath>
#include <cstdlib>

#include <jni.h>
#include <android/log.h>
using namespace std;


#define LOG_TAG "OCL"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

#endif //OPENCLADREON_CL_COMMON_H
