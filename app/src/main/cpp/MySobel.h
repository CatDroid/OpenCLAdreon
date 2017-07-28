//
// Created by rd0394 on 2017/7/26.
//

#ifndef OPENCLADREON_SOBEL_H
#define OPENCLADREON_SOBEL_H

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>

#include <CL/cl.h>

#include <jni.h>
#include <android/log.h>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//#include<opencv2/ocl/ocl.hpp>
#include<opencv2/core/ocl.hpp>
#include "CostHelper.h"


class MySobel {
public:
    uint32_t mImageWidth ;
    uint32_t mImageHeight ;
    uint32_t mImageStride ;
public:
    MySobel();
    bool doSobel(uint8_t* in, uint8_t* out, uint32_t width , uint32_t height, uint32_t stride );
    ~MySobel();

private:
    cl_platform_id mPlatform;
    cl_device_id   mDevice;

    cl_context      mContext ;
    cl_command_queue mCmdQueue;

    cl_mem mInImgMem;   // in image
    cl_mem mOutImgMem;  // out image
    cl_mem mInfoMem;    // info image : width height stride

    cl_program mProgram;
    cl_kernel mKernel;

private:
    void findPlatformAndDevices();
    void setupContextAndCmdQueue();
    void createProgramAndKernel();
    void createMemoryObject(uint8_t* input , uint8_t* output , uint32_t* info );
    void setKernelArg();
    void runKernel();
    void getKernelResult( uint8_t* output );
    void releaseMemoryObject();
    void releaseProgramAndKernel();
    void unsetupContextAndCmdQueue();

};


#endif //OPENCLADREON_SOBEL_H
