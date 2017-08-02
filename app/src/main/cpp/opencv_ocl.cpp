//
// Created by hl.he on 2017/7/25.
//

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

using namespace cv;

#define LOG_TAG "OCL"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define USING_OPENCL 1

extern "C"
JNIEXPORT void JNICALL Java_com_tom_opencladreon_MainActivity_testopencvocl (JNIEnv * env, jobject thisobject)
{
#if  USING_OPENCL == 1
    ocl::setUseOpenCL(true);
    ALOGD("useOpenCL %s " , ocl::useOpenCL()?"true":"false");

#endif

    // OpenCV 2.0 API
    // OpenCV-2.x 和 opencv-3.x使用区别 @ http://blog.csdn.net/tiemaxiaosu/article/details/52850820
    //ocl::oclMat oclIn(image), oclOut;
    //ocl::cvtColor(oclIn, oclOut, cv::COLOR_BGR2GRAY);
    //oclOut.download(grayImage);

#if  USING_OPENCL == 1
    UMat oclIn = imread("/mnt/sdcard/lena.jpg",CV_LOAD_IMAGE_COLOR).getUMat(cv::ACCESS_READ ) ;// ACCESS_READ, ACCESS_WRITE, ACCESS_RW, ACCESS_FAST
    UMat gray ;

    //int64_t before = cv::getTickCount() ;
    int64_t before = GetTickCount();
    cvtColor(oclIn, gray, cv::COLOR_BGR2GRAY);
#else
    Mat image=imread("/mnt/sdcard/lena.jpg",CV_LOAD_IMAGE_COLOR);
    if(image.data){
        ALOGD("load image success");
    }else{
        ALOGE("load image fail'");
        return ;
    }

    Mat gray ;
    int64_t before = GetTickCount();
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
#endif

    equalizeHist(gray, gray);
    GaussianBlur(gray, gray, Size(7,7), 1.5);
    Canny(gray, gray, 0, 50);


    int64_t after = GetTickCount();
    ALOGD("Cost => %" PRId64 " us ", (after - before) );

    imwrite("/mnt/sdcard/lena_gray.jpg",gray);
    ALOGD("imwrite done !");
}


extern "C"
JNIEXPORT void JNICALL Java_com_tom_opencladreon_MainActivity_openCvOclMatMul(JNIEnv * env, jobject thisobject)
{
    int loop = 500 ;
    Mat matA=imread("/mnt/sdcard/lena.jpg",CV_LOAD_IMAGE_COLOR);
    Mat matB=imread("/mnt/sdcard/lena.jpg",CV_LOAD_IMAGE_COLOR);
    ALOGD("matA dims %d type 0x%x channels 0x%x " ,  matA.dims  ,  matA.type() , matA.channels()  );
    // matA dims 2 type 0x10 channels 0x3
    assert(matA.data!=NULL);
    /*
     * 0-2位代表depth即数据类型（如CV_8U）
        #define CV_8U   0
        #define CV_8S   1
        #define CV_16U  2
        #define CV_16S  3
        #define CV_32S  4
        #define CV_32F  5
        #define CV_64F  6
     */
    Mat grayA ;
    Mat grayB ;
    cvtColor(matA, grayA, cv::COLOR_BGR2GRAY);
    cvtColor(matB, grayB, cv::COLOR_BGR2GRAY); // 这样转换还是整数 CV_8U
    ALOGD("grayA dims %d type 0x%x channels 0x%x " ,  grayA.dims  ,  grayA.type() , grayA.channels()  );
    // grayA dims 2 type 0x0 channels 0x1

    Mat floatA ;grayA.convertTo(floatA, CV_32FC1);
    Mat floatB ;grayB.convertTo(floatB, CV_32FC1);
    ALOGD("floatA dims %d type 0x%x channels 0x%x " ,  floatA.dims  ,  floatA.type() , floatA.channels()  );


    int64_t before = cv::getTickCount() ;
    loop = 10 ;
    while(loop-- ) {
        Mat matC = floatA * floatB;
    }
    // *是矩阵相乘 ,  mul方法才是计算两个Mat矩阵对应位的乘积
    // 数据类型（type）只能是 CV_32F、 CV_64FC1、 CV_32FC2、 CV_64FC2 这4种类型中的一种 @ cv::gemm @ Matmul.cpp
    // CV_Assert( type == B.type() && (type == CV_32FC1 || type == CV_64FC1 || type == CV_32FC2 || type == CV_64FC2) );
    int64_t after = cv::getTickCount() ;
    double  duration  =  (after - before)/cv::getTickFrequency();
    ALOGD("using Cpu Cost => %f s ", duration );


//    {
//        ocl::setUseOpenCL(true);
//        ALOGD("useOpenCL to Multiply Mat  ? %s " , ocl::useOpenCL()?"true":"false");
////    UMat umatA = floatA.getUMat( cv::ACCESS_READ);
////    UMat umatB = floatB.getUMat( cv::ACCESS_READ);
//
//        int64_t before_u = cv::getTickCount() ;
//        loop = 10 ;
//        while(loop--){
//            UMat umatC ; // 只要求返回值是 UMat  但是里面实现会对 Mat A Mat B  getUMat
//            cv::gemm( floatA, floatB, 1, cv::noArray() , 1, umatC , 0 );
//        }
//        int64_t after_u = cv::getTickCount() ;
//        double  duration_u  =  (after_u - before_u)/cv::getTickFrequency();
//        ALOGD("using OpenCl Cost => %f s ", duration_u );
//    }

    {
        int64_t before_u = cv::getTickCount() ;
        Mat simpleA (16, 16, CV_32F);
        for(int i = 0; i < simpleA.rows; i++)
            for(int j = 0; j < simpleA.cols; j++)
                simpleA.at<float>(i,j)= i * simpleA.rows + j ;

        Mat simpleB (16, 16, CV_32F);
        for(int i = 0; i < simpleB.rows; i++)
            for(int j = 0; j < simpleB.cols; j++)
                simpleB.at<float>(i,j)= i * simpleB.rows + j ;

        loop = 10 ;
        while(loop--){
            UMat umatC ; // 只要求返回值是 UMat  但是里面实现会对 Mat A Mat B  getUMat
            cv::gemm( simpleA, simpleB, 1, cv::noArray() , 1, umatC , 0 );
        }

        int64_t after_u = cv::getTickCount() ;
        double  duration_u  =  (after_u - before_u)/cv::getTickFrequency();
        ALOGD("Simple using OpenCl Cost => %f s ", duration_u );
    }


}
