//
// Created by hl.he on 2017/7/26.
//

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <fcntl.h>

#include <CL/cl.h>

#include <jni.h>
#include <android/log.h>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//#include<opencv2/ocl/ocl.hpp>
#include<opencv2/core/ocl.hpp>
#include "CostHelper.h"
#include "MySobel.h"

using namespace cv;

#define LOG_TAG "OCL"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define N 20	//这里的N是sobel滤波的阈值

void CPU_Sobel(cv::Mat& in , cv::Mat&  out ) {
    uchar a00, a01, a02;
    uchar a10, a11, a12;
    uchar a20, a21, a22;

//    int m[3][3] = {{-1 , 0 , 1}, {-2,  0 , 2} , {-1 , 0  ,1} };
//    Mat Gx = Mat(3,3, CV_32S, m);
//    Mat Gy = Gx.inv();


    //cv::Mat submat =  in(cv::Rect(1,1,3,3)); 左上角坐标 和 长 宽
    //ALOGD("submat %d %d %d", submat.dims , submat.cols , submat.rows );

    for (int i = 1; i < in.rows - 1; i++) {
        for (int j = 1; j < in.cols- 1; j++) {
            a00 = in.at<uchar>( i-1 ,  j-1  );
            a01 = in.at<uchar>( i-1 ,  j );
            a02 = in.at<uchar>( i-1 ,  j+1 );

            a10 = in.at<uchar>( i ,  j-1  );
            a11 = in.at<uchar>( i ,  j );
            a12 = in.at<uchar>( i ,  j+1 );

            a20 = in.at<uchar>( i+1 ,  j-1  );
            a21 = in.at<uchar>( i+1 ,  j );
            a22 = in.at<uchar>( i+1 ,  j+1 );

            //cv::Mat submat =  in(cv::Rect( i-1, j-1, 3, 3));
            /*
             *  Gx
             *  -1  0  1
             *  -2  0  2
             *  -1  0  1
             *
             *  Gy
             *  -1  -2 -1
             *   0   0  0
             *   1   2  1
             *
             *   梯度大小 G = ( Gx^2 + Gy^2 )^0.5
             */
            // x方向上的近似导数
            float ux = a20 * (1) + a21 * (2) + a22 * (1)
                       + a00 * (-1)  + a01 * (-2) + a02 * (-1);
            // y方向上的近似导数
            float uy = a02 * (1) + a12 * (2) + a22 * (1)
                       + a00 * (-1)  + a10 * (-2) + a20 * (-1);
            //梯度
            float u = sqrt(ux * ux + uy * uy);

            //阈值法确定边缘
            if (u > 255) {
                u = 255;
            } else if (u < N) {
                u = 0;
            }
            out.at<uchar>(i,j) = (uchar)u ;
        }
    }
}




MySobel* g_pSoble = NULL ;

extern "C"
JNIEXPORT void JNICALL Java_com_tom_opencladreon_MainActivity_nativeoclsobel(JNIEnv * env, jobject thisobject)
{
    Mat image=imread("/mnt/sdcard/lena.jpg",CV_LOAD_IMAGE_COLOR);
    if(image.data){
        ALOGD("load image success");
    }else{
        ALOGE("load image fail'");
        assert(image.data!=NULL);
        return ;
    }
    Mat gray ; cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    ALOGD("GPU_Sobel w %d h %d step %zd ", gray.cols , gray.rows , gray.step[0]);

//    {
//        int fd = open("/mnt/sdcard/lena_.gray",O_CREAT|O_WRONLY,0755);
//        write(fd , gray.data , gray.cols * gray.rows );
//        close(fd);
//    }

    cv::Mat out = cv::Mat(gray.rows, gray.cols, CV_8UC1);
    CostHelper c ;
    if( g_pSoble == NULL ){
        g_pSoble = new MySobel();
    }
    g_pSoble->doSobel( gray.data , out.data , gray.cols , gray.rows ,gray.step[0] );
    ALOGD("GPU_Sobel cost %" PRId64 " us", c.Get() );

    imwrite("/mnt/sdcard/lena_gpu_sobel.jpg",out);
}

extern "C"
JNIEXPORT void JNICALL Java_com_tom_opencladreon_MainActivity_nativecpusobel(JNIEnv * env, jobject thisobject)
{
    Mat image=imread("/mnt/sdcard/lena.jpg",CV_LOAD_IMAGE_COLOR);
    if(image.data){
        ALOGD("load image success");
    }else{
        ALOGE("load image fail'");
        assert(image.data!=NULL);
        return ;
    }
    Mat gray ; cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat out = cv::Mat(gray.rows, gray.cols, CV_8UC1);

    CostHelper c ;
    CPU_Sobel(gray, out);
    ALOGD("CPU_Sobel cost %" PRId64 " us", c.Get() );

    imwrite("/mnt/sdcard/lena_cpu_sobel.jpg",out);
}


