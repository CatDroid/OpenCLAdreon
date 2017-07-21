
/*
 * 高通平台 头文件来自 KhronosGroup https://github.com/KhronosGroup/OpenCL-Headers opencl20
 *          Note that Snapdragon 820 and other chipsets that have Adreno 5xx GPUs support OpenCL 2.0 Full profile.
 *
 *          https://developer.qualcomm.com/forum/qdn-forums/software/adreno-gpu-sdk/29883
 *        库文件 来自小米5
 *
 *
 * MTK平台 头文件来自 github https://developer.arm.com/technologies/compute-library --> https://github.com/ARM-software/ComputeLibrary
 *        库文件 来自MT6979
 */
#include <CL/cl.h>

#include <jni.h>
#include<malloc.h>
#include<stdio.h>
#include<stdlib.h>

#define LEN(arr) sizeof(arr) / sizeof(arr[0])
#define N 1024
#define NUM_THREAD 128

int num_block;

cl_uint num_device;
cl_uint num_platform;
cl_platform_id *platform;
cl_device_id *devices;
cl_int err;
cl_context context;
cl_command_queue cmdQueue;
cl_mem buffer,sum_buffer;
cl_program program ;
cl_kernel kernel;
const char* src[] = {
        "  __kernel void redution(  \n"
        "  __global int *data,     \n"
        "  __global int *output,   \n"
        "  __local int *data_local   \n"
        "  )  \n"
        " {   \n"
        "  int gid=get_group_id(0);   \n"
        "  int tid=get_global_id(0);    \n"
        "  int size=get_local_size(0);   \n"
        "  int id=get_local_id(0);     \n"
        "  data_local[id]=data[tid];   \n"
        "  barrier(CLK_LOCAL_MEM_FENCE);   \n"
        "  for(int i=size/2;i>0;i>>=1){    \n"
        "      if(id<i){   \n"
        "          data_local[id]+=data_local[id+i];   \n"
        "      }   \n"
        "      barrier(CLK_LOCAL_MEM_FENCE);   \n"
        "  }    \n"
        "  if(id==0){    \n"
        "      output[gid]=data_local[0];   \n"
        "  }    \n"
        " }   \n"

};



void Init_OpenCL()
{
    size_t nameLen1;
    char platformName[1024];

    err = clGetPlatformIDs(0, 0, &num_platform);
    platform=(cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platform);
    err = clGetPlatformIDs(num_platform, platform, NULL);

    err=clGetDeviceIDs(platform[0],CL_DEVICE_TYPE_GPU,0,NULL,&num_device);
    devices=(cl_device_id*)malloc(sizeof(cl_device_id)*num_device);
    err=clGetDeviceIDs(platform[0],CL_DEVICE_TYPE_GPU,num_device,devices,NULL);

}

void Context_cmd()
{
    context=clCreateContext(NULL,num_device,devices,NULL,NULL,&err);
    cmdQueue=clCreateCommandQueue(context,devices[0],0,&err);
}

void Create_Buffer(int *data)
{

    buffer=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(int)*N,data,&err);
    sum_buffer=clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(int)*num_block,0,&err);
}

void Create_program()
{
    program=clCreateProgramWithSource(context, LEN(src), src, NULL, NULL);
    err=clBuildProgram(program,num_device,devices,NULL,NULL,NULL);
    kernel = clCreateKernel(program, "redution", NULL);
}

void Set_arg()
{
    err=clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
    err=clSetKernelArg(kernel,1,sizeof(cl_mem),&sum_buffer);
    err=clSetKernelArg(kernel,2,sizeof(int)*NUM_THREAD,NULL);
}

void Execution()
{
    const size_t globalWorkSize[1]={N};
    const size_t localWorkSize[1]={NUM_THREAD};
    err=clEnqueueNDRangeKernel(cmdQueue,kernel,1,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
    clFinish(cmdQueue);
}

void CopyOutResult(int*out)
{
    err=clEnqueueReadBuffer(cmdQueue,sum_buffer,CL_TRUE,0,sizeof(int)*num_block,out,0,NULL,NULL);
}





int  test()
{
    int* in,*out;
    num_block=N/NUM_THREAD;
    in=(int*)malloc(sizeof(int)*N);
    out=(int*)malloc(sizeof(int)*num_block);
    for(int i=0;i<N;i++){
        in[i]=1;
    }
    Init_OpenCL();
    Context_cmd();
    Create_Buffer(in);
    Create_program();
    Set_arg();
    Execution();
    CopyOutResult(out);
    int sum=0;
    for(int i=0;i<num_block;i++){
        sum+=out[i];
    }
    return sum;
}

extern "C"
JNIEXPORT jstring JNICALL Java_com_tom_opencladreon_MainActivity_testopencl (JNIEnv * env, jobject thisobject)
{
    char result[10];
    sprintf(result,"%d\n",test());
    return env->NewStringUTF(result);
}

extern "C"
JNIEXPORT jstring JNICALL Java_com_tom_opencladreon_MainActivity_getPlatformName(JNIEnv *env , jobject thisobject)
{
    char buffer[1024];
    clGetPlatformInfo(platform[0],CL_PLATFORM_NAME,sizeof(buffer),buffer,NULL);
    return env->NewStringUTF(buffer);
}

extern "C"
JNIEXPORT jstring JNICALL Java_com_tom_opencladreon_MainActivity_getDeviceName(JNIEnv *env , jobject thisobject)
{

    char buffer[1024];
    clGetDeviceInfo(devices[0],CL_DEVICE_NAME,sizeof(buffer),buffer,NULL);
    return env->NewStringUTF(buffer);
}
