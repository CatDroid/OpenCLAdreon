//
// Created by hl.he on 2017/8/2.
//

#include "jni_cl_common.h"


int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));
    if (f.is_open()) {
        size_t fileSize;
        f.seekg(0, std::fstream::end); size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str) {
            f.close();
            return NULL;
        }
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    ALOGE("Error: Failed to open file %s\n", filename);
    return 1;
}


extern "C"
bool Java_com_tom_opencladreon_RunKernelActivity_nativeRunSimpeGEMM(JNIEnv* env, jclass clazz)
{
    cl_int status;
    cl_platform_id platform;

    //创建平台对象
    status = clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    //创建 GPU 设备
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device,  NULL);
    //创建context
    cl_context context = clCreateContext(NULL, 1,  &device,  NULL, NULL, &status );
    if(status != CL_SUCCESS) {
        ALOGE("Failed to Create Context for device 0.");
        return false ;
    }
    //创建命令队列
    cl_command_queue commandQueue = clCreateCommandQueue(context, device,  CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        ALOGE("Failed to Create CommandQueue for device 0.");
        return false ;
    }

    //建立要传入从机的数据 - 创建内核和内存对象

    const int Ndim = 35; // NxP * PxM
    const int Mdim = 35;
    const int Pdim = 35;
    int szA = Ndim * Pdim;
    int szB = Pdim * Mdim;
    int szC = Ndim * Mdim;

    float *A;
    float *B;
    float *C;

    A = (float *)malloc(szA * sizeof(float));
    B = (float *)malloc(szB * sizeof(float));
    C = (float *)malloc(szC * sizeof(float));
    int i, j;
    for (i = 0; i < szA; i++)
        A[i] = (float)((float)i + 1.0);
    for (i = 0; i < szB; i++)
        B[i] = (float)((float)i + 1.0);


    cl_mem memObjects[3] = { 0, 0, 0 };
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
                                   sizeof(float)* szA, A, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
                                   sizeof(float)* szB, B, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float)* szC, C, NULL);
    if (memObjects[0] == NULL || memObjects[1] == NULL ||memObjects[2] == NULL)
        ALOGE("Error in clCreateBuffer.\n");


    std::string sourceStr;
    status = convertToString("/data/data/com.tom.opencladreon/app_opencl_dir/simplegemm.cl", sourceStr);
    if (status){
        ALOGE("read file ERROR ");
        return false ;
    }

    const char * source = sourceStr.c_str();
    size_t source_len = strlen(source) ;
    //创建程序对象
    cl_program program = clCreateProgramWithSource( context,  1,  &source,  &source_len,   NULL);
    //编译程序对象
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if( status != CL_SUCCESS ){
        ALOGE("clBuildProgram  ERROR ");
        char tbuf[0x10000];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf,  NULL);
        ALOGE("clGetProgramBuildInfo %s ", tbuf );
        return false ;
    }

    //创建 Kernel 对象
    cl_kernel kernel = clCreateKernel(program, "simplegemm", NULL);

    //设置 Kernel 参数
    status = CL_SUCCESS ;
    status |= clSetKernelArg(kernel, 0, sizeof(int), &Ndim);
    status |= clSetKernelArg(kernel, 1, sizeof(int), &Mdim);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &Pdim);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[0]);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[1]);
    status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[2]);
    if (status != CL_SUCCESS ){
        ALOGE("参数设置错误");
        return false ;
    }

    //执行 kernel
    size_t global[2];
    cl_event prof_event;
    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time = (cl_ulong)0;
    double rum_time;
    global[0] = (size_t)Ndim; // NxP * PxM
    global[1] = (size_t)Mdim;
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,  global, NULL, 0, NULL, &prof_event);
    if (status != CL_SUCCESS ){
        ALOGE("执行内核时错误");
        return false ;
    }
    clFinish(commandQueue);

    //读取时间
    status = CL_SUCCESS ;
    status |= clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_QUEUED,  sizeof(cl_ulong),&ev_start_time,NULL);
    status |= clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_END,  sizeof(cl_ulong),&ev_end_time,NULL);

    if (status != CL_SUCCESS ){
        ALOGE("读取时间的时候发生错误\n");
        return false ;
    }
    rum_time = (double)(ev_end_time - ev_start_time);
    ALOGI("执行时间为:%f ms ", rum_time / 1000000.0   );

    //数据拷回 host 内存
    status = clEnqueueReadBuffer(commandQueue, memObjects[2],CL_TRUE, 0,  sizeof(float)* szC, C,0, NULL, NULL);
    if (status != CL_SUCCESS ) {
        ALOGE("读回数据的时候发生错误\n");
        return false ;
    }

    //结果显示

    FILE* fp = fopen("/mnt/sdcard/simplegemm_a.txt","w+");
    for (i = 0; i < Ndim; i++) {
        for (j = 0; j < Pdim; j++){
            fprintf(fp,"%.3f,", A[i*Pdim + j]);
        }
        fseek(fp,-1,SEEK_CUR);
        fprintf(fp,";\n");
    }
    fprintf(fp,"\n");
    fclose(fp);

    fp = fopen("/mnt/sdcard/simplegemm_b.txt","w+");
    for (i = 0; i < Pdim; i++) {
        for (j = 0; j < Mdim; j++){
            fprintf(fp,"%.3f,", B[i*Mdim + j]);
        }
        fseek(fp,-1,SEEK_CUR);
        fprintf(fp,";\n");
    }
    fprintf(fp,"\n");
    fclose(fp);

    fp = fopen("/mnt/sdcard/simplegemm_c.txt","w+");
    for (i = 0; i < Ndim; i++) {
        for (j = 0; j < Mdim; j++){
            fprintf(fp,"%.3f,", C[i*Mdim + j]);
        }
        fseek(fp,-1,SEEK_CUR);
        fprintf(fp,";\n");
    }
    fprintf(fp,"\n");
    fclose (fp);


    fp = fopen("/mnt/sdcard/simplegemm_cpu.txt","w+");
    for (i = 0; i < Ndim; i++) {
        for (j = 0; j < Mdim; j++){// NxP * PxM
            float sum = 0 ;
            for(int k = 0 ; k < Pdim ; k++ ){
                sum += A[i*Ndim + k ] * B[ k*Mdim + j ] ;
            }
            fprintf(fp,"%.3f,",sum );
        }
        fseek(fp,-1,SEEK_CUR);
        fprintf(fp,";\n");
    }
    fprintf(fp,"\n");
    fclose (fp);


    if (A) free(A);
    if (B) free(B);
    if (C) free(C);

    //删除 OpenCL 资源对象
    clReleaseMemObject(memObjects[2]);
    clReleaseMemObject(memObjects[1]);
    clReleaseMemObject(memObjects[0]);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    return true ;
}

/*
    % hl.he 2017.08.03  Matlab 对比 Kernel 矩阵相乘

    clear all;
    clc;

    a=single( load('C:\Users\rd0394\Desktop\simplegemm_a.txt') ) ;
    b=single( load('C:\Users\rd0394\Desktop\simplegemm_b.txt') );
    % a= load('C:\Users\rd0394\Desktop\simplegemm_a.txt') ;
    % b= load('C:\Users\rd0394\Desktop\simplegemm_b.txt') ;
    % 注意精度问题

    kernel_cal= load('C:\Users\rd0394\Desktop\simplegemm_c.txt')   ;
    cpu_cal=load('C:\Users\rd0394\Desktop\simplegemm_cpu.txt');

    matlab_cal = a * b ;

    cpudiff = cpu_cal - kernel_cal;

    matdiff = matlab_cal - kernel_cal ;


 * */