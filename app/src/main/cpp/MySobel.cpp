//
// Created by hl.he  on 2017/7/26.
//

#include <fcntl.h>
#include "MySobel.h"

#define LOG_TAG "MySobel"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define TEST_KERNEL_COST_TIME 1

#define N 20	//这里的N是sobel滤波的阈值

// clBuildProgram fail with -11 info <source>:26:3:
// error: use of type 'double' requires cl_khr_fp64 extension to be enabled
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"

//     "#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode : enable\n"\
//    "#pragma OPENCL SELECT_ROUNDING_MODE rtn\n"\

/*
 *     "#ifdef cl_khr_fp64\n"\
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"\
        "#endif\n"\
 */

/*
 *  标量类型  Kernel 与 API
 *  Type in OpenCL Language         API type for application
 *  char                            cl_char
 *  short                           cl_short
 *  https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html
 *
 */
// 往下 x正方向
// 往右 y正方向
#define  KERNEL_SRC "\n"\
    "#ifndef SOBEL_VALUE \n"\
    "#define SOBEL_VALUE 20 \n"\
    "#endif \n"\
	"__kernel void Sobel(__global uchar *array1, __global uchar *array2, __global int *array3)\n "\
	"{\n "\
	"   size_t gidx = get_global_id(0);\n"\
	"   size_t gidy = get_global_id(1);\n"\
	"   uchar a00, a01, a02;\n"\
	"   uchar a10, a11, a12;\n"\
	"   uchar a20, a21, a22;\n"\
	"   int width=array3[0];\n"\
	"   int heigh=array3[1];\n"\
	"   int widthStep=array3[2];\n"\
	"	if(gidy>0&&gidy<heigh-1&&gidx>0&&gidx<width-1)\n"\
	"   {\n"\
	"       a00=  array1[gidx-1   +  widthStep * (gidy-1)] ;\n"\
	"       a01=  array1[gidx     +  widthStep * (gidy-1)] ;\n"\
	"       a02=  array1[gidx+1   +  widthStep * (gidy-1)] ;\n "\
	"       a10=  array1[gidx-1   +  widthStep * gidy ]    ;\n"\
	"       a11=  array1[gidx     +  widthStep * gidy ]    ;\n"\
	"       a12=  array1[gidx+1   + widthStep * gidy]      ;\n"\
	"       a20=  array1[gidx-1   + widthStep * (gidy+1)]  ;\n"\
	"		a21=  array1[gidx     + widthStep * (gidy+1)]  ;\n"\
	"		a22=  array1[gidx+1   + widthStep * (gidy+1)]  ;\n"\
	"		float ux=  a20 + 2*a21 +a22 -a00 -2*a01 -a02 ;\n"\
	"		float uy=  a02 + 2*a12 +a22 -a00 -2*a10 -a20;\n"\
	"		float u=sqrt( ux * ux + uy * uy );\n"\
    "		if( u > 255) {\n"\
	"           u = 255 ;\n"\
	"		} else if ( u < SOBEL_VALUE ) {\n"\
	"           u = 0;\n"\
	"		}\n"\
	"		array2[gidx+widthStep*gidy] =  u ;\n"\
	"	}\n"\
"}"
// a11 是无用的

/*
 * 对于MTK Mali 平台 char array2[gidx+widthStep*gidy] = convert_char(u)  还是
 * char array2[gidx+widthStep*gidy] =  u  只要是从float -> char 都会出现 > 127 的数被压制到 127 0x0111 1111
 *      MTK平台 当浮点数是 float  160 209 176 164 169 141 130 超过127的  => char 都是 127
 *
 *      但是如果是 float -> uchar     float 160.0f -> uchar 160
 *
 * 对于高通平台
 *      float  160/209/130  -->  char 160/209/130
 *
 * MTK和高通都对 浮点数float -1  =>  char 255
 */


void MySobel::findPlatformAndDevices()
{
    cl_uint numPlatforms = 0 ;
    cl_platform_id* platforms = NULL ;
    cl_int status = CL_SUCCESS ;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms > 0) {
        platforms = (cl_platform_id*) malloc(
                numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        assert(status == CL_SUCCESS);
        mPlatform = platforms[0];
        free(platforms);
    }else{
        ALOGE("Platform is NOT found");
        assert(numPlatforms!=0);
    }

    cl_uint numDevices = 0;
    cl_device_id *devices;
    status = clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if( numDevices > 0 ){
        devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices,  NULL);
        assert(status == CL_SUCCESS);
        mDevice = devices[0];
        free(devices);
    } else{
        ALOGE("Devices is NOT found");
        assert(numDevices!=0);
    }

}


/*
 * cl_context clCreateContext(
 *          cl_context_properties *properties,// 属性列表  最后一个是0
 *          cl_uint num_devices,            //  设备数量
 *          const cl_device_id *devices,    //  创建上下文的设备 可以支持多个设备(同一个平台上)
 *          pfn_notify,  void*user_data,    //  回调函数以及数据
 *          cl_int *errcode_ret)
 *
 *
 * 属性表支持
 * 属性                               属性值
 * CL_CONTEXT_PLATFORM                  cl_platform_id  选择一个平台,如果没有的话,选择默认(implementation-defined 由具体实现定义)
 * CL_PRINTF_CALLBACK_ARM               回调函数指针(kernel中打印)
 * CL_PRINTF_BUFFERSIZE_ARM             0x100000(ARM打印buffer大小)
 *
 *  cl_command_queue clCreateCommandQueue(	cl_context context,
 *                                          cl_device_id device, // 创建Context时关联的一个设备(一个命令队列对应一个设备)
 *                                          cl_command_queue_properties properties, // cl_command_queue_properties 是  Enumerated Type
 *                                          cl_int *errcode_ret) // 可以动态调用clCreateCommandQueue来切换顺序和乱序执行 ?
 * 属性表支持
 * 属性                                       属性值
 * CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE       命令队列是按顺in-order还是乱序out-of-order
 * CL_QUEUE_PROFILING_ENABLE                    命令队列中加入配置命令??
 *
 *
 *  note:   1. 上下文Context 对应 一个平台的多个设备     命令队列是针对一个Context的一个设备
 *          2. clCreateContextFromType  可以在某一个平台上 所有同类设备上 创建上下文 CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU  不用指定具体设备
 *          3. 如果是顺序执行  clEnqueueNDRangeKernel 执行Kernel A 然后clEnqueueNDRangeKernel指向Kernel B
 *                           应用可以认为A先执行完再执行B 如果B的输入cl_mem是A的输出cl_mem 那么B可以见到A输出的正确的数据
 *             如果是乱序执行   为了保证顺序 对于执行B 可以在调用clEnqueueNDRangeKernel 使用event_wait_list
 *                          事件的等待(a wait)或者栅栏(a barrier)命令 可以加入命令队列
 *                          ‘The wait for events command’  之前 队列中‘identified by the list of events to wait for’的命令 执行完毕
 *                          ‘The barrier command’   之前队列中的‘所有’命令执行完毕 all in a command-queue
 *
 *             在乱序执行下，当read,write,copy or map memory objects等命令在clEnqueueNDRangeKernel, clEnqueueTask or clEnqueueNativeKernel
 *             之后加入队列 并不能保证kernel先被调度执行 所以,
 *             上面clEnqueueXXX命令返回的event object可以用来往队列中增加一个等待命令(a wait for event)
 *             或者 往队列增加一个栅栏对象(a barrier command)
 *             等待完成，然后再读写内存对象
 */
void MySobel::setupContextAndCmdQueue(){

    cl_int status = CL_SUCCESS ;

//#define  CL_PRINTF_CALLBACK_ARM    0x40B0
//#define  CL_PRINTF_BUFFERSIZE_ARM  0x40B1
//    cl_context_properties properties[] =  {
//            /* Enable a printf callback function for this context. */
//            CL_PRINTF_CALLBACK_ARM,   (cl_context_properties) printf_callback,
//            /* Request a minimum printf buffer size of 4MiB for devices in the
//               context that support this extension. */
//            CL_PRINTF_BUFFERSIZE_ARM, (cl_context_properties) 0x100000 ,
//            CL_CONTEXT_PLATFORM,      (cl_context_properties) mPlatform,
//            0
//    };
    mContext = clCreateContext(NULL/*默认平台*/ ,1 /*只在一个设备上创建上下文*/, &mDevice, NULL, NULL, &status);
    if(status != CL_SUCCESS) {
        ALOGE("clCreateContext fail");
        assert(status == CL_SUCCESS);
    }

    cl_command_queue_properties prop = 0 ;
#if TEST_KERNEL_COST_TIME == 1
    prop = CL_QUEUE_PROFILING_ENABLE ;
#endif
    mCmdQueue = clCreateCommandQueue(mContext, mDevice, prop , &status);
    if(status != CL_SUCCESS) {
        ALOGE("clCreateCommandQueue fail");
        assert(status == CL_SUCCESS);
    }

}


/*
   * 两个创建Program对象API: clCreateProgramWithBinary  clCreateProgramWithSource
   *
   * cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)
   *                  根据字符串/源码 创建Program
   *
   * count        in  指示 字符串数组的长度
   * strings      in  字符串数组 (可以多个kernel字符串,每个kernel字符串作为数组的一个成员)
   * lengths      in  字符串数组每个字符串的长度  如果是NULL 那么会认为每个字符串以NULL作为结束
   * errcode_ret out  返回结果  可能返回错误 CL_BUILD_PROGRAM_FAILURE  -11
   *
   *
   *  cl_int clBuildProgram( cl_program program,
   *                            cl_uint num_devices,
   *                            const cl_device_id *device_list,   // 只生成到指定的设备上,否则跟program相关的上下文所有设备
   *                            const char *options,               // 构建选项
   *                            void (*pfn_notify)(cl_program, void *user_data), // 如果指定了回调函数,build不会阻塞,否则直到构建完成
   *                            void *user_data)
   *
   * 构建选项
   *    预处理选项:
   *        -D name             设置宏name为=1          -D MALI_GPU
   *        -D name=definition  设置宏name为definition  -D MAX_VALUE=20
   *        -I dir              头文件搜索目录            -I /my/include/
   *    数学选项:
   *
   *    优化选项:
   *        -cl-opt-disable     禁用所有优化
   *        -cl-mad-enable      运行使用mad指令执行乘加操作 计算会减低精度 a*b+c
   *    其他选项:
   *
   * Note:
   *    1. Program一次可以多个kernel
   *    2. Program对象创建在一个上下文 但是跟设备无关(clCreateProgramWithSource)
   *    3. Program 构建的时候 就要指定在什么设备上build(clBuildProgram) 不指定的话 就上下文的所有的设备
   *    4. 在OpenCL1.1中，创建program，直接用clBuildProgram即可
   *        在OpenCL1.2中，新添加了一种方式: 先compiler 再linker  允许将多个源码编译成一个可执行程序
   *        clCompileProgram    将一段内核代码编译成非可执行的cl::Progam对象(类似于obj文件)
   *        clLinkProgram       将多个obj对象连接生成新的可执行的cl::Program对象(Executable Program)
   *
   *
   */
void MySobel::createProgramAndKernel(){

    cl_int status = CL_SUCCESS ;
    const char * source[] = { KERNEL_SRC };
    size_t sourceSize[]  = { strlen(source[0]) };
    mProgram = clCreateProgramWithSource(mContext, 1, source, sourceSize,  &status);
    if(status != CL_SUCCESS) {
        ALOGE("clCreateProgramWithSource fail with %d " , status );
        assert( status == CL_SUCCESS );
    }

    char options[256] ;
    snprintf(options , 256 , "-D SOBEL_VALUE=%d ", N  ); // 编译选项
    status = clBuildProgram(mProgram, 1, &mDevice, options , NULL, NULL);
    if(status != CL_SUCCESS) {
        char info_buf[1024];
        clGetProgramBuildInfo(mProgram, mDevice, CL_PROGRAM_BUILD_LOG, 1024, info_buf, NULL);
        ALOGE("clBuildProgram fail with %d info %s " , status , info_buf ); // 可以打印编译错误在哪里
        assert( status == CL_SUCCESS );
    }

    mKernel = clCreateKernel(mProgram, "Sobel", &status);
    if(status != CL_SUCCESS) {
        ALOGE("clCreateKernel fail with %d " , status );
        assert( status == CL_SUCCESS );
    }
}

/*
 * cl_mem clCreateBuffer (  cl_context context,
 *                          cl_mem_flags flags,     标记 分配和使用信息 (allocation and usage)
 *                          size_t size,
 *                          void *host_ptr,
 *                          cl_int *errcode_ret)
 *      创建buffer对象(其中一中内存对象)
 *
 *  CL_MEM_READ_WRITE       默认
 *  CL_MEM_WRITE_ONLY       内存对象(buffer or image object)在kernel里面只写 如果这情况在kernel里面读 后果不确定
 *  CL_MEM_READ_ONLY
 *
 *
 *  CL_MEM_USE_HOST_PTR(host_ptr 不能为null)
 *      刚开始buffer object的值是来自于host_ptr
 *      buffer object处理之后,写回到host_ptr主机内存中
 *      OpenCL内部 可以缓存host_ptr指向的数据 这样同一个设备上执行的kernel都可以使用(cached copy)
 *
 *  CL_MEM_COPY_HOST_PTR (host_ptr 不能为null)
 *      OpenCL内部 为这个内存对象 分配内存 并且  内存对象的初始值使用host_ptr(使用复制)
 *      buffer object操作完成后的值 不会写回到host_ptr主机内存中
 *      CL_MEM_COPY_HOST_PTR 与 CL_MEM_USE_HOST_PTR 互斥
 *      CL_MEM_COPY_HOST_PTR 和 CL_MEM_ALLOC_HOST_PTR 一起 代表 分配的内存在 主机端
 *
 *  CL_MEM_ALLOC_HOST_PTR
 *      OpenCL内部 分配 主机访问的内存
 *
 *  CL_MEM_USE_PERSISTENT_MEM_AMD ???  MALI or Adreno
 *      clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_USE_PERSISTENT_MEM_AMD,  sizeof(float)*N,  0, &err);
 *      clEnqueueMapBuffer
 *      clEnqueueUnmapMemObject
 */
void MySobel::createMemoryObject(uint8_t* input , uint8_t* output , uint32_t* info ) {

    cl_int status = CL_SUCCESS ;
    mInImgMem = clCreateBuffer(mContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               mImageHeight * mImageWidth * 1 /*gray CV_8UC1*/ , input, &status);
    if(status != CL_SUCCESS) {
        ALOGE("clCreateBuffer fail with %d " , status );
        assert( status == CL_SUCCESS );
    }


    mOutImgMem = clCreateBuffer(mContext, CL_MEM_WRITE_ONLY  ,
                                mImageHeight * mImageWidth * 1 , NULL, &status);
    if(status != CL_SUCCESS) {
        ALOGE("clCreateBuffer fail with %d " , status );
        assert( status == CL_SUCCESS );
    }

    mInfoMem = clCreateBuffer(mContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(uint32_t) * 3 /*width height stride */, info, &status);
    if(status != CL_SUCCESS) {
        ALOGE("clCreateBuffer fail with %d " , status );
        assert( status == CL_SUCCESS );
    }

}

/*
     * https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clSetKernelArg.html
     *
     * cl_int clSetKernelArg (	cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
     *
     *
     * clSetKernalArg会拷贝指向的内容 所以clSetKernalArg返回后可修改arg_value指向的内存 重复使用
     * 参数使用在每次调用 clEnqueueNDRangeKernel and clEnqueueTask
     *
     * 1. arg_value 可以指向 内存对象 (buffer or image object) 但必须是跟kernel同一上下文
     * 3. 如果变量声明为 __local   , arg_value 必须为NULL (只需要设置大小 告诉OpenCL内部如何分配局部内存)
     * 4. 如果变量类型为 sampler_t , arg_value 必须指向一个  sampler object.
     * 5. 其他内核变量  arg_value 必须指向实际数据
     * 6. 如果变量声明为‘自定义或者内置类型’的‘指针’ 并用 __global or __constant 修饰/限定，(__local arg_value一定NULL ，__private只能基本类型(ocl float4 cpu float[]) )
     *      那么内存对象，必须是buffer object(非image object)或者NULL(如，用于全局output的 OpenCl内部分配空间)
     *    如果变量类型是 ‘image2d_t/image3d_t’  作为参数值的内存对象，必须是 2D image object/3D image object
     *
     *    __constant修饰的变量 指向的  内存对象(buffer object) 不能超过 CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
     *    __constant修饰的变量数目 不能超过 CL_DEVICE_MAX_CONSTANT_ARGS
     *
     *
     * 传入kernel的'指针参数'必须是__global, __constant, 或者__local型的(指向指针的指针不可以作为参数传给kernel函数)
     *
     * 内核参数或声明变量 都会有地址空间限定符，地址空间限定符的主要作用是指出数据应该保存在哪个地方
     * 地址空间限定符有4个:
     *
     * __private  只在一个工作项中有效
     *              如果内核参数或者内核程序中的变量声明没有加限定符,那么他将被保存在私有内存中
     *              如果'指针变量'没有加限定符，他就会被设置'指向'私有内存
     *              image2d_t和image3d_t型'指针'会一直'指向全局内存'
     *
     * __local    保存工作组中工作项的数据
     *              局部内存在同一个工作组内存是可以共享的
     *              主机不能读写局部、私有内存。但是主机可以配置局部、私有内存
     *              只会针对处理内核的各个工作组分配一次 然后在工作组处理结束之后释放内存
     *              局部内存的访问速度比全局内存更快，因此，最好是
     *                      先将数据从全局内存读取到局部内存中
     *                      然后在局部内存中进行处理
     *                      在工作项处理完局部数据之后，再将结果写到全局内存中，再传输回主机
     *
     * __constant  常数内存 只可以读
     *
     * __global    保存一个设备中的数据
     *              一个设备中的各个工作组、各个工作项是可以共享的
     *              主机与设别之间的数据通信是通过全局内存实现的
     *              主机和设备都可以读写访问
     *              当主机应用程序将缓存对象传输给设备，缓存数据是存放在‘全局/常数空间’中
     *              当主机从设备中读取缓存对象，数据将来自设备的全局内存。
     *              全局/常数内存往往是一个opencl兼容设备上最大的内存区域 但是访问速度最慢
     *
     *  限定符所限定的对象:
     *      __global:可以限定所有的'内核参数' 内核之中所声明的'指针变量'
     *      __local:内核参数 以及 内核中声明的变量  都不能够对其进行'直接初始化' __local float x = 4.0; 报错 -> __local float x; x = 4.0;
     *      __private:内核参数 所有'非内涵函数??? inline??'的参数和变量
     *
     *  主机'配置' 局部内存__local
     *      主机与设别之间的数据通信是通过全局内存实现的
     *      主机'不能读写'局部、私有内存
     *      主机'可以配置'局部、私有内存
     *      主机可以告诉'设备'如何为'内核参数分配局部内存'
     *
     *
     *  主机'配置' 私有内存
     *      私有内存的访问速度最快
     *      内存空间最小
     *      '内核参数的私有数据'可以由主机来进行初始化：
     *          clSetKernelArg 一个参数设定为基本数据类型指针，如int*、float*，char*
     *                          内核函数中对应的私有内核参数必须是基本数据类型，对应为int、float、char
     *                          int num_iteration = 4;
     *                          clSetKernelArg(kernel,0,sizeof(num_iteration),&num_iteration);
     *                          ...
     *                          __kernel void proc_data(int num_iteration,...){
     *                          }
     *                          该 内核函数参数 没有限定符，因此默认是'私有内存' , 而且’不是一个指针‘
     *                          那么每一个'工作项'都会有一个'自己的副本'
     *
     *  全局/常数数据只能通过'引用传递'(指针??)的方式给内核，而私有数据是'值传递'的方式
     *
     *  私有内核参数必须是'基本数据类型'，但是不一定需要是'标量'，也可以是'向量'
     *                          float nums[4] = {0.0f,1.0f,2.0f,3.0f};
     *                          clSetKernelArg(kernel,0,sizeof(nums),nums);
     *                          ...
     *                          __kernel void proc_data(float4 values,...){ // values不是4个元素的数组 而是float4型向量
     *                              values[0] <-- 这是错误的
     *                              values.x values.y values.z values.w <-- 向量访问方式
     *                          }
     *
     *  一般情况下：
     * clSetKernelArg，指针指向内存对象(cl_mem)，那么对应的内核参数必须是声明为__global或__constant类型的指针。
     * clSetKernelArg，指针被声明NULL，对应的内核参数必须被声明为__local类型的指针，且主机程序能够做的只是 告诉设备如何为内核参数 分配局部内存
     * clSetKernelArg，指针指向的是基本数据类型，内核参数就不会是指针，也不需要有任何地址限定符(默认 __private )
     */
/*
    typedef union
    {
        cl_float  CL_ALIGNED(16) s[4];
        #if __CL_HAS_ANON_STRUCT__
           __CL_ANON_STRUCT__ struct{ cl_float   x, y, z, w; };
           __CL_ANON_STRUCT__ struct{ cl_float   s0, s1, s2, s3; };
           __CL_ANON_STRUCT__ struct{ cl_float2  lo, hi; };
        #endif
        #if defined( __CL_FLOAT2__)
            __cl_float2     v2[2];
        #endif
        #if defined( __CL_FLOAT4__)
            __cl_float4     v4;
        #endif
    }cl_float4;

    vector的前一半为lo，后一半为hi
    int4 v = (int4) 7 =(int4)(7,7,7,7)
    v=(in4)(1,2,3,4)
    int2 v2=v.lo ->(1,2)  低半部
         v2=v.hi ->(3,4)
    v2.v.odd -> (2,4)     偶数项
 */
void MySobel::setKernelArg(){

    cl_int status = CL_SUCCESS ;
    status |= clSetKernelArg(mKernel, 0, sizeof(cl_mem), &mInImgMem); // __global char *array1
    status |= clSetKernelArg(mKernel, 1, sizeof(cl_mem), &mOutImgMem); // __global char *array2
    status |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &mInfoMem); // __global int *array3

    if(status != CL_SUCCESS) {
        ALOGE("clSetKernelArg fail with %d " , status );
        assert( status == CL_SUCCESS );
    }

}

void MySobel::runKernel() {


#if TEST_KERNEL_COST_TIME == 1
//  在 OpenCL 1.1 中，建议不再使用 OpenCL 1.0 的如下特性：不再支持 API clSetCommandQueueProperty
//  cl_command_queue_properties old ;
//  clSetCommandQueueProperty(mCmdQueue,  CL_QUEUE_PROFILING_ENABLE , CL_TRUE , &old );
//  ALOGD("CommandQueue Property old is 0x%x" , old );
#endif


    cl_int status = CL_SUCCESS ;
    size_t local[] = { 16, 16 }; // MT6797 一个工作组只有256工作项 16*16  clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(maxItem), &maxItem, NULL);
    size_t global[2] = { 0, 0 };
    global[0] = ( mImageWidth % local[0] == 0 ?  mImageWidth  : (mImageWidth  + local[0] - mImageWidth  % local[0]));
    global[1] = (mImageHeight % local[1] == 0 ?  mImageHeight : (mImageHeight + local[1] - mImageHeight % local[1]));

#if TEST_KERNEL_COST_TIME == 1
    cl_event ev;
    status =  clEnqueueNDRangeKernel(mCmdQueue, mKernel, 2, NULL, global,  local, 0, NULL, &ev);
#else
    status = clEnqueueNDRangeKernel(mCmdQueue, mKernel, 2, NULL, global,  local, 0, NULL, NULL);
#endif
    if(status != CL_SUCCESS) {
        ALOGE("clEnqueueNDRangeKernel fail with %d " , status );
        assert( status == CL_SUCCESS );
        /*
         * #define CL_INVALID_WORK_GROUP_SIZE  -54
         */
    }
//    clFlush(mCmdQueue);
    clFinish(mCmdQueue);


#if TEST_KERNEL_COST_TIME == 1
    cl_ulong startTime = 0, endTime = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,  sizeof(cl_ulong), &startTime, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,  sizeof(cl_ulong), &endTime, NULL);
    cl_ulong kernelExecTimeNs = endTime - startTime; // in nanoseconds
    ALOGD("clEnqueueNDRangeKernel Kernel Exec Time :%8.6f ms\n", kernelExecTimeNs*1e-6);
#endif

}

/*
 * cl Enqueue ReadBuffer
 * cl Enqueue NDRangeKernel
 * 都是把命令放到命令队列
 *
    cl_int clEnqueueReadBuffer (
        cl_command_queue command_queue, // 命令队列
        cl_mem buffer,                  // 命令队列和Buffer对象(cl_mem 其中一种内存对象) 都必须在 同一个上下文
        cl_bool blocking_read,          // CL_TRUE  不会返回 直到 buffer中的数据读到ptr指向的内存
                                        // CL_FALSE 立刻返回 ptr指向的内存不能使用 event返回event对象可以用来查询状态
        size_t offset,                  // cl_mem buffer的 读取 偏移
        size_t cb,                      // cl_mem buffer的 读取 大小
        void *ptr,                      // 主机内存(host memory)
        cl_uint num_events_in_wait_list,// event_wait_list列表长度
        const cl_event *event_wait_list,// 事件等待列表 列表中的事件可以看成同步点 当前命令必须在这些时间完成之后才能执行
        cl_event *event                 // 返回事件 指示 这个命令(状态) 可以用来查询命令状态或等待这个命令完成
        )
 */
void MySobel::getKernelResult( uint8_t* output /*allocated by caller */){

//    cl_uchar sqrt_float[ mImageHeight * mImageWidth ] ;
//    memset( sqrt_float, 0  , mImageHeight * mImageWidth * sizeof(cl_uchar) );
//    ALOGD("sizeof(cl_uchar) = %zd " , sizeof(cl_uchar) );  // sizeof(cl_float) = 4  sizeof(cl_uchar) = 1
//    clEnqueueReadBuffer(mCmdQueue,
//                        mOutImgMem,
//                        CL_TRUE,// block
//                        0,
//                        mImageHeight * mImageWidth * sizeof(cl_uchar)/*each pixel 1 byte: gray uint8_t*/,
//                        sqrt_float , 0,
//                        NULL, NULL);

    clEnqueueReadBuffer(mCmdQueue,
                        mOutImgMem,
                        CL_TRUE,// block
                        0,
                        mImageHeight * mImageWidth * sizeof(cl_uchar)/*each pixel 1 byte: gray uint8_t*/,
                        output , 0,
                        NULL, NULL);


//    ALOGD("write dump begin ");
//    FILE * fp = fopen("/mnt/sdcard/gpu_strict.txt", "w+" );
//    for( int i = 0 ; i < mImageHeight ; i++  ){
//        for( int j = 0 ; j < mImageWidth ; j+=4 ){ // 图片宽要是4的倍数
//            fprintf(fp, "%4u %4u %4u %4u "
//                    "\n"// for beyond compare
//                    ,
//                    sqrt_float[i*mImageWidth + j ],
//                    sqrt_float[i*mImageWidth + j + 1 ],
//                    sqrt_float[i*mImageWidth + j + 2 ],
//                    sqrt_float[i*mImageWidth + j + 3 ]
//                    );
//
//        }
//        fprintf(fp,"\n");
//    }
//    fclose(fp);
//    ALOGD("write dump done ");
}

/*
 * cl_int clReleaseMemObject(cl_mem memobj)
 *      减少内存对象的引用计数
 *      内存对象会被删除 当 引用计数=0 和 依赖内存对象的命令已经执行完毕
 */
void MySobel::releaseMemoryObject() {

    clReleaseMemObject(mInImgMem);
    clReleaseMemObject(mOutImgMem);
    clReleaseMemObject(mInfoMem);

}

/*
 *
 *  cl_int clReleaseProgram(cl_program program)
 *      Program对象会被删除 当Program对象关联的kernel被删除 并且Program的引用计数为0
 *      返回 CL_SUCCESS
 *
 *  cl_int clReleaseKernel(cl_kernel kernel)
 *      Kernel对象被删除 当kernel对象的实例数目为0 而且 入队的命令也不再需要Kernel对象
 *
 */
void MySobel::releaseProgramAndKernel(){
    cl_int status = CL_SUCCESS ;
    status = clReleaseKernel(mKernel);
    if(status != CL_SUCCESS) {
        ALOGE("clReleaseKernel fail with %d " , status );
        assert( status == CL_SUCCESS );
    }

    status = clReleaseProgram(mProgram);
    if(status != CL_SUCCESS) {
        ALOGE("clReleaseProgram fail with %d " , status );
        assert( status == CL_SUCCESS );
    }
}

/*
 * cl_int clReleaseCommandQueue(cl_command_queue command_queue)
 *      命令队列删除 当 引用计数为0 和 所有入队的命令执行完毕 e.g kernel executions, memory object updates??
 *
 * cl_int clReleaseContext(cl_context context)
 *      上下文删除 当 引用计数为0 和 所有关联这个上下问的对象(内存对象 命令队列)释放了
 */
void MySobel::unsetupContextAndCmdQueue(){
    cl_int status = CL_SUCCESS ;

    status = clReleaseCommandQueue( mCmdQueue );
    if(status != CL_SUCCESS) {
        ALOGE("clReleaseCommandQueue fail with %d " , status );
        assert( status == CL_SUCCESS );
    }

    status = clReleaseContext( mContext );
    if(status != CL_SUCCESS) {
        ALOGE("clReleaseContext fail with %d " , status );
        assert( status == CL_SUCCESS );
    }
}




MySobel::MySobel(){
    {
        CostHelper c ;
        findPlatformAndDevices() ;
        setupContextAndCmdQueue();
        ALOGD("setup Context cost  %" PRId64 " us " , c.Get() );
    }
    {
        CostHelper c ;
        createProgramAndKernel();
        ALOGD("Setup Program& Kernel cost   %" PRId64 " us " , c.Get() );
    }

}

bool MySobel::doSobel(uint8_t* in /*gray CV_8UC1 */, uint8_t* out /*allocate by user*/, uint32_t width , uint32_t height, uint32_t stride ){

    mImageWidth = width ;
    mImageHeight = height ;
    mImageStride = stride ;

//    ALOGD("write dump begin ");
//    FILE * fp = fopen("/mnt/sdcard/gpu_gray_byte.txt", "w+" );
//    for( int i = 0 ; i < mImageHeight ; i++  ){
//        for( int j = 0 ; j < mImageWidth ; j+= 4 ){ // 图片宽要是4的倍数
//            fprintf(fp, "%4u %4u %4u %4u " ,
//                    in[i*mImageWidth + j ],
//                    in[i*mImageWidth + j + 1 ],
//                    in[i*mImageWidth + j + 2 ],
//                    in[i*mImageWidth + j + 3 ]
//            );
//        }
//        fprintf(fp,"\n");
//    }
//    fclose(fp);
//    ALOGD("write dump done ");

//    double result = sqrt(18*18 + 2*2)=sqrt(652)   sqrt(28*28 + 6*6 )=sqrt(4740)
//    ALOGD("sqrt result = %f " , result );

    ALOGD("width = %u height = %u stride = %u" , width , height , stride );

    {
        CostHelper c ;
        uint32_t info[3] = { width , height , stride } ;
        createMemoryObject( in  , out  ,  info );
        setKernelArg();
        runKernel();
        getKernelResult( out );
        releaseMemoryObject();
        ALOGD("run Kernel CPU cost   %" PRId64 " us " , c.Get() );
    }

    return true ;
}

MySobel::~MySobel(){
    {
        CostHelper c ;
        releaseProgramAndKernel();
        ALOGD("release Program& Kernel cost   %" PRId64 " us " , c.Get() );
    }
    {
        CostHelper c ;
        unsetupContextAndCmdQueue();
        ALOGD("unSetup Program& Kernel cost  %" PRId64 " us " , c.Get() );
    }
}



