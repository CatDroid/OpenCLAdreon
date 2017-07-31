
/*
 * 高通平台 头文件来自 KhronosGroup https://github.com/KhronosGroup/OpenCL-Headers opencl20
 *          Note that Snapdragon 820 and other chipsets that have Adreno 5xx GPUs support OpenCL 2.0 Full profile.
 *
 *          https://developer.qualcomm.com/forum/qdn-forums/software/adreno-gpu-sdk/29883
 *        库文件 来自小米5
 *
 *
 * MTK平台 头文件来自 github https://developer.arm.com/technologies/compute-library --> https://github.com/ARM-software/ComputeLibrary
 *        库文件 来自MT6797
 */

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>

#include <CL/cl.h>

#include <jni.h>
#include <android/log.h>
#include "CostHelper.h"

#define LOG_TAG "OCL"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define LEN(arr) sizeof(arr) / sizeof(arr[0])
#define N 1024
#define NUM_THREAD 128  // 一个工作组里面的工作项目不能超过 CL_DEVICE_MAX_WORK_ITEM_SIZES
                        // 否则计算错误 以及 clEnqueueNDRangeKernel 返回 -55 (e.g  NUM_THREAD 512 )
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

// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
/*
    OpenCL采用的数据并行模型就是采用CUDA的数据并行模型

    OpenCL			CUDA
    Kernel函数		Kernel函数
    主机程序		    主机程序
    N-DRange		网格
    工作项			线程
    工作组			线程块

    在设备端的程序中，CUDA主要是通过预定义的变量进行访问，而OpenCL是通过预定义的API访问

    OPENCL					含义									    CUDA
    get_global_id(0) 		工作项在X维度上的全局索引  				    blockIdx.x*blockDim.x+threadIdx.x
    get_local_id(0)			工作项在工作组中X维度上的局部索引 		        threadIdx.x
    get_global_size(0) 		N-DRange中X维度的大小 即线程数量		        gridDim.x*blockDim.x
    get_local_size(0)		每个工作组X维度上的大小 即工作项在X维度上数目    blockDim.x

    get_work_dim            工作组维度数目 1(X) 2(X Y) 3(X Y Z) 应该与全局的NRRange索引空间一样大
    get_num_groups          工作组数目
    get_group_id            工作组ID

    C99标准扩展   99年ansi C
    不支持:    头文件、函数指针、递归、变长数组
    增加类型:
            向量类型 char2  float4  int8
            图像类型 image2d_t image3d_t  sampler_t
            事件类型 event_t (可见于同步)

    类型转换:
        convert转换
            按照'变量语意'的类型转换
            convert_destType<_sat><_roundingMode>
            destType 目标类型
            _sat:超出范围自动归结为最大或最小表示的数
            _roundingMode:  _rte:表示成最接近的偶数
                            _rtz:朝0接近
                            _rtp:朝正无穷大
                            _rtn:朝负无穷大
            float4 f4=(float4)(1.0f,2.0f,3.0f,4.0f)
            int4 i4 = convert_int4_sat_rte(f4)

        as转换
            这是根据'bit值'重新解释的类型转换,这 个转换会保持 'bit值' 不变，在此基础上根据desttype重新解释数值
            as_desttype  e.g as_int4()   as_float4()
            desttype是目标类型
            其中转换前后类型的vector size是要一样的 float4 <-- int4
            float4 f4=(float4)(1.0f,2.0f,3.0f,4.0f)
            int4 i4=as_int4(f4)

    Work_group函数: 用于一个group内的item间的交互
        同步函数  void barrier (	cl_mem_fence_flags flags)
                 一个工作组内的所有工作项 必须全部执行完这个barrier函数之后 才能继续进行后续的事情
                 也可看做这是所有item的一个同步点 不管谁快谁慢 必须到这个点停一下 等大家都到了这个点
                 flags  CLK_LOCAL_MEM_FENCE和CLK_GLOBAL_MEM_FENCE
                 e.g barrier(CLK_LOCAL_MEM_FENCE);

        异步的内存copy和prefetch函数
                async_work_group_copy  完成global与local之间 '异步'内存拷贝 拷贝可能用DMA
                wait_group_events      来等待上面的event返回，用于同步
                async_work_group_strided_copy  区别在从src抽取一部分域出来到dst
                                                e.g 图形学 经常用一个大数组表示颜色 法向 纹理坐标等等
                                                他们是连在一起的 如
                                                    {color1,color2,color3,tex0,tex1,color1,color2,color3,text0,tex1,....}
                                                这时我们需要抽取其中的color信息出来
 */
/** Device side OpenCL C code. **/
const char* src[] = {
        "#ifdef cl_arm_printf \n"
        "#pragma OPENCL EXTENSION cl_arm_printf : enable\n"
        "#endif \n"
        "  __kernel void reduction(  \n"
        "  __global int *data,     \n"
        "  __global int *output,   \n"   // 三个参数  都是指针类型变量 (int *)
        "  __local int *data_local   \n" // 三个参数  最后一个声明为 __local
        "  )  \n"
        " {   \n"
        "  int gid=get_group_id(0);   \n"       // 获得工作组ID(相当于CUDA的线程块)   因为clEnqueueNDRangeKernel给定的维度数目是1
        "  int tid=get_global_id(0);    \n"     // 获得全局ID(相当于CUDA的线程)
        "  int size=get_local_size(0);   \n"    // 一个工作组中工作项的数目
        "  int id=get_local_id(0);     \n"      // 获得局部ID
        "  data_local[id]=data[tid];   \n"
        "#ifdef cl_arm_printf\n"
//        "  printf( \"work item -%d- \\n\", tid  );\n" // 必须加上\\n
         "  int item_id = tid;\n"  // 暂未清楚为什么 int item_id = tid 会出错 但是 int item_id = size 不会
         "  printf(\"work item -%d- \\n\" , size );\n"
         "#endif\n"
        "  barrier(CLK_LOCAL_MEM_FENCE);   \n"
        "  for(int i=size/2;i>0;i>>=1){    \n" // 每次减半
        "      if(id<i){   \n"
        "          data_local[id]+=data_local[id+i];   \n" // id 超过 i 的 以后就不用计算了~ 直接barrier
//        "          barrier(CLK_LOCAL_MEM_FENCE);   \n"   // 只有工作组里所有工作项都过了mem fence 大家才会一起继续执行
        "      }   \n"
        "      barrier(CLK_LOCAL_MEM_FENCE);   \n"
        "  }    \n"
        "  if(id==0){    \n" // 局部ID为0的工作项
        "      output[gid]=data_local[0];   \n" // 每个工作组 计算完结果 放到  output[gid]  output长度是工作组数量
        "  }    \n"
        " }   \n"
}; // 这个算法要 保证工作组的工作项数目 是 2个阶乘 这样确保了 每次除2  直到最后出现奇数一定是1的情况下

/*
 *      算法原理:
 *         假如一个工作组如下 有20个工作项
 *         0  1  2  3  4  5  6  7  8  9    10  11 12 13 14 15 16 17 18 19   <--- data_local
 *         .  .  .  .  .  .  .  .  .  .     .  .  .  .  .  .  .  .  .  .
 *      第一轮                            | 一半 (size/2)
 *         0和10 1和11 2和12  ... (data_local[id]和data_local[id+size/2}) 相加 放到 0 1 2 .... (data_local[id])
 *        0+10  1+11  2+12  3+13  4+14    5+15  6+16  7+17  8+18  9+19       结果只有 0~size/2
 *        .     .     .     .     .       .     .     .     .     .
 *      第二轮                           | 再一半 (size/2 >> 1 )
 *        (0+10)+(5+15)  (1+11)+(6+16)   (2+12)+(7+17)   (3+13)+(7+17)  (4+14)+(9+19)   结果只有 0~size/4
 *                                      |
 *                                      ? 这里会有错误  其实工作项  应该有 2的阶乘 2^N 个  这样保证
 *
 *
 *       0      1      2       3       4       5       6       7         8  9  10  11  12  13  14  15
 *       .      .      .       .       .       .       .       .         .  .  .   .   .   .   .   .
 *                                                                     |
 *       (0+8)  (1+9)  (2+10)  (3+11)  (4+12)  (5+13)  (6+14)  (7+15)
 *                                    |
 *       (0+8)+(5+13)  (1+9)+(5+13)  (2+10)+(6+14)  (3+11)+(7+15)
 *                                  |
 *       ((0+8)+(5+13))+((2+10)+(6+14))  ((1+9)+(5+13))+((3+11)+(7+15))
 *                                     |
 *        all~
 */
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


void printf_callback( const char *buffer, size_t len, size_t complete, void *user_data ) {
    //printf( "%.*s", len, buffer );
    //ALOGD("%s", buffer);
}

void Context_cmd()
{
    cl_int  errcode_ret = CL_SUCCESS  ;
#if USING_MTK == 1
#define  CL_PRINTF_CALLBACK_ARM    0x40B0
#define  CL_PRINTF_BUFFERSIZE_ARM  0x40B1
    /* Create a cl_context with a printf_callback and user specified buffer size. */
    cl_context_properties properties[] =  {
            /* Enable a printf callback function for this context. */
            CL_PRINTF_CALLBACK_ARM,   (cl_context_properties) printf_callback,
            /* Request a minimum printf buffer size of 4MiB for devices in the
               context that support this extension. */
            CL_PRINTF_BUFFERSIZE_ARM, (cl_context_properties) 0x100000 ,// 0x100000,
            //CL_CONTEXT_PLATFORM,      (cl_context_properties) platform,
            0
    };
    context=clCreateContext(properties,num_device,devices,NULL,NULL,&errcode_ret);
    if(errcode_ret != CL_SUCCESS) ALOGE("clCreateContext fail");
    ALOGD("using MTK");
#else
    context=clCreateContext(NULL,num_device,devices,NULL,NULL,&errcode_ret);
    if(errcode_ret != CL_SUCCESS) ALOGE("clCreateContext fail");
    ALOGD("using dragon");
#endif
    ALOGD("clCreateContext return %d " , errcode_ret);

    cmdQueue=clCreateCommandQueue(context,devices[0],0,&errcode_ret);
    if(errcode_ret != CL_SUCCESS) ALOGE("clCreateCommandQueue fail");
}


void Create_Buffer(int *data)
{
    buffer=clCreateBuffer(
            context,
            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  // CL_MEM_READ_ONLY CL_MEM_WRITE_ONLY 是对kernel来说
            sizeof(int)*N,  data,  // CPU 端内存
            &err);
    sum_buffer=clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY,
            sizeof(int)*num_block, 0,
            &err);
}

void Create_program()
{
    cl_int  errcode_ret = CL_SUCCESS ;


    program = clCreateProgramWithSource(context, LEN(src) /*1*/, src, NULL, &errcode_ret);
    if( errcode_ret != CL_SUCCESS ) ALOGE("clCreateProgramWithSource fail with %d " , errcode_ret );

    // 编译Program成二进制 from the program source or binary
    errcode_ret = clBuildProgram(program, num_device, devices, NULL, NULL, NULL );
    if( errcode_ret != CL_SUCCESS ) ALOGE("clCreateProgramWithSource fail with %d " , errcode_ret );

    // 创建kernel对象  program必须已经编译   "reduction"必须在program中以__kernel修饰
    // #define CL_INVALID_PROGRAM_EXECUTABLE               -45 没有成功编译program
    kernel = clCreateKernel(program, "reduction", &errcode_ret);
    if( errcode_ret != CL_SUCCESS ) ALOGE("clCreateKernel fail with %d" ,  errcode_ret );

}


void Set_arg()
{
    err=clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);       // __global int *data
    err=clSetKernelArg(kernel,1,sizeof(cl_mem),&sum_buffer);   // __global int *output
    err=clSetKernelArg(kernel,2,sizeof(int)*NUM_THREAD ,NULL); // __local int *data_local  __local 声明的参数变量
    // 要求设备 为 内核参数 分配 局部内存   (如果变量声明为 __local , arg_value 必须为NULL)
    // 配置 内核参数‘指针变量data_local’ 分配足够保存NUM_THREAD个int的内存空间 并用把指针变量data_local指向这个内存空间
}

void Execution()
{
    cl_int  errcode_ret = CL_SUCCESS ;

    const size_t globalWorkSize[1]={N};         // 工作项  一维  维度数 = 1 总共 1024
    const size_t localWorkSize[1]={NUM_THREAD}; // 工作组       128 个工作项  共8个工作组
    // clEnqueueNDRangeKernel 将数据并行的kernel入队并执行
    // 应用程序指明
    // 全局的工作量(global work size，即并行执行这个kernel的工作项(work item)的个数)
    // 局部的工作量(local work size，即一个工作组(work-group)中工作项的个数)

    errcode_ret = clEnqueueNDRangeKernel(cmdQueue,   // 命令队列
                               kernel,     // 由 clCreateKernel 创建的核心  cl_kernel 与 cl_program 有关联
                               1,          // 维度数量 1,2,3
                               NULL,       // 全局索引 每个维度 开始偏移
                               globalWorkSize,
                               localWorkSize,
                               0,NULL,NULL);
    if( errcode_ret != CL_SUCCESS ){
        ALOGE("clEnqueueNDRangeKernel fail with %d" ,  errcode_ret );
        // CL_INVALID_  cl.h

        // 如果工作组中工作项数目超过 CL_DEVICE_MAX_WORK_ITEM_SIZES(工作组每个维度的工作项数目)  (这里是localWorkSize or NUM_THREAD )
        // 返回错误 CL_INVALID_WORK_ITEM_SIZE
        // if the number of work-items specified in any of local_work_size[0], ... local_work_size[work_dim - 1]
        // is greater than the corresponding values specified by
        // CL_DEVICE_MAX_WORK_ITEM_SIZES[0], .... CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]
    }
    errcode_ret = clFinish(cmdQueue);
    if( errcode_ret != CL_SUCCESS ) ALOGE("clFinish fail with %d" ,  errcode_ret );

}

void CopyOutResult(int*out)
{
    err=clEnqueueReadBuffer(cmdQueue,sum_buffer,CL_TRUE,0,sizeof(int)*num_block,out,0,NULL,NULL);
}


int  test()
{
    int* in,*out;
    num_block=N/NUM_THREAD; // 1024  / 128 = 8  这里跟后面clEnqueueNDRangeKernel时候 创建的工作项数量 和 工作组数量一样
                            // num_block 工作组的数目
    in=(int*)malloc(sizeof(int)*N);             //  输入内存对象          clSetKernelArg
    out=(int*)malloc(sizeof(int)*num_block);    //  输出内存对象                              clEnqueueReadBuffer
    for(int i=0;i<N;i++){
        in[i]=1;
    }
    Init_OpenCL();      // 获取所有平台和设备信息
    Context_cmd();      // 建立上下文和命令队列
    Create_Buffer(in);  // 创建内存对象
    Create_program();   // 创建程序对象 和 创建内核
    Set_arg();          // 设置kernel函数参数

    CostHelper c ;

    Execution();        // kernel进入队列
    CopyOutResult(out); // 拷贝出kernel运行结果

    ALOGD("cost time %" PRId64 " us " , c.Get() ); // 4959 4627 4256 4867 5276 5144

    ALOGD("num_block = %d " , num_block );
    int sum=0;
    for(int i=0;i<num_block;i++){ // 所有工作组的结果加起来
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
    char buffer[1024]; memset(buffer,0,sizeof(buffer));
    /*
     * 每一个 cl_platform_id 结构表示一个在主机上的OpenCL执行平台
     * 就是指电脑中支持OpenCL的硬件，如nvidia显卡，intel CPU和显卡，AMD显卡和CPU等
     */
    cl_uint num_platforms;
    cl_platform_id *platforms;

#ifdef NDEBUG
#warning ("NDEBUG is Defined  ( assert is Disable ) ")
#else
#warning ("NDEBUG is Not Defined")
#endif

    /*
     * 询问主机有多少platforms
     *
     * 返回值 -1 fail ， 0 success
     * 第一个参数为1
     *              代表我们需要取最多1个platform。可以改为任意大如：INT_MAX整数最大值
     *              但是据说0，否则会报错，实际测试好像不会报错
     * 第二个参数为NULL
     *              代表要咨询主机上有多少个platform，并使用num_platforms取得实际flatform数量
     */
    err = clGetPlatformIDs(0, NULL, &num_platforms); // 调用两次clGetPlatformIDs函数，第一次获取可用的平台数量，第二次获取一个可用的平台
    if(err < 0) {
        ALOGE("Couldn't find any platforms.");
        exit(1);
    }

    // 本人计算机上显示为2，有intel和nvidia两个平台
    // 最高支持的OpenCL版本，本机显示：OpenCL1.1 CUDA 4.2.1
    ALOGD("I have platforms: %d\n", num_platforms);

    // 创建 cl_platform_id 并分配空间
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);

    // 第二个参数用指针platforms存储platform
    clGetPlatformIDs(num_platforms, platforms, NULL);

    // for(i=0; i<num_platforms; i++)  获取所有的主机上的platforms信息/获取额外的平台信息
    size_t ext_size = 0 ; // 第二个参数指明了需要什么样的信息 第五个参数ext_size获取额外信息的长度
                          // CL_PLATFORM_EXTENSIONS opencl支持的扩展功能信息
    clGetPlatformInfo(platforms[0], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);
    ALOGD("platform extra data max size %zd ", ext_size );
    char* ext_data = (char*)malloc(ext_size); memset(ext_data,0,ext_size );
    clGetPlatformInfo(platforms[0], CL_PLATFORM_EXTENSIONS, ext_size ,ext_data, NULL);
    ALOGD("extensions %s" , ext_data );
    // OpenCL Installable Client Driver (ICD) Loader是实现OpenCL应用程序与各硬件厂商提供的OpenCL驱动(platform)之间隔离的中间库
    // 从OpenCL 1.2开始，OpenCL提供了一个ICD扩展(cl_khr_icd),它允许不同厂商的多个OpenCL驱动(platform)共存于一个主机系统
    // 应用程序可以通过调用clIcdGetPlatformIDsKHR函数获取所有已经安装的platform的列表,自由选择使用其中的一个platform
    // 应用程序的所有OpenCL API请求将被转发到指定的平台(指定的OpenCL驱动)
    //
    // 支持ICD这一扩展功能的platform 本机的Intel和Nvidia都支持这一扩展功能
    const char icd_ext[] = "cl_khr_icd";
    if(strstr(ext_data, icd_ext) != NULL){
        ALOGD("platform support icd !");
    }else{
        ALOGD("platform do NOT support icd");
    }
    free(ext_data);

    // 获得第一个平台(platform)的 Name Vendor Profile Version
    clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR,sizeof(buffer),buffer, NULL);
    ALOGD("vendor  %s ", buffer ); // 供应商信息
    // 这个只有两个值：full profile 和 embeded profile
    clGetPlatformInfo(platforms[0], CL_PLATFORM_PROFILE, sizeof(buffer),buffer, NULL);
    ALOGD("profile  %s ", buffer );
    clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, sizeof(buffer),buffer, NULL);
    ALOGD("version  %s ", buffer );
    clGetPlatformInfo(platforms[0],CL_PLATFORM_NAME,sizeof(buffer),buffer,NULL);
    ALOGD("platform %s ", buffer );
    /*
     * 820:
     * I have platforms: 1
     * platform extra data max size 2 ???
     * platform QUALCOMM Snapdragon(TM)
     * vendor  QUALCOMM
     * profile  FULL_PROFILE
     * version  OpenCL 2.0 QUALCOMM build: commit #d57aba2 changeid #Ic27b94dfce Date: 10/26/16 Wed Local Branch:  Remote Branch:
     *
     * 6797:
     * I have platforms: 1
     * platform extra data max size 444
     * extensions
     *      cl_khr_global_int32_base_atomics    cl_khr_global_int32_extended_atomics
     *      cl_khr_local_int32_base_atomics     cl_khr_local_int32_extended_atomics
     *      cl_khr_byte_addressable_store       cl_khr_3d_image_writes
     *      cl_khr_fp64                         cl_khr_fp16
     *      cl_khr_int64_base_atomics           cl_khr_int64_extended_atomics
     *      cl_khr_gl_sharing                   cl_khr_icd
     *      cl_khr_egl_event                    cl_khr_egl_image
     *      cl_arm_core_id                      cl_arm_printf
     *      cl_arm_thread_limit_hint            cl_arm_non_uniform_work_group_size
     *      cl_arm_import_memory
     * platform ARM Platform
     * vendor  ARM
     * profile  FULL_PROFILE
     * version  OpenCL 1.1 v1.r7p0-02rel0.f2c69255b7319de8a90fdb262ee294bb
     */
    free(platforms);
    return env->NewStringUTF(buffer);
}

cl_device_type OCL_DEVICE_TYPES[] = {
    CL_DEVICE_TYPE_DEFAULT,
    CL_DEVICE_TYPE_CPU,
    CL_DEVICE_TYPE_GPU,
    CL_DEVICE_TYPE_ACCELERATOR,
    CL_DEVICE_TYPE_CUSTOM,
    CL_DEVICE_TYPE_ALL,
};

const char* ocl_type2str(cl_device_type type){
    switch(type){
        case CL_DEVICE_TYPE_DEFAULT:
            return "DEFAULT";
        case CL_DEVICE_TYPE_CPU:
            return "CPU";
        case CL_DEVICE_TYPE_GPU:
            return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR:
            return "ACCELERATOR";
        case CL_DEVICE_TYPE_CUSTOM:
            return "CUSTOM";
        case CL_DEVICE_TYPE_ALL:
            return "ALL";
        default:
            return "Unknown";
    }
}


extern "C"
JNIEXPORT jstring JNICALL Java_com_tom_opencladreon_MainActivity_getDeviceName(JNIEnv *env , jobject thisobject)
{
    cl_platform_id platform0; // cl_platform_id 不是整数!!  是一个结构体指针   结构体实现没有对外开放
    clGetPlatformIDs(1, &platform0, NULL);
    cl_int  status ;
    cl_uint numDevices = 0;
    cl_device_type validType = CL_DEVICE_TYPE_DEFAULT ;

//    status = clGetDeviceIDs( platform0,  NULL,  0, NULL, &numDevices);
//    if(status == CL_SUCCESS){
//        ALOGD("平台设备总数 %u" , numDevices );
//    }else{
//        ALOGE("可能对应平台没有任何类型设备");
//    }


    // 调用两次 clGetDeviceIDs 函数，第一次获取可用的设备数量，第二次获取一个可用的设备
    for( int i = 0 ; i < sizeof( OCL_DEVICE_TYPES)/ sizeof(cl_device_type) ; i++ ){
        //ALOGD("检测平台 设备类型 %s 设备信息:" , ocl_type2str(OCL_DEVICE_TYPES[i]) );
        status = clGetDeviceIDs( platform0,  OCL_DEVICE_TYPES[i],  0, NULL, &numDevices);
        switch(status){
            case CL_SUCCESS:
                if( OCL_DEVICE_TYPES[i]  ==  CL_DEVICE_TYPE_CPU ||
                        OCL_DEVICE_TYPES[i] == CL_DEVICE_TYPE_GPU ||
                        OCL_DEVICE_TYPES[i]== CL_DEVICE_TYPE_ACCELERATOR){
                    validType = OCL_DEVICE_TYPES[i] ;// 获取最后一个有效的设备 排除Custom All Default等
                    ALOGD("Current Valid Type %s ", ocl_type2str(OCL_DEVICE_TYPES[i]) );
                }
                ALOGD("platform 0 设备类型 %s 设备数目 %zd" , ocl_type2str(OCL_DEVICE_TYPES[i]) , numDevices);
                break;
            case CL_INVALID_PLATFORM:
                ALOGE("platform 指向一个无效的平台");
                break;
            case CL_INVALID_DEVICE_TYPE:
                ALOGE("platform 0 device_type 无效值 (%s)" , ocl_type2str(OCL_DEVICE_TYPES[i]) );
                break;
            case CL_INVALID_VALUE:
                ALOGE("platform 0 非法参数" );
                // num_entries=0且device_type不是空  或者  num_devices device_type都是空
                break;
            case CL_DEVICE_NOT_FOUND:
                ALOGE("platform 0 设备类型 %s 没有找到" , ocl_type2str(OCL_DEVICE_TYPES[i]) );
                break;
            default:
                ALOGE("Unknown Error while clGetDeviceIDs ");
                break;
        }
    }

    status = clGetDeviceIDs(
            platform0,     /*这里假设 获取平台0的设备 正常应该从clGetPlatformIDs获取设备ID */
            validType,  /*设备类型 可以是 CPU GPU ACCELERATOR   DEFAULT CUSTOM ALL */
            0, NULL, &numDevices);
    assert(status == CL_SUCCESS);

    ALOGD("Platform's Device Type %s [%" PRIu64 "]  Num Of Device [%d] " , ocl_type2str(validType), validType,  numDevices );
    cl_device_id *devices = ( cl_device_id *)malloc(sizeof(cl_device_id)* numDevices );
    assert(numDevices > 0 ) ; // 如果platform 0没有设备 这里断言; 正常应该有的
    clGetDeviceIDs(platform0, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

    char        *value;
    size_t      valueSize;
    size_t      maxItem;
    cl_uint     maxComputeUnits=0;
    cl_ulong    maxGlobalMemSize=0;
    cl_ulong    maxConstantBufferSize=0;
    cl_ulong    maxLocalMemSize=0;
    /*
    cl_int  clGetDeviceInfo(cl_device_id,cl_device_info,size_t,void *, size_t *)
                            cl_device_id    设备编号
                            cl_device_info  设备参数名字
                            size_t          参数长度
                            void*           out 存放对应参数值 (buffer需要自己申请)
                            size_t*         out 实际返回大小
     */
    // 设备上并行计算单元数目 一个work-group只在一个compute unit上执行
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    ALOGD("并行计算单元: %u\n", maxComputeUnits);

    // 工作项 工作组 global和local work-item IDS(索引空间)的最大维度(维度的数目) 该参数最小为3
    // 一个work-item对应硬件上的一个PE（processing element）
    // 一个work-group对应硬件上的一个CU（computing unit）
    // 一个work-item不能被拆分到多个PE上处理 同样 一个work-group也不能拆分到多个CU上同时处理
    //
    // 当映射到OpenCL硬件模型上时 每一个work-item运行在一个被称为'处理基元（processing element）'的抽象硬件单元上
    // 其中每个'处理基元'可以处理多个work-item

    // 如果work-item的数目不能在work-groups中均分,clEnqueueNDRangeKernel失败，返回错误码CL_INVALID_WORK_GROUP_SIZE

    // 索引空间的维度数量 是可以 1 2 3 维的
    // 处理2D图像或3D空间，work-items和work-groups可以被指定为2或3维
    //
    cl_uint max_dimensions = 0 ;
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(max_dimensions), &max_dimensions, NULL);
    ALOGD("最大维度数目 : %u\n", max_dimensions); // 目前是3   // Return type: cl_uint

    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(maxItem), &maxItem, NULL);     // Return type: size_t
    ALOGD("工作组最大工作项 %zd \n", maxItem);// 在一个compute unit中执行一个kernel的work-group中work-item的最大数目

    size_t three[3];
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(three), &three, NULL);         // Return type: size_t[]
    ALOGD("每个工作组各个维度最大大小: [%zd,%zd,%zd]\n", three[0],three[1],three[2]);
    // 在 work-group的每一个维度 声明的 work-item的 最大数目。最小值（1,1,1）
    // 返回的 size_t数目 是根据 CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS  返回的维度数量


    cl_device_fp_config fp_config;
    clGetDeviceInfo(devices[0], CL_DEVICE_SINGLE_FP_CONFIG ,sizeof(cl_device_fp_config), &fp_config ,NULL );
    ALOGD("设备单精度浮点数能力  0x%x\n" , fp_config ); //   6797 0x3f   820 0x16
    /*

        #define CL_FP_DENORM                                (1 << 0)
        #define CL_FP_INF_NAN                               (1 << 1) // < 820
        #define CL_FP_ROUND_TO_NEAREST                      (1 << 2) // < 820
        #define CL_FP_ROUND_TO_ZERO                         (1 << 3)
        #define CL_FP_ROUND_TO_INF                          (1 << 4) // < 820
        #define CL_FP_FMA                                   (1 << 5)
        #define CL_FP_SOFT_FLOAT                            (1 << 6)
        #define CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT         (1 << 7)
     */

    clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(maxGlobalMemSize), &maxGlobalMemSize, NULL);
    ALOGD("全局内存: %" PRIu64 " (MB)\n", maxGlobalMemSize/1024/1024);

    clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE,sizeof(maxLocalMemSize), &maxLocalMemSize, NULL);
    ALOGD("局部内存 (一个工作组共享局部内存): %" PRIu64 "(KB)\n", maxLocalMemSize/1024);

    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
    ALOGD("常量空间: %" PRIu64 "(KB)\n", maxConstantBufferSize/1024); // 一个opencl设备的常量空间是有限制的

    // 设备支持扩展
    size_t ext_size = 0 ;
    clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, 0, NULL, &ext_size); ALOGD("OpenCL扩展串长度 %zd ", ext_size );
    char* ext_data = (char*)malloc(ext_size);memset(ext_data,0,ext_size );
    clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, ext_size ,ext_data, NULL);
    ALOGD("OpenCL扩展 %s" , ext_data );
    free(ext_data);

    // Profile版本信息
    clGetDeviceInfo(devices[0], CL_DEVICE_PROFILE, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);memset(value,0,valueSize );
    clGetDeviceInfo(devices[0], CL_DEVICE_PROFILE , valueSize, value, NULL);
    ALOGD("Profile %s " , value);
    free(value);
    clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize); memset(value,0,valueSize );
    clGetDeviceInfo(devices[0], CL_DEVICE_VERSION , valueSize, value, NULL);
    ALOGD("Version %s " , value );
    free(value);

    // 设备名字
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);memset(value,0,valueSize );
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, valueSize, value, NULL);
    jstring ret = env->NewStringUTF(value);
    ALOGD ("Device Name: %s\n", value);
    free(value);
    return ret ;

}


/*
   820:
           platform 0 设备类型 DEFAULT 设备数目 1
           platform 0 设备类型 CPU 没有找到
           Current Valid Type GPU
           platform 0 设备类型 GPU 设备数目 1
           platform 0 设备类型 ACCELERATOR 没有找到
           platform 0 device_type 无效值 (CUSTOM)
           platform 0 设备类型 ALL 设备数目 1
           Platform's Device Type GPU [4]  Num Of Device [1]   --- 晓龙820和6797都只有一个平台和旗下一个设备:GPU

           Parallel compute units: 4                           --- 计算单元有4个
           Max Work Item Dimension : 3                         --- 索引表维度数目3个 X Y Z
           Max Work Item per Group: 1024 (on a single compute unit)
           Max Work Item Per Group: [1024,1024,1024]           --- 每个工作组工作项在各个维度X Y Z 上的数目
           maxGlobalMemSize: 1347 (MB)
           maxLocalMemSize: 32(KB)
           maxConstantBufferSize: 64(KB)                       --- 常量空间
           platform extra data max size 596
           extensions  cl_khr_3d_image_writes          cl_img_egl_image    -- 支持的扩展
                       cl_khr_byte_addressable_store   cl_khr_depth_images
                       cl_khr_egl_event                cl_khr_egl_image
                       cl_khr_fp16                     cl_khr_gl_sharing
                       cl_khr_global_int32_base_atomics    cl_khr_global_int32_extended_atomics
                       cl_khr_local_int32_base_atomics     cl_khr_local_int32_extended_atomics
                       cl_khr_image2d_from_buffer
                       cl_khr_mipmap_image
                       cl_khr_srgb_image_writes cl_khr_subgroups
                       cl_qcom_create_buffer_from_image           --- 可以跟Texture关联???
                       cl_qcom_ext_host_ptr
                       cl_qcom_ion_host_ptr                         ---- 只是 ION ??
                       cl_qcom_perf_hint
                       cl_qcom_read_image_2x2
                       cl_qcom_android_native_buffer_host_ptr      --- 可以跟GraphicBuffer关联 ????
                       cl_qcom_compressed_yuv_image_read
                       cl_qcom_compressed_image
           profile FULL_PROFILE
           version OpenCL 2.0 Adreno(TM) 530                       --- 支持 OpenCl 2.0 Full Profile
           Device Name: QUALCOMM Adreno(TM)

   6797:
            platform 0 设备类型 DEFAULT 设备数目 1
            platform 0 设备类型 CPU 没有找到
            Current Valid Type GPU
            platform 0 设备类型 GPU 设备数目 1
            platform 0 设备类型 ACCELERATOR 没有找到
            platform 0 device_type 无效值 (CUSTOM)
            platform 0 设备类型 ALL 设备数目 1
            Platform's Device Type GPU [4]  Num Of Device [1]

            并行计算单元: 4
            最大维度数目 : 3
            工作组最大工作项 256
            每个工作组各个维度最大大小: [256,256,256]
            全局内存: 3823 (MB)
            局部内存 (一个工作组共享局部内存): 32(KB)
            常量空间: 64(KB)
            OpenCL扩展串长度 444
            OpenCL扩展
                cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics
                cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics
                cl_khr_byte_addressable_store
                cl_khr_3d_image_writes
                cl_khr_fp64
                cl_khr_int64_base_atomics cl_khr_int64_extended_atomics
                cl_khr_fp16
                cl_khr_gl_sharing
                cl_khr_icd
                cl_khr_egl_event
                cl_khr_egl_image                                            --- ???可以跟EGLImageKHR关联 ????
                cl_arm_core_id
                cl_arm_printf                                               --- 可以在host打印 ????
                cl_arm_thread_limit_hint
                cl_arm_non_uniform_work_group_size
                cl_arm_import_memory
            Profile FULL_PROFILE
            Version OpenCL 1.1 v1.r7p0-02rel0.f2c69255b7319de8a90fdb262ee294bb
            Device Name: Mali-T880
*/
