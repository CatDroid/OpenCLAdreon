//
// Created by hl.he on 2017/8/2.
//

#include "jni_cl_common.h"
#include "common.h"
/**
 * \brief 用随机数初始化矩阵
 * \param[in] matrixOrder 矩阵的阶  比如 1024*1024 matrixOrder=1024
 * \param[in] matrixA First input matrix.
 * \param[in] matrixB Second input matrix.
 * \param[in] matrixC Third input matrix.
 * \return matrixA, matrixB and matrixC with random values.
 */
void sgemmInitialize (int matrixOrder, float* matrixA, float* matrixB, float * matrixC)
{
    for (int i = 0; i < matrixOrder; i++)
    {
        for (int j = 0; j < matrixOrder; j++)
        {
            int index = i * matrixOrder + j;

            /* Keep the values in the range [-1, 1]. */
            float randomeNumber = rand() / (float) RAND_MAX * 2 - 1;
            matrixA[index] = randomeNumber;

            randomeNumber = rand() / (float) RAND_MAX * 2 - 1;
            matrixB[index] = randomeNumber;

            randomeNumber = rand() / (float) RAND_MAX * 2 - 1;
            matrixC[index] = randomeNumber;
        }
    }
}

#define DUMP_TO_FILE 0

/**
 * \brief Simple SGEMM OpenCL sample.
 * \details A sample which calculates the following SGEMM equation:
 * matrixC = alpha * (matrixA * matrixB) + beta * matrixC.
 *
 * \return The exit code of the application, non-zero if a problem occurred.
 */
//int main(void)
extern "C"
bool Java_com_tom_opencladreon_RunKernelActivity_nativeRunSGEMM(JNIEnv* env, jclass clazz)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    const unsigned int numberOfMemoryObjects = 3;
    cl_mem memoryObjects[numberOfMemoryObjects] = {0, 0, 0};
    cl_int errorNumber;

    if (!createContext(&context))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE( "Failed to create an OpenCL context. " );
        return false;
    }

    if (!createCommandQueue(context, &commandQueue, &device))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE( "Failed to create the OpenCL command queue. " );
        return false;
    }

    if (!createProgram(context, device, "/data/data/com.tom.opencladreon/app_opencl_dir/sgemm.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Failed to create OpenCL program." );
        return false;
    }

    kernel = clCreateKernel(program, "sgemm", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Failed to create OpenCL kernel. " );
        return false;
    }

    // kernel的参数
    unsigned int matrixOrder = 16 ;
    float alpha = 1;
    float beta = 0.1;

    /* Create the matrices. */
    const size_t matrixSize = matrixOrder * matrixOrder;

    /* As all the matrices have the same size, the buffer size is common. */
    size_t bufferSize = matrixSize * sizeof(float);

    /* Create buffers for the matrices used in the kernel. */
    bool createMemoryObjectsSuccess = true;
    memoryObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    if (!createMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Failed to create OpenCL buffers. " );
        return false;
    }

    /* Map the input memory objects to a host side pointers. */
    bool mapMemoryObjectsSuccess = true;
    cl_float* matrixA = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    cl_float* matrixB = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    cl_float* matrixC = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[2], CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    if (!mapMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Mapping memory objects failed " );
        return false;
    }

    /* Fill the matrices with random data. */
    sgemmInitialize(matrixOrder, matrixA, matrixB, matrixC);

#if DUMP_TO_FILE == 1

    FILE* fp = fopen("/mnt/sdcard/sgemm.txt","w+");
    for(int i = 0 ; i <  matrixOrder ; i++ ){
        for(int j = 0 ; j <  matrixOrder ; j+=4 ) {
            fprintf(fp, "%10f,%10f,%10f,%10f," ,
                      matrixA[i*matrixOrder + j ],  matrixA[i*matrixOrder + j + 1  ],
                      matrixA[i*matrixOrder + j + 2 ],  matrixA[i*matrixOrder + j + 3  ] );
        }
        fseek(fp,-1,SEEK_CUR); // clear ,
        fprintf(fp,";\n");
    }
    fprintf(fp,"\n");
    for(int i = 0 ; i <  matrixOrder ; i++ ){
        for(int j = 0 ; j <  matrixOrder ; j+=4 ) {
            fprintf(fp, "%10f,%10f,%10f,%10f," ,
                    matrixB[i*matrixOrder + j ],  matrixB[i*matrixOrder + j + 1  ],
                    matrixB[i*matrixOrder + j + 2 ],  matrixB[i*matrixOrder + j + 3 ] );
        }
        fseek(fp,-1,SEEK_CUR); // clear ,
        fprintf(fp,";\n");
    }

    fprintf(fp,"\n");
    for(int i = 0 ; i <  matrixOrder ; i++ ){
        for(int j = 0 ; j <  matrixOrder ; j+=4 ) {
            fprintf(fp, "%10f,%10f,%10f,%10f," ,
                    matrixC[i*matrixOrder + j ],  matrixC[i*matrixOrder + j + 1  ],
                    matrixC[i*matrixOrder + j + 2 ],  matrixC[i*matrixOrder + j + 3 ] );
        }
        fseek(fp,-1,SEEK_CUR); // clear ,
        fprintf(fp,";\n");
    }

#endif


    /* Unmap the memory so we can pass it to the kernel. */
    bool unmapMemoryObjectsSuccess = true;
    unmapMemoryObjectsSuccess &= checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], matrixA, 0, NULL, NULL));
    unmapMemoryObjectsSuccess &= checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[1], matrixB, 0, NULL, NULL));
    unmapMemoryObjectsSuccess &= checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[2], matrixC, 0, NULL, NULL));
    if (!unmapMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Unmapping memory objects failed " );
        return false;
    }

    /* Setup kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(cl_mem), &memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(cl_mem), &memoryObjects[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 3, sizeof(cl_uint), &matrixOrder));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 4, sizeof(cl_float), &alpha));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 5, sizeof(cl_float), &beta));
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Failed setting OpenCL kernel arguments. " );
        return false;
    }

    /* An event to associate with the Kernel. Allows us to retrieve profiling information later. */
    cl_event event = 0;

    /* [Kernel size] */
    /*
     * Each kernel outputs one element in matrixC,
     * therefore the total number of work items must be the number of elements (matrixOrder * matrixOrder).
     * To accomplish this we use a global worksize split into 2 dimensions both of matrixOrder size.
     * 每个内核输出一个元素 到 矩阵matrixC  所以创建矩阵大小==工作项数目，并把工作项分到二维上去
     * C = αAB + βC (  matrixC = alpha * (matrixA * matrixB) + beta * matrixC  )
     * 输出每一项 Cij = α∑k AikBkj + βCij
     */
    size_t globalWorksize[2] = {matrixOrder, matrixOrder};
    /* [Kernel size] */

    /* Enqueue the kernel */
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorksize, NULL, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Failed enqueuing the kernel. " );
        return false;
    }

    /* Wait for kernel execution completion */
    if (!checkSuccess(clFinish(commandQueue)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Failed waiting for kernel execution to finish. " );
        return false;
    }

    /* Print the profiling information for the event. */
    printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Failed releasing the event object. " );
        return false;
    }

    /* Map the output to a host side pointer. */
    matrixC = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[2], CL_TRUE, CL_MAP_READ, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Mapping memory objects failed " );
        return false;
    }

    /* Do something with the output (matrixC) here.  matrixC保存结果 可以在这里打印 */
#if DUMP_TO_FILE == 1
    fprintf(fp,"\n");
    for(int i = 0 ; i <  matrixOrder ; i++ ){
        for(int j = 0 ; j <  matrixOrder ; j+=4 ) {
            fprintf(fp, "%10f,%10f,%10f,%10f," ,
                    matrixC[i*matrixOrder + j ],  matrixC[i*matrixOrder + j + 1 ],
                    matrixC[i*matrixOrder + j + 2 ],  matrixC[i*matrixOrder + j + 3 ] );
        }
        fseek(fp,-1,SEEK_CUR); // clear ,
        fprintf(fp,";\n");
    }
    fclose(fp);
#endif

    /* Unmap the output. */
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[2], matrixC, 0, NULL, NULL)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        ALOGE("Unmapping memory objects failed " );
        return false;
    }

    /* Release OpenCL objects. */
    cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);

    return true ;
}