
#ifndef MYOCLBILATERAL_H
#define MYOCLBILATERAL_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "CL/cl.hpp"


#include "jni_bilateral.h"
void openCLNR (unsigned char* bufIn, unsigned char* bufOut, int* info);


#endif
