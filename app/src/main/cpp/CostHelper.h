//
// Created by rd0394 on 2017/3/27.
//

#ifndef BQLIMAGE_COSTHELPER_H
#define BQLIMAGE_COSTHELPER_H

#define __STDC_FORMAT_MACROS

#include <cstdint>
#include <vector>
#include <inttypes.h>
#include <time.h>


inline int64_t GetTickCount(){  // us  系统开机 不包含休眠时间
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now); // 不包含休眠时间,相当于Java层的SystemClock.uptimeMillis();
    return (int64_t)now.tv_sec * 1000000 + now.tv_nsec / 1000;
}

inline int64_t GetTimeSinceBoot(){ // us  系统开机 包含休眠时间
    struct timespec now;
    clock_gettime(CLOCK_BOOTTIME, &now);
    return (int64_t)now.tv_sec * 1000000 + now.tv_nsec / 1000;

}

inline int64_t GetCurrentTimeMillis() // ms
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000LL + tv.tv_usec/1000LL ;
};

inline int64_t GetCurrentTimeUs() // us
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000LL*1000LL + tv.tv_usec  ;
};

class CostHelper {
public:
    CostHelper()  { this->ref = GetTickCount();}

public: // us
    int64_t Get() const { return GetTickCount() - this->ref;}

private:
    int64_t ref;
};






#endif //BQLIMAGE_COSTHELPER_H
