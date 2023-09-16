// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <queue>
#include <vector>
#define _BASETSD_H

#include "MvCameraControl.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rknnPool.hpp"
#include "ThreadPool.hpp"

/* MVS */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>

bool g_bExit = false;
/* === */

using std::queue;
using std::time;
using std::time_t;
using std::vector;

char *model_name = NULL;
// 设置线程数
int n = 12, frames = 0;
// 类似于多个rk模型的集合?
vector<rknn_lite *> rkpool;
// 线程池
dpool::ThreadPool pool(n);
// 线程队列
queue<std::future<int>> futs;

void PressEnterToExit(void)
{
    int c;
    while ((c = getchar()) != '\n' && c != EOF)
        ;
    fprintf(stderr, "\nPress enter to exit.\n");
    while (getchar() != '\n')
        ;
    g_bExit = true;
    sleep(1);
}

bool PrintDeviceInfo(MV_CC_DEVICE_INFO *pstMVDevInfo)
{
    if (NULL == pstMVDevInfo)
    {
        printf("The Pointer of pstMVDevInfo is NULL!\n");
        return false;
    }
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
    {
        int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
        int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
        int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
        int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

        // ch:打印当前相机ip和用户自定义名字 | en:print current ip and user defined name
        printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
        printf("CurrentIp: %d.%d.%d.%d\n", nIp1, nIp2, nIp3, nIp4);
        printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
    }
    else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
    {
        printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chModelName);
        printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
    }
    else
    {
        printf("Not support.\n");
    }

    return true;
}

static void *WorkThread(void *pUser)
{
    int nRet = MV_OK;

    // ch:获取数据包大小 | en:Get payload size
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    nRet = MV_CC_GetIntValue(pUser, "PayloadSize", &stParam);
    if (MV_OK != nRet)
    {
        printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
        return NULL;
    }

    MV_FRAME_OUT_INFO_EX stImageInfo = {0};
    memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
    unsigned char *pData = (unsigned char *)malloc(sizeof(unsigned char) * stParam.nCurValue);
    if (NULL == pData)
    {
        return NULL;
    }
    unsigned int nDataSize = stParam.nCurValue;

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto initTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    gettimeofday(&time, nullptr);
    long tmpTime, lopTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    cv::Size newSize(256, 256);
    int initFrames = 0;
    while (1)
    {
        if (g_bExit)
        {
            break;
        }
        nRet = MV_CC_GetOneFrameTimeout(pUser, pData, nDataSize, &stImageInfo, 1000);
        if (nRet == MV_OK)
        {
            // printf("GetOneFrame, Width[%d], Height[%d], nFrameNum[%d]\n",
            //        stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);

            int type = (stImageInfo.enPixelType == PixelType_Gvsp_RGB8_Packed) ? CV_8UC3 : CV_8UC1;

            cv::Mat srcImage(stImageInfo.nHeight, stImageInfo.nWidth, type, pData);

            // resize图片尺寸
            cv::Mat resizedImage;
            cv::resize(srcImage, resizedImage, newSize);
            // 使用resizedImage

            // 如果需要，可以显示图像
            // cv::imshow("Image", resizedImage);
            // cv::waitKey(30);

            if (initFrames < n)
            {
                rknn_lite *ptr = new rknn_lite(model_name, initFrames % 3);
                rkpool.push_back(ptr);
                ptr->ori_img = resizedImage;
                futs.push(pool.submit(&rknn_lite::interf, &(*ptr)));
                initFrames += 1;
                continue;
            }

            if (futs.front().get() != 0)
                break;
            futs.pop();
            cv::imshow("Camera FPS", rkpool[frames % n]->ori_img);
            if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出
                break;
            // if (!capture.read(rkpool[frames % n]->ori_img))
            //     break;
            rkpool[frames % n]->ori_img = resizedImage;

            futs.push(pool.submit(&rknn_lite::interf, &(*rkpool[frames++ % n])));

            if (frames % 60 == 0)
            {
                gettimeofday(&time, nullptr);
                tmpTime = time.tv_sec * 1000 + time.tv_usec / 1000;
                printf("60帧平均帧率:\t%f帧\n", 60000.0 / (float)(tmpTime - lopTime));
                lopTime = tmpTime;
            }

            // rkpool[frames % n]->ori_img = resizedImage;
            // futs.push(pool.submit(&rknn_lite::interf, &(*rkpool[frames++ % n])));

            // if (frames % 60 == 0)
            // {
            //     gettimeofday(&time, nullptr);
            //     tmpTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            //     printf("60帧平均帧率:\t%f帧\n", 60000.0 / (float)(tmpTime - lopTime));
            //     lopTime = tmpTime;
            // }
        }
        else
        {
            printf("No data[%x]\n", nRet);
        }
    }

    free(pData);

    gettimeofday(&time, nullptr);
    printf("\n平均帧率:\t%f帧\n", float(frames) / (float)(time.tv_sec * 1000 + time.tv_usec / 1000 - initTime + 0.0001) * 1000.0);

    // 释放剩下的资源
    while (!futs.empty())
    {
        if (futs.front().get())
            break;
        futs.pop();
    }
    for (int i = 0; i < n; i++)
        delete rkpool[i];
    // capture.release();
    cv::destroyAllWindows();

    return 0;
}

int main(int argc, char **argv)
{
    /* 模型加载 */
    model_name = (char *)argv[1]; // 参数二，模型所在路径
    printf("模型名称:\t%s\n", model_name);
    printf("线程数:\t%d\n", n);

    // 初始化
    // for (int i = 0; i < n; i++)
    // {
    //     rknn_lite *ptr = new rknn_lite(model_name, i % 3);
    //     rkpool.push_back(ptr);
    //     // capture >> ptr->ori_img;
    //     futs.push(pool.submit(&rknn_lite::interf, &(*ptr)));
    // }

    /* 摄像头初始化 */
    int nRet = MV_OK;
    void *handle = NULL;
    do
    {
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

        // 枚举设备
        // enum device
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet)
        {
            printf("MV_CC_EnumDevices fail! nRet [%x]\n", nRet);
            break;
        }

        if (stDeviceList.nDeviceNum > 0)
        {
            for (int i = 0; i < stDeviceList.nDeviceNum; i++)
            {
                printf("[device %d]:\n", i);
                MV_CC_DEVICE_INFO *pDeviceInfo = stDeviceList.pDeviceInfo[i];
                if (NULL == pDeviceInfo)
                {
                    break;
                }
                PrintDeviceInfo(pDeviceInfo);
            }
        }
        else
        {
            printf("Find No Devices!\n");
            break;
        }

        printf("Please Intput camera index: ");
        unsigned int nIndex = 0;
        scanf("%d", &nIndex);

        if (nIndex >= stDeviceList.nDeviceNum)
        {
            printf("Intput error!\n");
            break;
        }

        // 选择设备并创建句柄
        // select device and create handle
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
        if (MV_OK != nRet)
        {
            printf("MV_CC_CreateHandle fail! nRet [%x]\n", nRet);
            break;
        }

        // 打开设备
        // open device
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet)
        {
            printf("MV_CC_OpenDevice fail! nRet [%x]\n", nRet);
            break;
        }

        // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if (stDeviceList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
        {
            int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
            if (nPacketSize > 0)
            {
                nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
                if (nRet != MV_OK)
                {
                    printf("Warning: Set Packet Size fail nRet [0x%x]!\n", nRet);
                }
            }
            else
            {
                printf("Warning: Get Packet Size fail nRet [0x%x]!\n", nPacketSize);
            }
        }

        // 设置触发模式为off
        // set trigger mode as off
        nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
        if (MV_OK != nRet)
        {
            printf("MV_CC_SetTriggerMode fail! nRet [%x]\n", nRet);
            break;
        }

        // 开始取流
        // start grab image
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
            break;
        }

        pthread_t nThreadID;

        nRet = pthread_create(&nThreadID, NULL, WorkThread, handle);
        if (nRet != 0)
        {
            printf("thread create failed.ret = %d\n", nRet);
            break;
        }

        PressEnterToExit();

        // 停止取流
        // end grab image
        nRet = MV_CC_StopGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("MV_CC_StopGrabbing fail! nRet [%x]\n", nRet);
            break;
        }

        // 关闭设备
        // close device
        nRet = MV_CC_CloseDevice(handle);
        if (MV_OK != nRet)
        {
            printf("MV_CC_CloseDevice fail! nRet [%x]\n", nRet);
            break;
        }

        // 销毁句柄
        // destroy handle
        nRet = MV_CC_DestroyHandle(handle);
        if (MV_OK != nRet)
        {
            printf("MV_CC_DestroyHandle fail! nRet [%x]\n", nRet);
            break;
        }

    } while (0);

    /* 之前的代码
    嘻
    嘻
    嘻
    嘻
    */

    return 0;

    // char *model_name = NULL;
    // if (argc != 3)
    // {
    //     printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
    //     return -1;
    // }
    // model_name = (char *)argv[1]; // 参数二，模型所在路径
    // char *image_name = argv[2];   // 参数三, 视频/摄像头
    // printf("模型名称:\t%s\n", model_name);

    // cv::VideoCapture capture;
    // cv::namedWindow("Hikvision Camera");
    // if (strlen(image_name) == 1)
    //     // capture.open((int)(image_name[0] - '0'));
    //     capture.open('rtsp://admin:admin@169.254.46.208/ch1/main/av_stream');
    // // gst-launch-1.0 -vv udpsrc uri=udp://239.192.1.1:1042 caps="video/mpegts" ! queue2 use-buffering=true max-size-buffers=1000 ! tsparse  ! decodebin ! videoconvert ! videoscale ! xvimagesink
    // else
    //     capture.open(image_name);

    // // 设置线程数
    // int n = 12, frames = 0;
    // printf("线程数:\t%d\n", n);
    // // 类似于多个rk模型的集合?
    // vector<rknn_lite *> rkpool;
    // // 线程池
    // dpool::ThreadPool pool(n);
    // // 线程队列
    // queue<std::future<int>> futs;

    // // 初始化
    // for (int i = 0; i < n; i++)
    // {
    //     rknn_lite *ptr = new rknn_lite(model_name, i % 3);
    //     rkpool.push_back(ptr);
    //     capture >> ptr->ori_img;
    //     futs.push(pool.submit(&rknn_lite::interf, &(*ptr)));
    // }

    // struct timeval time;
    // gettimeofday(&time, nullptr);
    // auto initTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    // gettimeofday(&time, nullptr);
    // long tmpTime, lopTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    // while (capture.isOpened())
    // {
    //     if (futs.front().get() != 0)
    //         break;
    //     futs.pop();
    //     cv::imshow("Camera FPS", rkpool[frames % n]->ori_img);
    //     if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出
    //         break;
    //     if (!capture.read(rkpool[frames % n]->ori_img))
    //         break;
    //     futs.push(pool.submit(&rknn_lite::interf, &(*rkpool[frames++ % n])));

    //     if (frames % 60 == 0)
    //     {
    //         gettimeofday(&time, nullptr);
    //         tmpTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    //         printf("60帧平均帧率:\t%f帧\n", 60000.0 / (float)(tmpTime - lopTime));
    //         lopTime = tmpTime;
    //     }
    // }

    // gettimeofday(&time, nullptr);
    // printf("\n平均帧率:\t%f帧\n", float(frames) / (float)(time.tv_sec * 1000 + time.tv_usec / 1000 - initTime + 0.0001) * 1000.0);

    // // 释放剩下的资源
    // while (!futs.empty())
    // {
    //     if (futs.front().get())
    //         break;
    //     futs.pop();
    // }
    // for (int i = 0; i < n; i++)
    //     delete rkpool[i];
    // capture.release();
    // cv::destroyAllWindows();
    // return 0;
}
