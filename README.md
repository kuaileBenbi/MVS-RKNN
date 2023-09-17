# 简介
* 此仓库为c++实现, 实时目标检测使用yolov5网络，目标检测代码改自：https://github.com/leafqycc/rknn-cpp-Multithreading
* 所用摄像头为海康hikvision千兆以太网口工业相机 MV-CA020-10GM：200万像素 CMOS 黑白 GigE

# 更新说明
* 无

# 使用说明
### 演示
  * 系统需安装有**OpenCV** **海康工业相机SDK**
  * 可切换至root用户运行performance.sh定频提高性能和稳定性
  * 编译完成后进入install运行命令./rknn_yolov5_demo **模型所在路径**

### 部署应用
  * 修改include/rknnPool.hpp中的rknn_lite类
  * 修改inclue/rknnPool.hpp中的rknnPool类的构造函数

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型来源: 
* [yolov5s_relu_tk2_RK3588_i8.rknn](https://github.com/airockchip/rknn_model_zoo/tree/main/models)


# Acknowledgements
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo
* https://github.com/rockchip-linux/rknn-toolkit2
