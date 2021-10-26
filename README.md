# faceswap_cxx
3D FaceSwap, Using cpu realtime
# realtime face swap using cpu with 3D model

## Introduction
- c++版使用cpu实时换脸，参考git: https://github.com/MarekKowalski/FaceSwap
- 换脸逻辑
  - 初始化: 根据上传的src图片初始化人脸3D模型参数
  - 检测识别: 摄像头读取每帧图，检测dst的脸和landmark位置
  - 人脸关键点滤波: oneEuroFilter算法
  - 计算3D模型顶点位置: 使用GaussNewton算法计算相关参数(共20个参数:0-5为缩放，平移，旋转相关, 6-20为shape相关)
  - 人脸融合贴回视频帧

## Directory structure
- 外部依赖项：                    ./dep/
- 3D 画图依赖相关：               ./deps, ./Eigen, ./GL, ./GLFW, ./zlib
- 人脸检测，landmarks检测依赖相关：./onnxruntime
- ./lib       依赖lib文件： Debug和Release
- ./dll       依赖dll文件： Debug和Release
- ./include   项目头文件目录
- ./src       项目cpp文件目录
- ./resource  人脸3D模型，网络模型等需要的资源路径

## Note
- 本项目使用vs2019编译能成功， 如果需要更换编译器版本，遇到错误可能需要重新编译./lib中的依赖相关文件
- debug 模式下运行时会出现"__acrt_first_block == header"， 由opencv中的convexHull（）函数引起。具体原因是opencv使用MD编译， 本项目使用MT编译
- debug模式下， 使用 #include "vld.h" 对本项目进行内存泄漏检测。如果检测结果没有具体定位位置：需要设置Linker 下 Generate Debug Info 选为(/DEBUG:FULL)。\
注意一定要程序运行完退出时，才会在vs Output中显示内存检测信息
