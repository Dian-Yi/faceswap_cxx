#pragma once
#include "FaceSwap.h"

/******************************************************/
/*  1. convert float keypoints to MatrixXd          
 *  2. select kps according to mode3D.idx2d */
Eigen::MatrixXd getKpt(const mode3D_info *mode3D, float* keypoints);

/******************************************************/
/* select mean3Dshape, blendshaps according to idx3d  */
void getFuncArgs(const mode3D_info *mode3D, funArgs *mode3D_args);
void getInitialParameters(const funArgs *args, Eigen::Vector<double, 20> *params);
Eigen::MatrixXd fun(const funArgs *args, const Eigen::Vector<double, 20> *params);
Eigen::Matrix3d get_shape3D_comm(const funArgs *args, const Eigen::Vector<double, 20> *params, Eigen::MatrixXd *shape3D);
Eigen::MatrixXd jacobian(const Eigen::Vector<double, 20> *params, const funArgs *args, const OrthographicParams *ograpParams);
Eigen::VectorXd residual(const Eigen::Vector<double, 20> *params, const funArgs *args);
Eigen::MatrixXd get_shape3d(const funArgs* margs, Eigen::Vector<double, 20> *model_params);