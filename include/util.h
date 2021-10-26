#pragma once
#include <Eigen/Dense>
#include <cnpy.h>
#include <chrono>
#include "FaceSwap.h"
using namespace std::chrono;

void npyarry2matrixi(cnpy::npz_t* src, std::string name, Eigen::MatrixXi* matrix);
void npyarry2matrixd(cnpy::npz_t* src, std::string name, Eigen::MatrixXd* matrix);
void npyarry2matrix3D(cnpy::npz_t* src, std::string name, std::vector<Eigen::MatrixXd> &blend_shapes);
void faceSwapByThree(unsigned char* im1, unsigned char * im2, int width, int height);

/****************************/
/* mode = 0:                */
/* mode = 1: flip mode      */
std::vector<float> track_box(std::vector<std::vector<float>> &boxes, std::vector<float> &hist, int mode);
void get_roi(int *box_roi, int *render_roi, float *box, int height, int width);
//void read_lmks(const char *file, float *lmks);
bool read_lmks(const char* file, std::vector<float>& lmks);

/**************************************************************/
/*                 add gloab params hull                      */
/*  just for debug assert info "__acrt_first_block == header" */
/*  because of opencv build MD and this project build MT      */
void faceBlend(cv::Mat &src, cv::Mat &dst, std::vector<cv::Point> &hull, bool colorTranf=true, float featherAmount=0.2, float alpha = 0.5);
unsigned long long ms_since_epoch();