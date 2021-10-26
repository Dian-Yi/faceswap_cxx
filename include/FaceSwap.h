#pragma once
#include <Eigen/Dense>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

typedef struct {
	char path[128];
	Eigen::MatrixXd mean_3d_shape;
	Eigen::MatrixXi mesh;
	Eigen::MatrixXi idxs_3d;
	Eigen::MatrixXi idxs_2d;
	std::vector<Eigen::MatrixXd> blend_shapes;
}mode3D_info;

typedef struct {
	//int nBlendshapes = 0;
	//int nParams = 6;
	int nBlendshapes;
	int nParams;
}OrthographicParams;

typedef struct {
	Eigen::MatrixXd mean_3d_shape;
	std::vector<Eigen::MatrixXd> blend_shapes;
	Eigen::MatrixXd y;
}funArgs;

typedef struct {
	std::vector<std::vector<float>> hisBoxes;
	// cv::Mat hisImage;
	// float his_lmks[136] = { 0 };
	float his_lmks[136];
} hisInfo;