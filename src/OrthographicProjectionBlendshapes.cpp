#include "OrthographicProjectionBlendshapes.h"

Eigen::MatrixXd getKpt(const mode3D_info *mode3D, float* keypoints)
{
	int i;
	int cols_3d = (mode3D->idxs_3d).cols();
	int cols_2d = (mode3D->idxs_2d).cols();
	assert(cols_3d == cols_2d);
	Eigen::MatrixXd kps(2, cols_2d);

	for (i = 0; i < cols_2d; ++i) {
		int j_2d = (mode3D->idxs_2d)(0, i);

		kps(0, i) = keypoints[j_2d * 2];
		kps(1, i) = keypoints[j_2d * 2 + 1];
	}

	return kps;
}


void getFuncArgs(const mode3D_info *mode3D, funArgs *mode3D_args)
{
	int i, j;
	int rows_3d = (mode3D->mean_3d_shape).rows();
	int cols_3d = (mode3D->idxs_3d).cols();
	int cols_2d = (mode3D->idxs_2d).cols();
	assert(cols_3d == cols_2d);

	Eigen::MatrixXd mean3DShape(rows_3d, cols_3d);
	assert(rows_3d == 3);

	for (i = 0; i < cols_2d; ++i) {
		int j_3d = (mode3D->idxs_3d)(0, i);
		mean3DShape.col(i) = (mode3D->mean_3d_shape).col(j_3d);
	}
	(*mode3D_args).mean_3d_shape = mean3DShape;

	for (i = 0; i < (*mode3D).blend_shapes.size(); ++i) {
		Eigen::MatrixXd tmp(rows_3d, cols_3d);
		for (j = 0; j < cols_3d; ++j) {
			int j_3d = (mode3D->idxs_3d)(0, j);
			tmp.col(j) = (mode3D->blend_shapes[i]).col(j_3d);
		}
		mode3D_args->blend_shapes.push_back(tmp);
	}
}

void getInitialParameters(const funArgs *args, Eigen::Vector<double, 20> *params)
{
	int i;
	int rows_3d = (args->mean_3d_shape).rows();
	int cols_3d = (args->mean_3d_shape).cols();

	Eigen::MatrixXd kps = (*args).y.transpose();
	Eigen::MatrixXd mean3DShape = (args->mean_3d_shape).transpose();

	Eigen::MatrixXd shape3DCentered = mean3DShape;
	Eigen::MatrixXd shape2DCentered = kps;

	Eigen::MatrixXd shape3DMean_col = shape3DCentered.colwise().mean();
	Eigen::MatrixXd shape2DMean_col = shape2DCentered.colwise().mean();

	for (i = 0; i < shape3DCentered.rows(); ++i) {
		shape3DCentered.row(i) -= shape3DMean_col;
		shape2DCentered.row(i) -= shape2DMean_col;
	}
	Eigen::MatrixXd shape3D_2d(shape3DCentered.rows(), 2);
	shape3D_2d = shape3DCentered.block(0, 0, cols_3d, 2);
	
	double scale = shape2DCentered.norm() / shape3D_2d.norm();
	shape3D_2d = mean3DShape.block(0, 0, cols_3d, 2);
	shape3DMean_col = shape3D_2d.colwise().mean();
	Eigen::MatrixXd t = shape2DMean_col - shape3DMean_col;

	(*params).setZero();
	(*params)(0) = scale;
	(*params)(4) = t(0,0);
	(*params)(5) = t(0,1);
}

Eigen::Matrix3d get_shape3D_comm(const funArgs *args, const Eigen::Vector<double, 20> *params, Eigen::MatrixXd *shape3D)
{
	Eigen::Matrix3d R;
	int i, j;
	double s = (*params)(0);

	Eigen::Vector<double, 14> w = (*params).segment(6, 14);
	cv::Mat f = (cv::Mat_<double>(1, 3) << (*params)(1), (*params)(2), (*params)(3));
	cv::Mat dst;
	cv::Rodrigues(f, dst);
	cv2eigen(dst, R);
	
	Eigen::MatrixXd tmp(3, args->blend_shapes[0].cols());
	std::vector<Eigen::MatrixXd> blend_shapes;
	for (i = 0; i < 14; ++i)
	{
		Eigen::MatrixXd tmp = args->blend_shapes[i] * w(i);
		blend_shapes.push_back(tmp);
	}
	for (i = 0; i < (*shape3D).cols(); ++i)
	{
		for (j = 0; j < 14; ++j)
		{
			(*shape3D).col(i) += blend_shapes[j].col(i);
		}
	}
	(*shape3D) = (*shape3D) + args->mean_3d_shape;
	return R;
}

Eigen::MatrixXd fun(const funArgs *args, const Eigen::Vector<double, 20> *params)
{
	double s = (*params)(0);
	Eigen::Vector<double, 2> t = (*params).segment(4, 2);

	int cols = (*args).mean_3d_shape.cols();
	int rows = (*args).mean_3d_shape.rows();
	Eigen::MatrixXd shape3d = Eigen::MatrixXd::Zero(rows, cols);
	Eigen::Matrix3d R = get_shape3D_comm(args, params, &shape3d);
	Eigen::MatrixXd P = R.block(0, 0, 2, 3);
	Eigen::MatrixXd projected;
	projected = s * (P * shape3d);

	for (int i = 0; i < projected.cols(); ++i) projected.col(i) += t;

	return projected;
}

Eigen::MatrixXd jacobian(const Eigen::Vector<double, 20> *params, const funArgs *args, const OrthographicParams *ograpParams)
{
	int shape3d_cols = (*args).mean_3d_shape.cols();
	int shape3d_rows = (*args).mean_3d_shape.rows();
	Eigen::MatrixXd shape3d = Eigen::MatrixXd::Zero(shape3d_rows, shape3d_cols);
	Eigen::Matrix3d R = get_shape3D_comm(args, params, &shape3d);
	Eigen::MatrixXd P = R.block(0, 0, 2, 3);

	int nPoints = shape3d.cols();
	Eigen::MatrixXd jacob = Eigen::MatrixXd::Zero(nPoints *2, ograpParams->nParams);
	double stepSize = 10e-4;
	Eigen::Vector<double, 20> step = Eigen::VectorXd::Zero(20);
	Eigen::MatrixXd projected1, projected2, tmp;

	projected1 = P * shape3d;
	projected2 = fun(args, params);
	for (int i = 0; i < jacob.rows(); ++i)
		jacob(i, 0) = projected1(i/nPoints, i % nPoints);

	for (int i = 1; i < 4; ++i) {
		step.setZero();
		step(i) = stepSize;
		step = step + (*params);
		projected1 = fun(args, &step);
		tmp = (projected1 - projected2) * (1.0 / stepSize);

		for (int j = 0; j < jacob.rows(); ++j)
			jacob(j, i) = tmp(j / nPoints, j % nPoints);
	}
	
	for (int i = 0; i < nPoints; ++i) jacob(i, 4) = 1.0;
	for (int i= nPoints; i< jacob.rows(); ++i) jacob(i, 5) = 1.0;

	int startIdx = (*ograpParams).nParams - (*ograpParams).nBlendshapes;
	double s = (*params)(0);
	for (int i = 0; i < (*ograpParams).nBlendshapes; ++i) {
		projected1 = s * (P * ((*args).blend_shapes[i]));
		for (int j = 0; j < jacob.rows(); ++j)
			jacob(j, i + startIdx) = projected1(j / nPoints, j % nPoints);
	}
	return jacob;
}

Eigen::VectorXd residual(const Eigen::Vector<double, 20> *params, const funArgs *args)
{
	Eigen::MatrixXd r = (*args).y - fun(args, params);
	r.transposeInPlace();
	Eigen::VectorXd vr(Eigen::Map<Eigen::VectorXd>(r.data(), r.cols()*r.rows()));
	return vr;
}

Eigen::MatrixXd get_shape3d(const funArgs* margs, Eigen::Vector<double, 20> *model_params)
{
	double s = (*model_params)(0);
	Eigen::Vector<double, 2> t = (*model_params).segment(4, 2);

	int cols = (*margs).mean_3d_shape.cols();
	int rows = (*margs).mean_3d_shape.rows();
	Eigen::MatrixXd shape3d = Eigen::MatrixXd::Zero(rows, cols);
	Eigen::Matrix3d R = get_shape3D_comm(margs, model_params, &shape3d);
	shape3d = s * (R * shape3d);
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 113; ++j)
		{
			shape3d(i, j) += t(i);
		}
	}
	return shape3d;
}