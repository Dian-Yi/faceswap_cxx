#include "util.h"
#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)

void npyarry2matrixi(cnpy::npz_t* src, std::string name, Eigen::MatrixXi* matrix)
{
	int i, j;
	cnpy::npy_array array = (*src)[name];
	int *num = array.data<int>();
	int row, column;
	if (array.shape.size() == 2)
	{
		row = array.shape[0];
		column = array.shape[1];
	}
	else if (array.shape.size() == 1)
	{
		row = 1;
		column = array.shape[0];
	}
	(*matrix) = Eigen::MatrixXi(row, column);
	for (i = 0; i < row; ++i)
	{
		for (j = 0; j < column; ++j)
		{
			(*matrix)(i, j) = num[i * column + j];
		}
	}
}

void npyarry2matrixd(cnpy::npz_t* src, std::string name, Eigen::MatrixXd* matrix)
{
	int i, j;
	cnpy::npy_array array = (*src)[name];
	double* num = array.data<double>();
	int row, column;
	if (array.shape.size() == 2)
	{
		row = array.shape[0];
		column = array.shape[1];
	}
	(*matrix) = Eigen::MatrixXd(row, column);
	for (i = 0; i < row; ++i)
	{
		for (j = 0; j < column; ++j)
		{
			(*matrix)(i, j) = num[j * row + i];
		}
	}
}

void npyarry2matrix3D(cnpy::npz_t* src, std::string name, std::vector<Eigen::MatrixXd> &blend_shapes)
{
	int i, j, k;
	cnpy::npy_array array = (*src)[name];
	double* num = array.data<double>();
	int dim1, dim2, dim3;
	dim1 = array.shape[0];
	dim2 = array.shape[1];
	dim3 = array.shape[2];
	int step = dim2 * dim3;
	for (i = 0; i < dim1; ++i) {
		Eigen::MatrixXd matrix(dim2, dim3);
		for (j = 0; j < dim2; ++j) {
			for (k = 0; k < dim3; ++k) {
				int idx = i * step + j * dim3 + k;
				matrix(j, k) = num[idx];
			}
		}
		blend_shapes.push_back(matrix);
	}
}

void colorTransfer(cv::Mat &src, cv::Mat &dst, std::vector<cv::Point> &points)
{
	int i, j;
	int shape = points.size();
	if (shape == 0)return;

	double meanDst[3] = { 0 };
	double meanSrc[3] = { 0 };
	double stdDst[3] = { 0 };
	double stdSrc[3] = { 0 };
	
	// get mean
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < shape; j++) {
			meanSrc[i] += src.at<cv::Vec3b>(points[j].x, points[j].y)[i];
			meanDst[i] += dst.at<cv::Vec3b>(points[j].x, points[j].y)[i];
		}
		meanDst[i] = meanDst[i] / shape;
		meanSrc[i] = meanSrc[i] / shape;
	}
	// get std
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < shape; ++j) {
			stdSrc[i] += pow(src.at<cv::Vec3b>(points[j].x, points[j].y)[i] - meanSrc[i], 2);
			stdDst[i] += pow(dst.at<cv::Vec3b>(points[j].x, points[j].y)[i] - meanDst[i], 2);
		}
		stdSrc[i] /= shape - 1;
		stdDst[i] /= shape - 1;
		stdSrc[i] = sqrt(stdSrc[i]);
		stdDst[i] = sqrt(stdDst[i]);
	}

	// do instance normalization
	double tmp;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < shape; j++) {
			tmp = (dst.at<cv::Vec3b>(points[j].x, points[j].y)[i] - meanDst[i]) / stdDst[i];
			tmp = tmp * stdSrc[i] + meanSrc[i];
			// tmp = (dst.at<cv::Vec3b>(points[j].x, points[j].y)[i] - meanDst[i]) + meanSrc[i];
			if (tmp > 255)tmp = 255;
			else if (tmp < 0)tmp = 0;
			dst.at<cv::Vec3b>(points[j].x, points[j].y)[i] = int(tmp);
		}
	}
}
// std::vector<cv::Point> hull;
void faceBlend(cv::Mat &src, cv::Mat &dst, std::vector<cv::Point> &hull, bool colorTranf, float featherAmount, float alpha)
{
	int width = dst.rows;
	int height = dst.cols;
	
	std::vector<cv::Point> points;
	cv::Point pt;
	int minx = 100000, miny = 100000, maxx = 0, maxy = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			if (dst.at<cv::Vec3b>(i, j)[0] > 0 || dst.at<cv::Vec3b>(i, j)[1] > 0
				|| dst.at<cv::Vec3b>(i, j)[2] > 0) {
				pt.x = i;
				pt.y = j;
				points.push_back(pt);
				minx = minx > i ? i : minx;
				miny = miny > j ? j : miny;
				maxx = maxx < i ? i : maxx;
				maxy = maxy < j ? j : maxy;
			}
		}
	}
	int shape = points.size();
	if (shape == 0) return;
	if (colorTranf) 
		colorTransfer(src, dst, points);
	
	cv::Mat mask(width, height, CV_64F, cv::Scalar(0));
	//std::vector<cv::Point> hull;
	cv::convexHull(cv::Mat(points), hull);
	if (maxx + 1 >= width)maxx -= 1;
	if (maxy + 1 >= height)maxy -= 1;
	double dis;
	featherAmount = featherAmount * max(maxx - minx, maxy - miny);
	if (featherAmount < 0)return;
	featherAmount = 1.0 / featherAmount;
	if (colorTranf) {
		for (int i = minx + 1; i < maxx; i += 3) {
			for (int j = miny + 1; j < maxy; j += 3) {
				dis = cv::pointPolygonTest(hull, cv::Point2f(i, j), true) * featherAmount;
				if (dis > 1)dis = 1;
				else if (dis < 0)dis = 0;
				for (int n = i - 1; n <= i + 1; n++) {
					for (int m = j - 1; m <= j + 1; m++) {
						mask.at<double>(n, m) = dis;
					}
				}
			}
		}
	}

	int x, y;
	for (int j = 0; j < shape; j++) {
		x = points[j].x;
		y = points[j].y;

		if (colorTranf) {
			src.at<cv::Vec3b>(x, y)[0] = dst.at<cv::Vec3b>(x, y)[0] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*src.at<cv::Vec3b>(x, y)[0];
			src.at<cv::Vec3b>(x, y)[1] = dst.at<cv::Vec3b>(x, y)[1] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*src.at<cv::Vec3b>(x, y)[1];
			src.at<cv::Vec3b>(x, y)[2] = dst.at<cv::Vec3b>(x, y)[2] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*src.at<cv::Vec3b>(x, y)[2];
		}
		else {
			src.at<cv::Vec3b>(x, y)[0] = dst.at<cv::Vec3b>(x, y)[0] * alpha + (1 - alpha)*src.at<cv::Vec3b>(x, y)[0];
			src.at<cv::Vec3b>(x, y)[1] = dst.at<cv::Vec3b>(x, y)[1] * alpha + (1 - alpha)*src.at<cv::Vec3b>(x, y)[1];
			src.at<cv::Vec3b>(x, y)[2] = dst.at<cv::Vec3b>(x, y)[2] * alpha + (1 - alpha)*src.at<cv::Vec3b>(x, y)[2];
		}
	}
	//mask.release();
	//std::vector<cv::Point>().swap(points);
	//std::vector<cv::Point>().swap(hull);
}

// old blend face by  CuiXianfeng
void faceSwapByThree(unsigned char* im1, unsigned char * im2, int width, int height) 
{
	cv::Mat out(width, height, CV_8UC3, im1);
	cv::Mat src = out;
	cv::Mat dst(width, height, CV_8UC3, im2);

	std::vector<cv::Point> points;
	cv::Point pt;
	int minx = 100000, miny = 100000, maxx = 0, maxy = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			if (dst.at<cv::Vec3b>(i, j)[0] > 0 || dst.at<cv::Vec3b>(i, j)[1] > 0 
				|| dst.at<cv::Vec3b>(i, j)[2] > 0) {
				pt.x = i;
				pt.y = j;
				points.push_back(pt);
				minx = minx > i ? i : minx;
				miny = miny > j ? j : miny;
				maxx = maxx < i ? i : maxx;
				maxy = maxy < j ? j : maxy;
			}
		}
	}
	int shape = points.size();
	if (shape == 0)return;
	int avDst[3] = { 0 };
	int avSrc[3] = { 0 };
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < shape; j++) {
			avSrc[i] += src.at<cv::Vec3b>(points[j].x, points[j].y)[i];
			avDst[i] += dst.at<cv::Vec3b>(points[j].x, points[j].y)[i];
		}
		avDst[i] = avDst[i] / shape;
		avSrc[i] = avSrc[i] / shape;
	}
	int tmp;
	
	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < shape; j++) {
	//		tmp = dst.at<cv::Vec3b>(points[j].x, points[j].y)[i] - avDst[i] + avSrc[i];
	//		if (tmp > 255)tmp = 255;
	//		else if (tmp < 0)tmp = 0;
	//		dst.at<cv::Vec3b>(points[j].x, points[j].y)[i] = tmp;
	//	}
	//}
	cv::Mat mask(width, height, CV_64F, cv::Scalar(0));
	std::vector<cv::Point> hull;
	cv::convexHull(cv::Mat(points), hull);
	if (maxx + 1 >= width)maxx -= 1;
	if (maxy + 1 >= height)maxy -= 1;
	double dis;
	float featherAmount = 0.2;
	featherAmount = featherAmount * max(maxx - minx, maxy - miny);
	if (featherAmount < 0)return;
	featherAmount = 1.0 / featherAmount;
	for (int i = minx + 1; i < maxx; i += 3) {
		for (int j = miny + 1; j < maxy; j += 3) {
			dis = cv::pointPolygonTest(hull, cv::Point2f(i, j), true) * featherAmount;
			if (dis > 1)dis = 1;
			else if (dis < 0)dis = 0;
			for (int n = i - 1; n <= i + 1; n++) {
				for (int m = j - 1; m <= j + 1; m++) {
					mask.at<double>(n, m) = dis;
				}
			}
		}
	}
	int x, y;
	for (int j = 0; j < shape; j++) {
		x = points[j].x;
		y = points[j].y;
		out.at<cv::Vec3b>(x, y)[0] = dst.at<cv::Vec3b>(x, y)[0] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*out.at<cv::Vec3b>(x, y)[0];
		out.at<cv::Vec3b>(x, y)[1] = dst.at<cv::Vec3b>(x, y)[1] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*out.at<cv::Vec3b>(x, y)[1];
		out.at<cv::Vec3b>(x, y)[2] = dst.at<cv::Vec3b>(x, y)[2] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*out.at<cv::Vec3b>(x, y)[2];
	}
}
// old blend face by  CuiXianfeng
void faceSwapByFive(unsigned char* im1, unsigned char * im2, int width, int height) {
	cv::Mat out(width, height, CV_8UC3, im1);
	cv::Mat src = out;
	cv::Mat dst(width, height, CV_8UC3, im2);
	std::vector<cv::Point> points;
	cv::Point pt;
	int minx = 100000, miny = 100000, maxx = 0, maxy = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			if (dst.at<cv::Vec3b>(i, j)[0] > 0) {
				pt.x = i;
				pt.y = j;
				points.push_back(pt);
				minx = minx > i ? i : minx;
				miny = miny > j ? j : miny;
				maxx = maxx < i ? i : maxx;
				maxy = maxy < j ? j : maxy;
			}
		}
	}
	int shape = points.size();
	if (shape == 0)return;
	int avDst[3] = { 0 };
	int avSrc[3] = { 0 };
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < shape; j++) {
			avSrc[i] += src.at<cv::Vec3b>(points[j].x, points[j].y)[i];
			avDst[i] += dst.at<cv::Vec3b>(points[j].x, points[j].y)[i];
		}
		avDst[i] = avDst[i] / shape;
		avSrc[i] = avSrc[i] / shape;
	}
	int tmp;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < shape; j++) {
			tmp = dst.at<cv::Vec3b>(points[j].x, points[j].y)[i] - avDst[i] + avSrc[i];
			if (tmp > 255)tmp = 255;
			else if (tmp < 0)tmp = 0;
			dst.at<cv::Vec3b>(points[j].x, points[j].y)[i] = tmp;
		}
	}
	cv::Mat mask(width, height, CV_64F, cv::Scalar(0));
	std::vector<cv::Point> hull;
	cv::convexHull(cv::Mat(points), hull);
	if (maxx + 2 >= width)maxx -= 2;
	if (maxy + 2 >= height)maxy -= 2;
	double dis;
	float featherAmount = 0.2;
	featherAmount = featherAmount * max(maxx - minx, maxy - miny);
	if (featherAmount < 0)return;
	featherAmount = 1.0 / featherAmount;
	for (int i = minx + 2; i < maxx; i += 5) {
		for (int j = miny + 2; j < maxy; j += 5) {
			dis = cv::pointPolygonTest(hull, cv::Point2f(i, j), true) * featherAmount;
			if (dis > 1)dis = 1;
			else if (dis < 0)dis = 0;
			for (int n = i - 2; n <= i + 2; n++) {
				for (int m = j - 2; m <= j + 2; m++) {
					mask.at<double>(n, m) = dis;
				}
			}
		}
	}
	int x, y;
	for (int j = 0; j < shape; j++) {
		x = points[j].x;
		y = points[j].y;
		out.at<cv::Vec3b>(x, y)[0] = dst.at<cv::Vec3b>(x, y)[0] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*out.at<cv::Vec3b>(x, y)[0];
		out.at<cv::Vec3b>(x, y)[1] = dst.at<cv::Vec3b>(x, y)[1] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*out.at<cv::Vec3b>(x, y)[1];
		out.at<cv::Vec3b>(x, y)[2] = dst.at<cv::Vec3b>(x, y)[2] * mask.at<double>(x, y) + (1 - mask.at<double>(x, y))*out.at<cv::Vec3b>(x, y)[2];
	}
}

extern inline float box_iou(float *a, float *b, float eps = 1e-5);
std::vector<float> track_box(std::vector<std::vector<float>> &boxes, std::vector<float> &hist, int mode)
{
	std::vector<float> re;
	if (boxes.size() == 0 && mode == 1) return hist;
	if (hist.size() == 0 && mode == 0)  re = boxes[0];

	else {
		float min_dist = -1.f;
		{
			for (int i = 0; i < boxes.size(); ++i) {

				float iou = box_iou(boxes[i].data(), hist.data());
				if (min_dist < iou) {
					min_dist = iou;
					re = boxes[i];
				}
			}
		}

		if (min_dist > 0.8f) {
			if(mode==0) re = hist;
		}
		else {
			if (mode == 1) re = hist;
		}
	}
	return re;
}

void get_roi(int *box_roi, int *render_roi, float *box, int height, int width) {
	float box_w, box_h, face_w, face_h, center_x, center_y;
	box_w = box[2] - box[0];
	box_h = box[3] - box[1];
	face_w = 1.4 * box_w;
	face_h = 1.4 * box_h;
	center_x = (box[2] + box[0]) / 2;
	center_y = (box[3] + box[1]) / 2;
	box_roi[0] = int(max(0, center_x - face_w * 2));
	box_roi[1] = int(max(0, center_y - face_h * 2));
	box_roi[2] = int(min(width, center_x + face_w * 2));
	box_roi[3] = int(min(height, center_y + face_h * 2));

	int crop_w = box_roi[2] - box_roi[0];
	int crop_h = box_roi[3] - box_roi[1];
	crop_w -= crop_w % 4;
	crop_h -= crop_h % 4;
	box_roi[2] = box_roi[0] + crop_w;
	box_roi[3] = box_roi[1] + crop_h;

	render_roi[0] = box_roi[0];
	render_roi[1] = min(height - box_roi[1], height - box_roi[3]);
	render_roi[2] = crop_w;
	render_roi[3] = crop_h;
}

bool read_lmks(const char* file, std::vector<float> &lmks)
{
	FILE* fp = fopen(file, "r");
	if (fp == NULL) { printf("error\n");  return false; }
	float a, b;

	while(fscanf(fp, "%f %f\n", &a, &b) > 0) {
		if (a < 0 || b < 0) return false;
		lmks.push_back(a);
		lmks.push_back(b);
	}
	if (lmks.size() != 136) return false;
	fclose(fp);
	return true;
}

unsigned long long ms_since_epoch()
{
	return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}