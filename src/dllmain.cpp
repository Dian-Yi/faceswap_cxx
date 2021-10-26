// dllmain.cpp : 定义 DLL 应用程序的入口点。
// #include "pch.h"
#include <Windows.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <onnxruntime_c_api.h>
#include <vector>
#include <FaceSwap.h>
#include <io.h>
#include <string>
#include "FaceDetector.h"
#include "LandmarksDetector.h"
#include <chrono>
#include <omp.h>
#include <Eigen/Dense>
#include "FaceRender.h"
#include "OrthographicProjectionBlendshapes.h"
#include "NonLinearLeastSquares.h"
#include "util.h"
#include "filter.h"
#include "dllmain.h"
#include "VideoDecode.h"

static const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
static OrtEnv* env;
static LandmarksDetector *lmk_det;
static FaceDetector *face_det;
static Filter *lmk_filter;
static char modepath[128] = "./resource/candide.npz";
//static FaceRender frender(modepath);
FaceRender* frender = NULL;
FWParams FWparams;
VideoCapture cap;
int cap_w;
int cap_h;
bool is_blend = false;
std::vector<std::vector<float>> hisBoxes;
Mat camera_image;

/**************************************************************/
/*  just for debug assert info "__acrt_first_block == header" */
/*  because of opencv build MD and this project build MT      */
std::vector<cv::Point> hull;

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

FACESWAP_API int openCamera(int cameraID, int width, int height)
{
	//if (cap.open("./resource/putin1.mp4") == 0)
	if (cap.open(0) == 0)
		printf("open camera failed ...\n");

	cap.set(3, width);
	cap.set(4, height);
	cap_w = cap.get(3);
	cap_h = cap.get(4);
	
	return 1;
}

FACESWAP_API void setFWParams(FWParams params)
{
	FWparams = params;
}

int get_srcface_lmk(string file, string &img_path, vector<float>& lmks)
{
	img_path = file;
	string file_endwith = file.substr(file.length() - 4, file.length());

	if ((_access(file.c_str(), 0)) == -1) return -1;

	if (file_endwith.compare(".txt") == 0) {
		if (read_lmks(file.c_str(), lmks) == false) return -1;
		img_path = file.replace(file.end() - 3, file.end(), "jpg");
		if ((_access(img_path.c_str(), 0)) == -1) return -1;
	}
	else {
		Mat image = imread(file.c_str());
		if (image.data == NULL) return -1;
		float* input = (*face_det).get_input(image);
		vector<vector<float>> out_boxes = (*face_det).predict(input);

		if (out_boxes.size() == 0) return 0;
		float* in_lmk = (*lmk_det).get_input(image, out_boxes[0]);
		lmks = (*lmk_det).predict(in_lmk);
	}
	return 1;
}

FACESWAP_API int initCnnModel()
{
	FWparams.alpha = 0.5;
	FWparams.mutil_people = 0;
	FWparams.blend = 0;
	omp_set_num_threads(1);
	g_ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "face_det", &env);
	const wchar_t model_path[128] = L"./resource/slim_160_latest.onnx";
	lmk_det = new LandmarksDetector();
	if (!(lmk_det->init(g_ort, env, model_path))) return 0;

	const wchar_t model_path_1[128] = L"./resource/version-RFB-320.onnx";
	face_det = new FaceDetector();
	if(!(face_det->init(g_ort, env, model_path_1))) return 0;

	lmk_filter = new Filter(1920, 0.8);
	return 1;
}

FACESWAP_API int initFace3DModel(int width, int height, const char *srcimg)
{
	camera_image = cv::Mat::zeros(height, width, CV_8UC3);
	vector<float> lmks;
	string img_path;
	string file = srcimg;
	int re = get_srcface_lmk(file, img_path, lmks);
	if (re != 1) return re;
	if (frender != NULL) delete frender;
	frender = new FaceRender(modepath);
	frender->Init((char*)img_path.c_str(), lmks.data(), cap_w, cap_h);
	return 1;
}

FACESWAP_API int swapface(Mat &image, Mat& data)
{
	if (image.data != NULL) {
		//memcpy(camera_image.data, image, sizeof(unsigned char) * cap_w * cap_h * 3);
		camera_image = image;
		//imshow("im", camera_image);
		//waitKey(0);
		// flip(camera_image, camera_image, 1);
		float* input = face_det->get_input(camera_image);
		vector<vector<float>> out_boxes = face_det->predict(input);
		if (out_boxes.size() == 0) {
			data = camera_image.clone();
			return 0;
		}

		// get lmks;
		vector<vector<float>> lmks;
		vector<vector<float>> lmk_boxes;
		if (FWparams.mutil_people == 0) {
			vector<float> lmk_box;
			if (hisBoxes.size() == 0) lmk_box = out_boxes[0];
			else lmk_box = track_box(out_boxes, hisBoxes[0], 0);
			float* in_lmk = lmk_det->get_input(camera_image, lmk_box);
			vector<float> out_lmk = lmk_det->predict(in_lmk);
			lmks.push_back(out_lmk);
			lmk_boxes.push_back(lmk_box);
		}
		else {
			for (int i = 0; i < out_boxes.size(); ++i) {
				vector<float> lmk_box = track_box(hisBoxes, out_boxes[i], 1);
				float* in_lmk = lmk_det->get_input(camera_image, lmk_box);
				vector<float> out_lmk = lmk_det->predict(in_lmk);
				lmks.push_back(out_lmk);
				lmk_boxes.push_back(lmk_box);
			}
		}
		data = camera_image.clone();
		// filter lmks
		vector<vector<float>> smooth_lmks = lmk_filter->calculate(lmks);

		for (int i = 0; i < smooth_lmks.size(); ++i) {
			int box_roi[4] = { 0 };
			int render_roi[4] = { 0 };
			get_roi(box_roi, render_roi, lmk_boxes[i].data(), cap_h, cap_w);

			// get render img 
			funArgs fargs = frender->fargs;
			Eigen::MatrixXd kps = getKpt(&(frender->model3D), smooth_lmks[i].data());
			fargs.y = kps;
			Eigen::Vector<double, 20> modelParams;
			getInitialParameters(&fargs, &modelParams);

			Eigen::Vector<double, 20> newParams;
			newParams = GaussNewton(&modelParams, &fargs, &(frender->ograpParams));

			Mat out_roi(render_roi[3], render_roi[2], CV_8UC3, Scalar(0, 0, 0));

			frender->render_roi(newParams, out_roi.data, render_roi);
			flip(out_roi, out_roi, 0);

			Mat crop = camera_image(Range(box_roi[1], box_roi[3]), Range(box_roi[0], box_roi[2]));
			Mat crop_img;
			crop.copyTo(crop_img);
			//imshow("render", out_flip);
			//imshow("crop", crop_img);
			//waitKey(0);
			// float amount = is_blend ? 0.2 : 0.01;

			faceBlend(crop_img, out_roi, hull, FWparams.blend, 0.2, FWparams.alpha);

			crop_img.copyTo(camera_image(Range(box_roi[1], box_roi[3]), Range(box_roi[0], box_roi[2])));
		}

		hisBoxes = lmk_boxes;
		data = camera_image.clone();
		return 1;
	}
	else {
		return -1;
	}
}



FACESWAP_API int changeFace3D(const char* srcimg)
{
	vector<float> lmks;
	string img_path;
	string file = srcimg;
	int re = get_srcface_lmk(file, img_path, lmks);
	if (re != 1) return re;

	frender->change_texture((char*)img_path.c_str(), lmks.data());
}

FACESWAP_API void releaseCamera()
{
	cap.release();
}

FACESWAP_API void releaseSwapALL()
{
	delete lmk_det;
	delete face_det;
	delete lmk_filter;
	delete frender;
	g_ort->ReleaseEnv(env);
}

vector<string> glob(char *dir)
{
	vector<string> img_names;
	long long handle;
	struct _finddata_t fileinfo;

	handle = _findfirst(dir, &fileinfo);
	if (handle == -1)
		return img_names;
	do
	{
		string name = fileinfo.name;
		img_names.push_back(name);
		//printf("%s\n", fileinfo.name);

	} while (!_findnext(handle, &fileinfo));
	//_findnext()
	_findclose(handle);
	return img_names;
}

int main_dll()
{
	int w = 1280;
	int h = 720;
	cv::namedWindow("camera", CV_WINDOW_NORMAL);

	char dir[128] = "./resource/srcface/*.*g";
	string root_dir = "./resource/srcface/";
	vector<string> all_imgs = glob(dir);
	cv::Mat image;
	int re;
	re = openCamera(0, w, h);
	if (re != 1) return 0;
	re = initCnnModel();
	if (re != 1) return 0;
	int image_idx = 0;
	int image_len = all_imgs.size();
	string faceImage = root_dir + all_imgs[image_idx];
	while (initFace3DModel(cap_w, cap_h, faceImage.c_str()) != 1) {
		faceImage = root_dir + all_imgs[++image_idx];
		if (image_idx == all_imgs.size()) return 0;
	}


	int i = 0;
	cout << "iInitialization succeeded ... \n";
	FWparams.mutil_people = 0;
	FWparams.blend = 1;
	cv::Mat srcImage;
	int id = 0;
	char str[100];
	while (1) {
		unsigned long long t0 = ms_since_epoch();
		if (cap.read(srcImage)) {
			int re = swapface(srcImage, image);
			if (re == -1) break;
			unsigned long long t1 = ms_since_epoch();
			float total = t1 - t0;
			// printf("fps:%f\n", 1000 / total);
			// cout << re << endl;
			//sprintf(str, "C:\\Users\\12419\\Desktop\\GitHub\\reface\\%d.jpg", id);
			cv::imshow("camera", image);
			//cv::imwrite(str, image);
			id++;
		}
		else break;

		int key = cv::waitKey(10);
		if (key == 'q') break;
		else if (key == 'n') {
			do  {
				image_idx = (image_idx + 1) % image_len;
				faceImage = root_dir + all_imgs[image_idx];
				if (image_idx == image_len) return 0;
			} while (changeFace3D(faceImage.c_str())!= 1);
		}
		else if (key == 'b') {
			do {
				image_idx = (image_idx - 1 + image_len) % image_len;
				faceImage = root_dir + all_imgs[image_idx];
				if (image_idx == image_len) return 0;
			} while ((changeFace3D(faceImage.c_str()) != 1));
		}
		else if (key == 'm') {
			FWparams.mutil_people = (FWparams.mutil_people + 1) % 2;
		}
		else if (key == 'c') FWparams.blend = (FWparams.blend + 1) % 2;
	}

	releaseCamera();
	releaseSwapALL();
	cv::destroyAllWindows();

	return 0;
}

//int main_video()
//{
//	//int w = 1920;
//	//int h = 1080;
//	char srcs[4][128] = { "./resource/1.txt", "./resource/test.jpg" , "./resource/test2.jpg", "./resource/p2.txt" };
//
//	int id = 0;
//	char* srcimage = srcs[id];
//	cv::Mat image;
//	char videp_path[128] = "./resource/video1.mp4";
//	VideoCapture cap(videp_path);
//
//	int w, h;
//	w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
//	h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
//
//	int re;
//	re = initCnnModel();
//	if (re != 1) return 0;
//	re = initFace3DModel(w, h, srcimage);
//	if (re != 1) return 0;
//	printf("w =%d, h=%d\n", w, h);
//
//	while (1) {
//		if (cap.read(image)) {
//			int re = swapface(image.data, image);
//			cv::imshow("camera", image);
//			int key = cv::waitKey(0);
//		}
//		else break;
//	}
//	cap.release();
//	return 0;
//}

//FACESWAP_API int init3DModel(int width, int height, const char *srcimg, double *&textureCoords, int *&mesh)
//{
//	cap.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
//	cap.set(3, width);
//	cap.set(4, height);
//	cap_w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
//	cap_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
//
//	Mat image = imread(srcimg);
//	if (image.data == NULL) return -1;
//	float* input = (*face_det).get_input(image);
//	vector<vector<float>> out_boxes = (*face_det).predict(input);
//	
//	if (out_boxes.size() == 0) return 0;
//	float* in_lmk = (*lmk_det).get_input(image, out_boxes[0]);
//	vector<float> lmk = (*lmk_det).predict(in_lmk);
//	frender.Init((char*)srcimg, lmk.data(), cap_w, cap_h);
//	textureCoords = frender.textureCoords.data();
//	mesh = frender.model3D.mesh.data();
//
//	return 1;
//}

//FACESWAP_API int render(unsigned char* data, double *& shape3D)
//{
//	Mat image;
//	Mat image_flip;
//	if (cap.read(image)) {
//		float* input = face_det->get_input(image);
//		vector<vector<float>> out_boxes = face_det->predict(input);
//		
//		if (out_boxes.size() == 0) {
//			flip(image, image_flip, 1);
//			cvtColor(image_flip, image_flip, CV_BGR2RGB);
//			memcpy(data, image_flip.data, sizeof(unsigned char)*cap_w *cap_h * 3);
//			return false;
//		}
//
//		if (hinfo.hisBoxes.size() == 0) {
//			hinfo.hisBoxes = out_boxes[0];
//			hinfo.hisImage = image.clone();
//			flip(hinfo.hisImage, image_flip, 1);
//			cvtColor(image_flip, image_flip, CV_BGR2RGB);
//			memcpy(data, image_flip.data, sizeof(unsigned char)*cap_w *cap_h * 3);
//			return false;
//		}
//
//		//imshow("img1", image);
//		//waitKey(10);
//		flip(hinfo.hisImage, image_flip, 1);
//		cvtColor(image_flip, image_flip, CV_BGR2RGB);
//		memcpy(data, image_flip.data, sizeof(unsigned char)*cap_w *cap_h * 3);
//
//		vector<float> lmk_box = track_box(out_boxes, hinfo.hisBoxes);
//		float* in_lmk = lmk_det->get_input(image, lmk_box);
//		vector<float> out_lmk = lmk_det->predict(in_lmk);
//		vector<float> soomth_lmk = lmk_filter->calculate(out_lmk);
//		out_lmk = soomth_lmk;
//
//		memcpy(hinfo.his_lmks, out_lmk.data(), sizeof(float) * 136);
//
//		int box_roi[4] = { 0 };
//		int render_roi[4] = { 0 };
//		//get_roi(box_roi, render_roi, (hinfo.hisBoxes).data(), height, width);
//
//		// get render img 
//		funArgs fargs = frender.fargs;
//		Eigen::MatrixXd kps = getKpt(&(frender.model3D), out_lmk.data());
//		fargs.y = kps;
//		Eigen::Vector<double, 20> modelParams;
//		getInitialParameters(&fargs, &modelParams);
//
//		Eigen::Vector<double, 20> newParams;
//		newParams = GaussNewton(&modelParams, &fargs, &(frender.ograpParams));
//		frender.shape3d = get_shape3d(&(frender.margs), &newParams);
//		shape3D = frender.shape3d.data();
//
//		hinfo.hisImage = image.clone();
//		hinfo.hisBoxes = lmk_box;
//
//		return true;
//	} 
//	else return false;
//}

//int main_dll()
//{
//	initCnnModel();
//	char srcimage[128] = "./resource/test.jpg";
//	double *textureCoords = NULL;
//	int *mesh = NULL;
//	double *shape3D = NULL;
//	unsigned char *data = (unsigned char *)malloc(sizeof(unsigned char) * 1920*1080*3);
//
//	init3DModel(1920, 1080, srcimage, textureCoords, mesh);
//	
//	printf("textureCoords: \n");
//	for (int i = 0; i < 2; ++i) {
//		for (int j = 0; j < 113; ++j) {
//			printf("%lf ", textureCoords[i * 113 + j]);
//		}
//		printf("\n");
//	}
//	printf("---------------\n");
//
//	printf("mesh: \n");
//	for (int i = 0; i < 175; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			printf("%d ", mesh[i * 3 + j]);
//		}
//		printf("\n");
//	}
//	printf("---------------\n");
//
//	while (1) {
//		bool re = render(data, shape3D);
//		Mat image(1080, 1920, CV_8UC3, data);
//		imshow("img", image);
//		waitKey(10);
//
//		if (re) {
//			for (int i = 0; i < 3; ++i) {
//				for (int j = 0; j < 113; ++j) {
//					printf("%lf ", shape3D[i * 113 + j]);
//				}
//				printf("\n");
//			}
//			printf("---------------\n");
//		}
//	}
//	free(data);
//	return 0;
//}