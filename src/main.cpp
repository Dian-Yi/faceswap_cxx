//#include <stdio.h>
//#include <iostream>
//#include <windows.h>
//#include <onnxruntime_c_api.h>
//#include <vector>
//#include <io.h>
//#include <string>
//#include "FaceSwap.h"
//#include "FaceDetector.h"
//#include "LandmarksDetector.h"
//#include <omp.h>
//#include <thread>
//#include <Eigen/Dense>
//#include "FaceRender.h"
//#include "OrthographicProjectionBlendshapes.h"
//#include "NonLinearLeastSquares.h"
//#include "util.h"
//#include "filter.h"
//#include "dllmain.h"
//#include "PQ.Webcam.h"
////#include <vld.h>
////
////#define LOG 1
//////#pragma comment(lib, "D:/git/FaceSwap/faceswap_cxx/faceswap/lib/ReleasePQ.Webcam_x64_static.lib")
////
//void draw(Mat &image, vector<vector<float>> boxes, float *lmk, Scalar color = Scalar(0,0,255))
//{
//	//Scalar color(255, 0, 0);
//	//for (int i = 0; i < boxes.size(); ++i) {
//	//	vector<float> box = boxes[i];
//	//	rectangle(image, Point(int(box[0]), int(box[1])), Point(int(box[2]), int(box[3])), color, 1, 8);
//	//}
//	//Scalar color1(0, 0, 255);
//	for (int i = 0; i < 136; ++i) {
//		if (i % 2 == 1) continue;
//		circle(image, Point(int(lmk[i]), int(lmk[i + 1])), 1, color, 2);
//		//printf("%d %f %f\n", i, lmk[i], lmk[i + 1]);
//	}
//	//waitKey(10);
//}
////
////// test face detector results
//void test_face()
//{
//	omp_set_num_threads(1);
//	//export OMP_NUM_THREADS = 1;
//	const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
//	OrtEnv* env;
//	g_ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "face_det", &env);
//	const wchar_t* model_path = L"./resource/slim_160_latest.onnx";
//	LandmarksDetector lmk_det;
//	lmk_det.init(g_ort, env, model_path);
//	const wchar_t* model_path_1 = L"./resource/version-RFB-320.onnx";
//	FaceDetector face_det;
//	face_det.init(g_ort, env, model_path_1);
//
//	//Mat img = imread("black1.jpg");
//	//vector<float> input = face_det.get_input(img);
//	//vector<vector<float>> out_boxes = face_det.predict(input.data());
//	//printf("%d\n", out_boxes.size());
//	//vector<float> in_lmk = lmk_det.get_input(img, out_boxes[0]);
//	//vector<float> out_lmk = lmk_det.predict(in_lmk.data());
//	//draw(img, out_boxes, out_lmk);
//
//	
//	Mat image;
//	VideoCapture cap(0);
//	cap.set(3, 1920);
//	cap.set(4, 1080);
//	Filter lmk_filter(1920);
//	int his = 0;
//	vector<float> x0;
//	vector<float> hisBoxes;
//	int i = 0;
//	while (i<30) {
//		++i;
//		if (cap.read(image)) {
//		//if (1) {
//			//imshow("or", image);
//			//image = imread("./resource/frame0038.jpg");
//			float* input = face_det.get_input(image);
//			unsigned long long t0 = ms_since_epoch();
//			vector<vector<float>> out_boxes = face_det.predict(input);
//			if (out_boxes.size() == 0)  continue;
//			if (hisBoxes.size() == 0) hisBoxes = out_boxes[0];
//			vector<float> lmk_box; // = track_box(out_boxes, hisBoxes);
//			lmk_box = out_boxes[0];
//			unsigned long long t1 = ms_since_epoch();
//			float tot = t1 - t0;
//			float* in_lmk = lmk_det.get_input(image, lmk_box);
//			t0 = ms_since_epoch();
//			vector<float> out_lmk = lmk_det.predict(in_lmk);
//
//			vector<float> smooth_out = lmk_filter.calculate(out_lmk);
//
//			t1 = ms_since_epoch();
//			tot += t1 - t0;
//
//			//printf("time cost:%f\n", tot);
//			vector<vector<float>> lmk_boxes;
//			lmk_boxes.push_back(lmk_box);
//			draw(image, lmk_boxes, smooth_out.data());
//			//draw(image, out_boxes, out_lmk, Scalar(0, 255, 0));
//			
//			if (his == 0) {
//				his = 1;
//				x0 = out_lmk;
//			}
//			//else {
//			//	float dis_smooth = sqrt(pow(smooth_out[16]-x0[16],2)+pow(smooth_out[17]-x0[17],2));
//			//	float dis_out = sqrt(pow(out_lmk[16] - x0[16], 2) + pow(out_lmk[17] - x0[17], 2));
//			//	//cout << dis_smooth - dis_out << endl;
//			//	for (int i = 0; i < out_lmk.size(); ++i) {
//			//		cout << i << ": " << out_lmk[i] - smooth_out[i] << endl;
//			//	}
//			//	cout << "------------------" << endl;
//			//}
//			imshow("res", image);
//			waitKey(10);
//		}
//		else {
//			break;
//		}
//	}
//	g_ort->ReleaseEnv(env);
//	cap.release();
//}
////
////
//// test GaussNewton for render Params
//void testGaussNewton()
//{
//	char npfile[128] = "./resource/npdata2.txt";
//	FILE *fp = fopen(npfile, "r");
//	if (fp == NULL) { printf("error\n"); }
//	float lmks[136];
//	float a, b;
//	for (int i = 0; i < 68; ++i) {
//		fscanf(fp, "%f %f\n", &a, &b);
//		lmks[i * 2] = a;
//		lmks[i * 2 + 1] = b;
//	}
//
//	char modepath[128] = "./resource/candide.npz";
//	FaceRender frender(modepath);
//	//frender.load3DFaceModel(modepath);
//
//	funArgs fargs;
//	funArgs margs;
//	margs.mean_3d_shape = frender.model3D.mean_3d_shape;
//	margs.blend_shapes = frender.model3D.blend_shapes;
//	getFuncArgs(&(frender.model3D), &fargs);
//
//	Eigen::MatrixXd kps = getKpt(&(frender.model3D), lmks);
//	fargs.y = kps;
//	Eigen::Vector<double, 20> modelParams;
//	getInitialParameters(&fargs, &modelParams);
//
//	Eigen::Vector<double, 20> newParams;
//	OrthographicParams ograpParams{0,6};
//	ograpParams.nBlendshapes = 14;
//	ograpParams.nParams += 14;
//	newParams = GaussNewton(&modelParams, &fargs, &ograpParams);
//	cout << newParams << endl;
//}
////
////// test face swap
////void test_all() {
////	char npfile[128] = "./resource/npdata2.txt";
////	FILE *fp = fopen(npfile, "r");
////	if (fp == NULL) { printf("error\n"); }
////	float lmks[136];
////	float a, b;
////	for (int i = 0; i < 68; ++i) {
////		fscanf(fp, "%f %f\n", &a, &b);
////		lmks[i * 2] = a;
////		lmks[i * 2 + 1] = b;
////	}
////
////	char modepath[128] = "./resource/candide.npz";
////	FaceRender frender(modepath);
////	char img[128] = "./resource/pumpkin2.jpg";
////
////	const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
////	OrtEnv* env;
////	g_ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "face_det", &env);
////	const wchar_t* model_path = L"./resource/slim_160_latest.onnx";
////	LandmarksDetector lmk_det;
////	lmk_det.init(g_ort, env, model_path);
////
////	const wchar_t* model_path_1 = L"./resource/version-RFB-320.onnx";
////	FaceDetector face_det;
////	face_det.init(g_ort, env, model_path_1);
////	Mat image;
////	image = imread("./resource/1.jpg");
////	float* input = face_det.get_input(image);
////	vector<vector<float>> out_boxes = face_det.predict(input);
////	float* in_lmk = lmk_det.get_input(image, out_boxes[0]);
////	vector<float> out_lmk = lmk_det.predict(in_lmk);
////	frender.Init(img, lmks, 1920, 1080);
////
////	VideoCapture cap(0);
////	cap.set(3, 1920);
////	cap.set(4, 1080);
////	Mat out_frame(1080, 1920, CV_8UC3, Scalar(0, 0, 0));
////
////	while (1) {
////		if (cap.read(image)) {
////
////			// get face mode  result 
////			float* input = face_det.get_input(image);
////			vector<vector<float>> out_boxes = face_det.predict(input);
////			float* in_lmk = lmk_det.get_input(image, out_boxes[0]);
////			vector<float> out_lmk = lmk_det.predict(in_lmk);
////
////			unsigned long long t0 = ms_since_epoch();
////			// get render img 
////			funArgs fargs = frender.fargs;
////			Eigen::MatrixXd kps = getKpt(&(frender.model3D), out_lmk.data());
////			fargs.y = kps;
////			Eigen::Vector<double, 20> modelParams;
////			getInitialParameters(&fargs, &modelParams);
////
////			Eigen::Vector<double, 20> newParams;
////			OrthographicParams ograpParams = {0,6};
////			ograpParams.nBlendshapes = 14;
////			ograpParams.nParams += 14;
////			newParams = GaussNewton(&modelParams, &fargs, &ograpParams);
////			//frender.render(newParams, out_frame.data);
////			unsigned long long t1 = ms_since_epoch();
////			float tot_all = t1 - t0;
////			printf("time cost:%f\n", tot_all);
////
////			Mat out_flip;
////			flip(out_frame, out_flip, 0);
////
////			// blend img
////			faceSwapByThree(image.data, out_flip.data, 1080, 1920);
////
////			imshow("render", image);
////			waitKey(10);
////		}
////	}
////}
////
////

//
////bool get_srclmks(string img,  char *root_dir, float *out_lmk, FaceDetector *face_det, LandmarksDetector *lmk_det)
////{
////	// char root_dir[128] = "./resource/srcface/";
////	string img_path = root_dir + img;
////	string img_txt = img.replace(img.end() - 3, img.end(), "txt");
////	img_txt = root_dir + img_txt;
////	
////	if ((_access(img_txt.c_str(), 0)) != -1) {
////		read_lmks(img_txt.c_str(), out_lmk);
////		return true;
////	}
////	else {
////		Mat image = imread(img_path);
////		float* input = (*face_det).get_input(image);
////		vector<vector<float>> out_boxes = (*face_det).predict(input);
////		if (out_boxes.size() == 0) return false;
////
////		float* in_lmk = (*lmk_det).get_input(image, out_boxes[0]);
////		vector<float> lmk = (*lmk_det).predict(in_lmk);
////		memcpy(out_lmk, lmk.data(), sizeof(float) * 136);
////		return true;
////	}
////}
//
////int main_api()
////{
////	FILE *flog = fopen("log.txt", "w");
////	fprintf(flog, "start ...\n");
////	omp_set_num_threads(1);
////	char dir[128] = "./resource/srcface/*.*g";
////	vector<string> all_imgs = glob(dir);
////	int srcface_len = all_imgs.size();
////	if (srcface_len == 0) return 0;
////
////	int width = 1920;
////	int height = 1080;
////	char modepath[128] = "./resource/candide.npz";
////	// load face detection modes
////	const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
////	OrtEnv* env;
////	g_ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "face_det", &env);
////	const wchar_t* model_path = L"./resource/slim_160_latest.onnx";
////	LandmarksDetector lmk_det;
////	bool re = lmk_det.init(g_ort, env, model_path);
////	const wchar_t* model_path_1 = L"./resource/version-RFB-320.onnx";
////	FaceDetector face_det;
////	face_det.init(g_ort, env, model_path_1);
////
////	fprintf(flog, "onnx models load successfully ...\n");
////
////	Mat image;
////	char root_dir[128] = "./resource/srcface/";
////	int flag_img = 5;
////	string img = all_imgs[flag_img];
////	string img_path = root_dir + img;
////	float src_lmk[136] = {0};
////	bool is_srcface;
////	bool is_blend = false;
////	while (1) {
////		img = all_imgs[flag_img];
////		img_path = root_dir + img;
////		is_srcface = get_srclmks(img, root_dir, src_lmk, &face_det, &lmk_det);
////		if (is_srcface) break;
////		cout << "file error: " << img_path << "!!!\n";
////		flag_img = (flag_img + 1) % srcface_len;
////	}
////	//vector<vector<float>> tmp;
////	//image = imread(img_path);
////	//draw(image, tmp, src_lmk);
////	//imshow("image", image);
////	//imwrite("5_kpt.jpg", image);
////	//FILE *fp = fopen("5_kpt.txt", "w");
////	//for (int i = 0; i < 68; ++i) {
////	//	fprintf(fp, "%d %d\n", (int)src_lmk[i * 2], (int)src_lmk[i * 2 + 1]);
////	//}
////	//fclose(fp);
////	//waitKey(0);
////	FaceRender frender(modepath);
////	frender.Init((char *)img_path.c_str(), src_lmk, width, height);
////	
////	fprintf(flog, "load 3d reander successfully... \n");
////
////	Filter lmk_filter(width, 0.8);
////	hisInfo hinfo;
////	
////	VideoCapture cap;
////	if (cap.open(0)) {
////		fprintf(flog, "Open the camera successfully... \n");
////	}
////	else {
////		return 0;
////	}
////	cap.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
////	cap.set(3, width);
////	cap.set(4, height);
////	//cap.read(image);
////	cv::namedWindow("camera", CV_WINDOW_NORMAL);
////	float alpha = 0.5;
////	while (1) {
////		unsigned long long t0 = ms_since_epoch();
////		if (cap.read(image)) {
////			float* input = face_det.get_input(image);
////			vector<vector<float>> out_boxes = face_det.predict(input);
////
////			// first frame
////			if (hinfo.hisImage.data == NULL) {
////				hinfo.hisImage = image.clone();
////				Mat data = image.clone();
////				if (out_boxes.size() > 0) {
////					hinfo.hisBoxes = out_boxes[0];
////				}
////				imshow("camera", data);
////				waitKey(10);
////				continue;
////			}
////			if (out_boxes.size() == 0) {
////				Mat data = hinfo.hisImage.clone();
////				hinfo.hisImage = image.clone();
////				hinfo.hisBoxes.clear();
////				imshow("camera", data);
////				waitKey(10);
////				continue;
////			}
////			if (hinfo.hisBoxes.size() == 0) {
////				Mat data = hinfo.hisImage.clone();
////				hinfo.hisBoxes = out_boxes[0];
////				hinfo.hisImage = image.clone();
////				imshow("camera", data);
////				waitKey(10);
////				continue;
////			}
////
////			vector<float> lmk_box = track_box(out_boxes, hinfo.hisBoxes);
////			float* in_lmk = lmk_det.get_input(image, lmk_box);
////			vector<float> out_lmk = lmk_det.predict(in_lmk);
////			vector<float> soomth_lmk = lmk_filter.calculate(out_lmk);
////			out_lmk = soomth_lmk;
////
////			memcpy(hinfo.his_lmks, out_lmk.data(), sizeof(float) * 136);
////
////			// get render img 
////			funArgs fargs = frender.fargs;
////			Eigen::MatrixXd kps = getKpt(&(frender.model3D), out_lmk.data());
////			fargs.y = kps;
////			Eigen::Vector<double, 20> modelParams;
////			getInitialParameters(&fargs, &modelParams);
////			Eigen::Vector<double, 20> newParams;
////			newParams = GaussNewton(&modelParams, &fargs, &(frender.ograpParams));
////			// render roi image
////			int box_roi[4] = { 0 };
////			int render_roi[4] = { 0 };
////			get_roi(box_roi, render_roi, (hinfo.hisBoxes).data(), height, width);
////			Mat out_roi(render_roi[3], render_roi[2], CV_8UC3, Scalar(0, 0, 0));
////			frender.render_roi(newParams, out_roi.data, render_roi);
////			// blender roi image
////			Mat crop = hinfo.hisImage(Range(box_roi[1], box_roi[3]), Range(box_roi[0], box_roi[2]));
////			Mat crop_img;
////			crop.copyTo(crop_img);
////			Mat out_flip;
////			flip(out_roi, out_flip, 0);
////			//imshow("render_roi", out_flip);
////			//imshow("crop_roi", crop_img);
////			float amount = is_blend ? 0.2 : 0.01;
////			//faceBlend(crop_img, out_flip, color_tranf, amount);
////			faceBlend(crop_img, out_flip, is_blend, amount, alpha);
////			crop_img.copyTo(hinfo.hisImage(Range(box_roi[1], box_roi[3]), Range(box_roi[0], box_roi[2])));
////
////			Mat renderImage = hinfo.hisImage.clone();
////			hinfo.hisImage = image.clone();
////			hinfo.hisBoxes = lmk_box;
////
////			unsigned long long t1 = ms_since_epoch();
////			float tot_all = t1 - t0;
////			printf("time cost:%f\n", 1000 / tot_all);
////			char text[128];
////			float fps = 1000.0 / tot_all;
////			sprintf(text, "fps:%.3f", fps);
////			
////			putText(renderImage, text, Point(40, 100), FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2, 8, 0);
////			imshow("camera", renderImage);
////
////			int key = waitKey(20);
////			if (key == 'n' || key == 'm') {
////				if (key == 'n') 
////					flag_img = (flag_img - 1 + srcface_len) % srcface_len;
////				if (key == 'm')
////					flag_img = (flag_img + 1) % srcface_len;
////				while (1) {
////					img = all_imgs[flag_img];
////					img_path = root_dir + img;
////					is_srcface = get_srclmks(img, root_dir, src_lmk, &face_det, &lmk_det);
////					if (is_srcface) break;
////					cout << "file error:" << img_path << "!!!\n";
////					image = imread(img_path);
////					imshow("camera", image);
////					int key = waitKey(0);
////					if (key == 'n')
////						flag_img = (flag_img - 1 + srcface_len) % srcface_len;
////					if (key == 'm')
////						flag_img = (flag_img + 1) % srcface_len;
////				}
////				//frender.Init((char *)img_path.c_str(), src_lmk, width, height);
////				frender.change_face((char*)img_path.c_str(), src_lmk);
////			}
////			else if (key == 'c') {
////				cout << "Input alpha (value:0 - 1.0):";
////				cin >> alpha;
////			}
////			else if (key == 'b') {
////				is_blend = !is_blend;
////			}
////			else if (key == 'q') break;
////		}
////		else break;
////	}
////
////	return 0;
////}
////
//
//
////int main_test_jpg() {
////
////	int width = 1920;
////	int height = 1080;
////	char modepath[128] = "./resource/candide.npz";
////	//char img[128] = "./resource/1.jpg";            // faceswap src face
////	char img[128] = "./resource/srcface/test3.jpg";
////	char lmk_path[128] = "./resource/p3.txt";
////	// load face detection modes
////	const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
////	OrtEnv* env;
////	g_ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "face_det", &env);
////	const wchar_t* model_path = L"./resource/slim_160_latest.onnx";
////	LandmarksDetector lmk_det;
////	lmk_det.init(g_ort, env, model_path);
////	const wchar_t* model_path_1 = L"./resource/version-RFB-320.onnx";
////	FaceDetector face_det;
////	face_det.init(g_ort, env, model_path_1);
////
////	Mat image;
////	image = imread(img);
////	float* input = face_det.get_input(image);
////	vector<vector<float>> out_boxes = face_det.predict(input);
////	float* in_lmk = lmk_det.get_input(image, out_boxes[0]);
////	vector<float> out_lmk = lmk_det.predict(in_lmk);
////	//float out_lmk[136];
////	// Init  face render
////	FaceRender frender(modepath);
////	//read_lmks(lmk_path, out_lmk);
////	frender.Init(img, out_lmk.data(), width, height);
////	//frender.Init(img, out_lmk, width, height);
////
////	Filter lmk_filter(width, 0.8);
////	hisInfo hinfo;
////
////	VideoCapture cap(0);
////	cap.set(3, width);
////	cap.set(4, height);
////	//Mat out_frame(height, width, CV_8UC3, Scalar(0, 0, 0));
////	namedWindow("camera", CV_WINDOW_NORMAL);
////	cvNamedWindow("lmk", CV_WINDOW_NORMAL);
////	while (1) {
////		if (cap.read(image)) {
////			// get face mode  result 
////			image = imread("./resource/frame0038.jpg");
////			unsigned long long t0 = ms_since_epoch();
////			float* input = face_det.get_input(image);
////			vector<vector<float>> out_boxes = face_det.predict(input);
////			if (out_boxes.size() < 1) {
////				imshow("camera", image);
////				waitKey(10);
////				continue;
////			}
////			if (hinfo.hisBoxes.size() == 0) {
////				hinfo.hisBoxes = out_boxes[0];
////				hinfo.hisImage = image.clone();
////				continue;
////			}
////			vector<float> lmk_box =  track_box(out_boxes, hinfo.hisBoxes);
////			float* in_lmk = lmk_det.get_input(image, lmk_box);
////			vector<float> out_lmk = lmk_det.predict(in_lmk);
////			Mat tmp = image.clone();
////			draw(tmp, out_boxes, out_lmk.data());
////			imshow("lmk", tmp);
////			//vector<float> soomth_lmk = lmk_filter.calculate(out_lmk);
////			//out_lmk = soomth_lmk;
////
////			memcpy(hinfo.his_lmks, out_lmk.data(), sizeof(float) * 136);
////
////			int box_roi[4] = {0};
////			int render_roi[4] = { 0 };
////			get_roi(box_roi, render_roi, (hinfo.hisBoxes).data(), height, width);
////
////			// get render img 
////			funArgs fargs = frender.fargs;
////			Eigen::MatrixXd kps = getKpt(&(frender.model3D), out_lmk.data());
////			fargs.y = kps;
////			Eigen::Vector<double, 20> modelParams;
////			getInitialParameters(&fargs, &modelParams);
////			
////			Eigen::Vector<double, 20> newParams;
////			newParams = GaussNewton(&modelParams, &fargs, &(frender.ograpParams));
////			cout << newParams << endl << endl;
////			Mat out_roi(render_roi[3], render_roi[2], CV_8UC3, Scalar(0, 0, 0));
////			Mat crop = hinfo.hisImage(Range(box_roi[1], box_roi[3]), Range(box_roi[0], box_roi[2]));
////			Mat crop_img;
////			crop.copyTo(crop_img);
////			frender.render_roi(newParams, out_roi.data, render_roi);
////			Mat out_flip;
////			flip(out_roi, out_flip, 0);
////			//imshow("render", out_flip);
////
////			//faceSwapByThree(crop_img.data, out_flip.data, crop.rows, crop.cols);
////			faceBlend(crop_img, out_flip, false, 0.15);
////			crop_img.copyTo(hinfo.hisImage(Range(box_roi[1], box_roi[3]), Range(box_roi[0], box_roi[2])));
////			Mat renderImage = hinfo.hisImage.clone();
////			hinfo.hisImage = image.clone();
////			hinfo.hisBoxes = lmk_box;
////			
////			unsigned long long t1 = ms_since_epoch();
////			float tot_all = t1 - t0;
////			//printf("time cost:%f\n", tot_all);
////			char text[128];
////			float fps = 1000.0 / tot_all;
////			sprintf(text, "fps:%.3f", fps);
////			putText(renderImage, text, Point(40,100), FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2, 8, 0);
////			imshow("camera", renderImage);
////			if (waitKey(0) == 'q')
////				break;
////		}
////	}
////	g_ort->ReleaseEnv(env);
////	//system("pause");
////	return 0;
////}
////
////
//
//
////int main_pqcamera()
////{
////	FILE *flog = fopen("log.txt", "w");
////	fprintf(flog, "start ...\n");
////	omp_set_num_threads(1);
////	char dir[128] = "./resource/srcface/*.*g";
////	vector<string> all_imgs = glob(dir);
////	int srcface_len = all_imgs.size();
////	if (srcface_len == 0) return 0;
////
////	int width = 1920;
////	int height = 1080;
////	char modepath[128] = "./resource/candide.npz";
////	// load face detection modes
////	const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
////	OrtEnv* env;
////	g_ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "face_det", &env);
////	const wchar_t* model_path = L"./resource/slim_160_latest.onnx";
////	LandmarksDetector lmk_det;
////	lmk_det.init(g_ort, env, model_path);
////	const wchar_t* model_path_1 = L"./resource/version-RFB-320.onnx";
////	FaceDetector face_det;
////	face_det.init(g_ort, env, model_path_1);
////
////	fprintf(flog, "onnx models load successfully ...\n");
////
////	char root_dir[128] = "./resource/srcface/";
////	int flag_img = 5;
////	string img = all_imgs[flag_img];
////	string img_path = root_dir + img;
////	float src_lmk[136] = { 0 };
////	bool is_srcface;
////	bool is_blend = false;
////	while (1) {
////		img = all_imgs[flag_img];
////		img_path = root_dir + img;
////		is_srcface = get_srclmks(img, root_dir, src_lmk, &face_det, &lmk_det);
////		if (is_srcface) break;
////		cout << "file error: " << img_path << "!!!\n";
////		flag_img = (flag_img + 1) % srcface_len;
////	}
////
////	FaceRender frender(modepath);
////	frender.Init((char *)img_path.c_str(), src_lmk, width, height);
////
////	fprintf(flog, "load 3d reander successfully... \n");
////
////	Filter lmk_filter(width, 0.8);
////	hisInfo hinfo;
////
////	Webcam cap;
////	if (cap.open(0, width, height) == 0) {
////		cap.setTargetColorFmt("BGR");
////		fprintf(flog, "Open the camera successfully... \n");
////	}
////	else {
////		return 0;
////	}
////	
////	cv::namedWindow("camera", CV_WINDOW_NORMAL);
////	float alpha = 0.5;
////	while (1) {
////		unsigned char *p = cap.getFrame();
////		if (p!=NULL) {
////			// get face mode  result 
////			Mat image(Size(width, height), CV_8UC3, p, Mat::AUTO_STEP);
////			unsigned long long t0 = ms_since_epoch();
////			float* input = face_det.get_input(image);
////			vector<vector<float>> out_boxes = face_det.predict(input);
////			if (out_boxes.size() == 0) {
////				flip(image, image, 1);
////				hinfo.hisBoxes.clear();
////				imshow("camera", image);
////				waitKey(10);
////				continue;
////			}
////		
////			vector<float> lmk_box = track_box(out_boxes, hinfo.hisBoxes);
////			float* in_lmk = lmk_det.get_input(image, lmk_box);
////			vector<float> out_lmk = lmk_det.predict(in_lmk);
////			vector<float> soomth_lmk = lmk_filter.calculate(out_lmk);
////			out_lmk = soomth_lmk;
////
////			int box_roi[4] = { 0 };
////			int render_roi[4] = { 0 };
////			
////			get_roi(box_roi, render_roi, lmk_box.data(), height, width);
////
////			// get render img 
////			funArgs fargs = frender.fargs;
////			Eigen::MatrixXd kps = getKpt(&(frender.model3D), out_lmk.data());
////			fargs.y = kps;
////			Eigen::Vector<double, 20> modelParams;
////			getInitialParameters(&fargs, &modelParams);
////
////			Eigen::Vector<double, 20> newParams;
////			newParams = GaussNewton(&modelParams, &fargs, &(frender.ograpParams));
////
////			Mat out_roi(render_roi[3], render_roi[2], CV_8UC3, Scalar(0, 0, 0));
////			Mat crop = image(Range(box_roi[1], box_roi[3]), Range(box_roi[0], box_roi[2]));
////			Mat crop_img;
////			crop.copyTo(crop_img);
////			frender.render_roi(newParams, out_roi.data, render_roi);
////			Mat out_flip;
////			flip(out_roi, out_flip, 0);
////			//imshow("render", out_flip);
////			//imshow("crop", crop_img);
////			//waitKey(0);
////			float amount = is_blend ? 0.2 : 0.01;
////			faceBlend(crop_img, out_flip, is_blend, amount, alpha);
////			
////			crop_img.copyTo(image(Range(box_roi[1], box_roi[3]), Range(box_roi[0], box_roi[2])));
////			Mat renderImage = image.clone();
////			hinfo.hisImage = image.clone();
////			hinfo.hisBoxes = lmk_box;
////
////			unsigned long long t1 = ms_since_epoch();
////			float tot_all = t1 - t0;
////			//printf("time cost:%f\n", tot_all);
////			char text[128];
////			float fps = 1000.0 / tot_all;
////			sprintf(text, "fps:%.3f", fps);
////			flip(renderImage, renderImage, 1);
////			putText(renderImage, text, Point(40, 100), FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2, 8, 0);
////
////			imshow("camera", renderImage);
////
////			int key = waitKey(20);
////			if (key == 'n' || key == 'm') {
////				if (key == 'n')
////					flag_img = (flag_img - 1 + srcface_len) % srcface_len;
////				if (key == 'm')
////					flag_img = (flag_img + 1) % srcface_len;
////				while (1) {
////					img = all_imgs[flag_img];
////					img_path = root_dir + img;
////					is_srcface = get_srclmks(img, root_dir, src_lmk, &face_det, &lmk_det);
////					if (is_srcface) break;
////					cout << "file error:" << img_path << "!!!\n";
////					image = imread(img_path);
////					imshow("camera", image);
////					int key = waitKey(0);
////					if (key == 'n')
////						flag_img = (flag_img - 1 + srcface_len) % srcface_len;
////					if (key == 'm')
////						flag_img = (flag_img + 1) % srcface_len;
////				}
////				frender.Init((char *)img_path.c_str(), src_lmk, width, height);
////			}
////			else if (key == 'c') {
////				cout << "Input alpha (value:0 - 1.0):";
////				cin >> alpha;
////			}
////			else if (key == 'b') {
////				is_blend = !is_blend;
////			}
////			else if (key == 'q') break;
////		}
////		else break;
////	}
////
////	return 0;
////}
//
#include <windows.h>
#include <stdio.h>
extern int main_dll();
//extern int main_video();
int main()
{
	int i = 0;
	//while (i<10)
	//{
	//	char* test = new char[100];
	//	Sleep(100);
	//	++i;
	//	//delete test;
	//}
	////cin.get();
	//return 0;

	// test_all();
	// main_test_jpg();
	// test_face();
	// main_api();
	// main_dll();
	printf("Add your wanted portrait in folder \"resource/srcface\". \n");
	printf("Press Key:  ...\n");
	printf("         q :Quti            n :Next face      b :Before face \n");
	printf("(switch) m: Multi people    c :Blend mode \n");
	main_dll();
	//main_video();
	//system("pause");
	//std::thread t1(main_zhouyang);
	//for (int i = 0; ; ++i) {
	//	printf("other %d\n", i);
	//	Sleep(40);
	//}
	//t1.join();
	//main_pqcamera();
	//return 0;
}