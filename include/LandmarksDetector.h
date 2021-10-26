#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <onnxruntime_c_api.h>
#include <vector>

#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)


using namespace std;
using namespace cv;

class LandmarksDetector
{
private:
	const OrtApi* g_ort;
	OrtSession* session;
	size_t output_size;   //landmarks shape: (1,143) ---> need [0-136]
	size_t input_size;
	float *input_data;
	int crop_info[4];  // crop face info: (w, h, left, top)
	int net_w = 160;
	int net_h = 160;
	int lmk_num = 68;  // 68 points
	int image_w;
	int image_h;

	vector<int64_t> input_node_dims;
	vector<const char*> input_node_names;
	vector<const char*> output_node_names;


public:
	OrtMemoryInfo* memory_info;
	LandmarksDetector() {};
	bool init(const OrtApi* g_ort, OrtEnv* env, const wchar_t*model_path);
	~LandmarksDetector();
	bool CheckStatus(OrtStatus* status);
	vector<float> predict(float *image);
	float* get_input(Mat &image, vector<float> box);
};


