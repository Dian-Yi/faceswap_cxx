#pragma once
#include <onnxruntime_c_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

using namespace std;
using namespace cv;

typedef struct {
	float box[4];
	float prob;
} detection;

class FaceDetector
{
private:
	int img_width = 1920;
	int img_height = 1080;
	int net_w = 320;
	int net_h = 240;
	int net_c;
	float *input_data;
	float prob_threshold = 0.85;
	size_t boxes_size; // cls shape: [1,boxes_size, 2], boxes shape: [1, boxes_size, 4]
	size_t input_size;
	vector<int64_t> input_node_dims;
	vector<const char*> input_node_names;
	vector<const char*> output_node_names;
	OrtSession* session;
	detection *dets;
	const OrtApi* g_ort;

public:
	~FaceDetector();
	bool CheckStatus(OrtStatus* status);
	FaceDetector() {};
	bool init(const OrtApi* g_ort_only, OrtEnv* env, const wchar_t *model_path);
	vector<vector<float>> predict(float *image);
	float* get_input(Mat& image);
};