#include <iostream>
#include <stdio.h>
#include "FaceDetector.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)

bool FaceDetector::CheckStatus(OrtStatus* status)
{
	if (status != NULL) {
		const char* msg = g_ort->GetErrorMessage(status);
		fprintf(stderr, "%s\n", msg);
		g_ort->ReleaseStatus(status);
		return false;
	}
	return true;
}

bool FaceDetector::init(const OrtApi* g_ort_only, OrtEnv* env, const wchar_t*model_path)
{
	g_ort = g_ort_only;
	//CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "face_det", &env));
	OrtSessionOptions* session_options;
	if(!CheckStatus(g_ort->CreateSessionOptions(&session_options))) return false;
	//g_ort->SetIntraOpNumThreads(session_options, 1);
	g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

	if (!CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session))) return false;

	size_t num_input_nodes;
	OrtStatus* status;
	OrtAllocator* allocator;
	CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));

	// print number of model input nodes
	status = g_ort->SessionGetInputCount(session, &num_input_nodes);
	input_node_names.resize(num_input_nodes);

	for (size_t i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name;
		status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
		input_node_names[i] = input_name;

		// print input node types
		OrtTypeInfo* typeinfo;
		status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
		const OrtTensorTypeAndShapeInfo* tensor_info;
		CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

		// print input shapes/dims
		size_t num_dims;
		size_t element_count;

		g_ort->GetTensorShapeElementCount(tensor_info, &element_count);
		// printf("Input %zu : num_elements=%zu\n", i, element_count);
		input_size = element_count;
		CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
		// printf("Input %zu : num_dims=%zu\n", i, num_dims);
		input_node_dims.resize(num_dims);
		g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);

		g_ort->ReleaseTypeInfo(typeinfo);
	}

	// print number of model output nodes
	size_t num_output_nodes;
	status = g_ort->SessionGetOutputCount(session, &num_output_nodes);
	output_node_names.resize(num_output_nodes);
	vector<int64_t> input_node_dims;
	for (size_t i = 0; i < num_output_nodes; i++) {
		char* output_name;
		status = g_ort->SessionGetOutputName(session, i, allocator, &output_name);
		output_node_names[i] = output_name;
	}


	// print output node types
	OrtTypeInfo* typeinfo;
	status = g_ort->SessionGetOutputTypeInfo(session, 0, &typeinfo);
	const OrtTensorTypeAndShapeInfo* tensor_info;
	size_t num_dims;
	CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
	CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
	vector<int64_t> output_node_dims(num_dims);
	CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
	g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims.data(), num_dims);
	boxes_size = output_node_dims[1];
	dets = (detection*)malloc(boxes_size*sizeof(detection));
	input_data = (float *)malloc(input_size*sizeof(float));
	g_ort->ReleaseTypeInfo(typeinfo);

	g_ort->ReleaseSessionOptions(session_options);
	return true;
}

FaceDetector::~FaceDetector()
{
	g_ort->ReleaseSession(session);
	free(dets);
	free(input_data);
}

int nms_comparator_v3(const void *pa, const void *pb)
{
	detection a = *(detection *)pa;
	detection b = *(detection *)pb;
	float diff = 0;
	
	diff = a.prob - b.prob;
	
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}
inline float overlap(float x1, float x2, float y1, float y2)
{
	float max_ = max(x1, y1);
	float min_ = min(x2, y2);
	return min_ - max_;
}
inline float area_of(float *x)
{
	float w = max(0.f, x[2] - x[0]);
	float h = max(0.f, x[3] - x[1]);
	return w * h;
}

inline float box_iou(float *a, float *b, float eps = 1e-5)
{
	float overlap_w = overlap(a[0], a[2], b[0], b[2]);
	float overlap_h = overlap(a[1], a[3], b[1], b[3]);
	overlap_w = max(overlap_w, 0.f);
	overlap_h = max(overlap_h, 0.f);
	float i = overlap_w * overlap_h;

	float u = area_of(a) + area_of(b) - i;

	return i / (u + eps);
}

void hard_nms(detection*dets, int num,  float iou_thresh=0.3, int candidate_size=200)
{
	qsort(dets, num, sizeof(detection), nms_comparator_v3);
	int total = num < candidate_size ? num : candidate_size;

	for (int i = 0; i < total; ++i) {
		if (dets[i].prob == 0) continue;
		for (int j = i + 1; j < total; ++j) {
			float iou = box_iou(dets[i].box, dets[j].box);
			if (iou > iou_thresh)
				dets[j].prob = 0;
		}
	}
}

vector<vector<float>> FaceDetector::predict(float *image)
{
	OrtMemoryInfo* memory_info;
	CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

	OrtValue* input_tensor = NULL;
	CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, image,
		input_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

	OrtValue* output_tensor[2] = { NULL };
	CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1,
		output_node_names.data(), 2, output_tensor));

	float *confidence;
	float *boxes;
	CheckStatus(g_ort->GetTensorMutableData(output_tensor[0], (void**)&confidence));
	CheckStatus(g_ort->GetTensorMutableData(output_tensor[1], (void**)&boxes));

	//float prob_threshold = 0.8;
	int det_num = 0;
	int candidate_size = 200;
	vector<vector<float>> all_boxes;
	for (int i = 0; i < boxes_size; ++i) {
		if (confidence[i * 2 + 1] > prob_threshold) {
			dets[det_num].prob = confidence[i * 2 + 1];
			for (int j = 0; j < 4; ++j)
				dets[det_num].box[j] = boxes[i * 4 + j];
			det_num++;
		}
	}
	if(det_num > 0)
		hard_nms(dets, det_num);
	float scale[4] = { img_width , img_height, img_width , img_height };
	int total = min(det_num, candidate_size);
	for (int i = 0; i < total; ++i) {
		if(dets[i].prob > 0){
			vector<float> box;
			for(int j=0; j<4; ++j)
				box.push_back(dets[i].box[j] * scale[j]);
			all_boxes.push_back(box);
		}
	}

	g_ort->ReleaseValue(output_tensor[0]);
	g_ort->ReleaseValue(output_tensor[1]);
	g_ort->ReleaseValue(input_tensor);

	return all_boxes;
}

float *FaceDetector::get_input(Mat& image)
{
	img_width = image.cols;
	img_height = image.rows;

	Mat rgb;
	cvtColor(image, rgb, CV_BGR2RGB);
	resize(rgb, rgb, cv::Size(net_w, net_h));

	int h = rgb.rows;
	int w = rgb.cols;
	int c = rgb.channels(); // must == 3
	assert(c == 3);
	int index = 0;
	for (int k = 0; k < c; ++k) {
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) 
				input_data[index++] = (float(rgb.at<Vec3b>(i, j)[k]) - 127.0) / 128;
		}
	}
	
	return input_data;
}