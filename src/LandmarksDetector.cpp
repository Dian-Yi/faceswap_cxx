#include <onnxruntime_c_api.h>
#include <assert.h>
#include "LandmarksDetector.h"

bool LandmarksDetector::CheckStatus(OrtStatus* status)
{
	if (status != NULL) {
		const char* msg = g_ort->GetErrorMessage(status);
		fprintf(stderr, "%s\n", msg);
		g_ort->ReleaseStatus(status);
		return false;
	}
	return true;
}


bool LandmarksDetector::init(const OrtApi* g_ort_only, OrtEnv* env, const wchar_t*model_path)
{
	//CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "face_lmk", &env));
	g_ort = g_ort_only;
	OrtSessionOptions* session_options;
	if(!CheckStatus(g_ort->CreateSessionOptions(&session_options))) return false;
	//g_ort->SetIntraOpNumThreads(session_options, 1);
	g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

	if(!CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session))) return false;

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
	for (size_t i = 0; i < num_output_nodes; i++) {
		char* output_name;
		status = g_ort->SessionGetOutputName(session, i, allocator, &output_name);
		output_node_names[i] = output_name;
		
		// print output node types
		OrtTypeInfo* typeinfo;
		status = g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo);
		const OrtTensorTypeAndShapeInfo* tensor_info;
		size_t num_dims;
		CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
		g_ort->GetTensorShapeElementCount(tensor_info, &num_dims);
		//printf("output %zu : num_dims=%zu\n", i, num_dims);
		output_size = num_dims;
		g_ort->ReleaseTypeInfo(typeinfo);
	}
	input_data = (float*)malloc(input_size*sizeof(float));
	g_ort->ReleaseSessionOptions(session_options);

	return true;
}

LandmarksDetector::~LandmarksDetector()
{
	g_ort->ReleaseSession(session);
	free(input_data);
}

vector<float> LandmarksDetector::predict(float *image)
{
	OrtMemoryInfo* memory_info;
	CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

	OrtValue* input_tensor = NULL;
	CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, image,
		input_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

	OrtValue* output_tensor = NULL;
	CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1,
		output_node_names.data(), 1, &output_tensor));

	float *out;
	CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&out));
	vector<float> landmark;
	for (int i = 0; i < lmk_num; ++i) {
		float x = out[i*2] * crop_info[0] + crop_info[2];
		float y = out[i * 2 + 1] * crop_info[1] + crop_info[3];
		landmark.push_back(x);
		landmark.push_back(y);
	}
	g_ort->ReleaseValue(output_tensor);
	g_ort->ReleaseValue(input_tensor);

	return landmark;
}

// input image BGR?
float* LandmarksDetector::get_input(Mat &image, vector<float> box)
{
	float box_w, box_h, face_w, face_h, center_x, center_y;
	int left, right, top, bot;
	Mat crop;

	image_w = image.cols;
	image_h = image.rows;

	box_w = box[2] - box[0];
	box_h = box[3] - box[1];
	face_w = 1.4 * box_w;
	face_h = 1.4 * box_h;
	center_x = (box[2] + box[0]) / 2;
	center_y = (box[3] + box[1]) / 2;
	left = int(max(0, center_x - face_w / 2));
	top = int(max(0, center_y - face_h / 2));
	right = int(min(image_w, center_x + face_w / 2));
	bot = int(min(image_h, center_y + face_h / 2));
	crop = image(Range(top, bot), Range(left, right));

	crop_info[0] = crop.cols; //face_w;
	crop_info[1] = crop.rows; //face_h;
	crop_info[2] = left;
	crop_info[3] = top;

	resize(crop, crop, Size(net_w, net_h));
	
	// normlize
	int w, h, c;
	w = crop.cols;
	h = crop.rows;
	c = crop.channels();
	assert(c == 3);
	int index = 0;
	for (int k = 0; k < c; ++k) {
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j)
				input_data[index++] = (float(crop.at<Vec3b>(i, j)[k]) - 127.0) / 127.0;
		}
	}
	return input_data;
}