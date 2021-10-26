#pragma once
#include <glad/gl.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <FaceSwap.h>
#include <GLFW/glfw3.h>
#include "util.h"

class FaceRender
{
private:
	GLuint face_texture, frame_buffer, render_buffer;
	GLFWwindow* window;
	int canvas_width;
	int canvas_height;

public:
	mode3D_info model3D;
	Eigen::MatrixXd textureCoords;
	Eigen::MatrixXd shape3d;
	funArgs fargs;
	funArgs margs;
	OrthographicParams ograpParams = {0,6};
	FaceRender(char *filepath) {
		ograpParams.nBlendshapes = 0;
		ograpParams.nParams = 6;
		load3DFaceModel(filepath); 
	};
	~FaceRender() { glfwTerminate(); };
	void load3DFaceModel(char *filepath);
	Eigen::MatrixXd getFaceTextureCoords(float *keypoints);
	void init_texture(const char * file_name);
	void load_texture(const char * file_name);
	void change_texture(char* imgpath, float* keypoints);
	void init_rbo();
	void init_fbo(char *imgpath);
	void init_gl(char *imgpath);
	void Init(char *imgpath, float *keypoints, int width, int height);
	//void render(Eigen::Vector<double, 20> model_params, unsigned char* data);
	void render_roi(Eigen::Vector<double, 20> model_params, unsigned char* data, int *roi);
	GLuint loadDDS(const char* imagepath);
};