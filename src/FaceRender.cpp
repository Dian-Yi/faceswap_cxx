#include "FaceRender.h"
#include "OrthographicProjectionBlendshapes.h"
#include "NonLinearLeastSquares.h"


Eigen::Vector3d getnorm(Eigen::Matrix3d matrix)
{
	int i, j;
	Eigen::Vector3d a = matrix.row(0);
	Eigen::Vector3d b = matrix.row(1);
	Eigen::Vector3d c = matrix.row(2);
	Eigen::Vector3d axis_x = b - a;
	Eigen::Vector3d axis_y = c - a;
	Eigen::Vector3d axis_z;
	axis_x = axis_x / axis_x.norm();
	axis_y = axis_y / axis_y.norm();
	axis_z = axis_x.cross(axis_y);
	axis_z = axis_z / axis_z.norm();

	return axis_z;
}

Eigen::MatrixXi fix_mesh_winding(Eigen::MatrixXi mesh, Eigen::MatrixXd mean_3d_shape)
{
	int i;
	for (i = 0; i < mesh.rows(); ++i)
	{
		Eigen::Vector3i triangle = mesh.row(i);
		Eigen::Matrix3d matrix;
		matrix.row(0) = mean_3d_shape.col(triangle(0));
		matrix.row(1) = mean_3d_shape.col(triangle(1));
		matrix.row(2) = mean_3d_shape.col(triangle(2));
		Eigen::Vector3d norm = getnorm(matrix);
		if (norm(2) > 0)
		{
			mesh(i, 0) = triangle(1);
			mesh(i, 1) = triangle(0);
			mesh(i, 2) = triangle(2);
		}
	}
	return mesh;
}

void FaceRender::load3DFaceModel(char *filepath)
{
	cnpy::npz_t model_file = cnpy::npz_load(filepath);
	npyarry2matrixd(&model_file, "mean3DShape", &(model3D.mean_3d_shape));
	npyarry2matrixi(&model_file, "mesh", &(model3D.mesh));
	npyarry2matrixi(&model_file, "idxs3D", &(model3D.idxs_3d));
	npyarry2matrixi(&model_file, "idxs2D", &(model3D.idxs_2d));

	npyarry2matrix3D(&model_file, "blendshapes", (model3D.blend_shapes));
	model3D.mesh = fix_mesh_winding(model3D.mesh, model3D.mean_3d_shape);

	ograpParams.nBlendshapes = model3D.blend_shapes.size();
	ograpParams.nParams += ograpParams.nBlendshapes;
}
Eigen::MatrixXd FaceRender::getFaceTextureCoords(float *keypoints)
{
	Eigen::Vector<double, 20> modelParams;
	// set fargs: kps, mean3DShape, blendshapes using select idx
	Eigen::MatrixXd kps = getKpt(&model3D, keypoints);
	getFuncArgs(&(model3D), &fargs);
	fargs.y = kps;

	// set margs
	margs.blend_shapes = model3D.blend_shapes;
	margs.mean_3d_shape = model3D.mean_3d_shape;

	getInitialParameters(&fargs, &modelParams);
	// GaussNewton
	Eigen::Vector<double, 20> newParams = GaussNewton(&modelParams, &fargs, &ograpParams);
	Eigen::MatrixXd textureCoords = fun(&margs, &newParams);

	return textureCoords;
}

void FaceRender::load_texture(const char* file_name)
{
	cv::Mat frame;
	GLint width, height;
	GLubyte* pixels = 0;

	glGenTextures(1, &face_texture);

	frame = cv::imread(file_name);
	width = frame.cols;
	height = frame.rows;

	glBindTexture(GL_TEXTURE_2D, face_texture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
		GL_BGR, GL_UNSIGNED_BYTE, frame.data);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	textureCoords.row(0) /= width;
	textureCoords.row(1) /= height;
}

void FaceRender::change_texture(char* imgpath, float* keypoints)
{
	textureCoords = getFaceTextureCoords(keypoints);
	cv::Mat frame;
	GLint width, height;

	frame = cv::imread(imgpath);
	width = frame.cols;
	height = frame.rows;

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
		GL_BGR, GL_UNSIGNED_BYTE, frame.data);

	textureCoords.row(0) /= width;
	textureCoords.row(1) /= height;

	
}


void FaceRender::init_texture(const char * file_name)
{
	cv::Mat frame;
	frame = cv::imread(file_name);
	int width = frame.cols;
	int height = frame.rows;
	glGenTextures(1, &face_texture);
	glBindTexture(GL_TEXTURE_2D, face_texture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width
		, height, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0
		, GL_TEXTURE_2D, face_texture, 0);

	textureCoords.row(0) /= width;
	textureCoords.row(1) /= height;

}

//void FaceRender::init_rbo()
//{
//	glGenRenderbuffers(1, &render_buffer);
//	glBindRenderbuffer(GL_RENDERBUFFER, render_buffer);
//	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8
//		, canvas_width, canvas_height);
//	glBindRenderbuffer(GL_RENDERBUFFER, 0);
//	glFramebufferRenderbuffer(GL_FRAMEBUFFER
//		, GL_DEPTH_STENCIL_ATTACHMENT
//		, GL_RENDERBUFFER, render_buffer);
//}
//
//void FaceRender::init_fbo(char *imgpath)
//{
//	glGenFramebuffers(1, &frame_buffer);
//	glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);
//	init_texture(imgpath);
//	init_rbo();
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//	if (!(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE))
//		printf("init frame buffer failed\n");
//}

void FaceRender::init_gl(char *imgpath)
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	window = glfwCreateWindow(canvas_width, canvas_height, "", NULL, NULL);
	if (window == NULL) {
		printf("glfwCreateWindow error!!! \n");
	}
	int re;
	glfwMakeContextCurrent(window);
	re = gladLoadGL(glfwGetProcAddress);
	if (!re) {
		printf("gladLoadGL error!!! \n");
	}
	//init_fbo(imgpath);
	//init_rbo();
	
	glViewport(0, 0, canvas_width, canvas_height);

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, canvas_width, canvas_height, 0, -10000, 10000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	load_texture(imgpath);
}

void FaceRender::Init(char *imgpath, float *keypoints, int width, int height)
{
	canvas_width = width;
	canvas_height = height;

	textureCoords = getFaceTextureCoords(keypoints);
	init_gl(imgpath);
}

//// old render, not use
//void FaceRender::render(Eigen::Vector<double, 20> model_params, unsigned char* data)
//{
//	int i, j;
//	shape3d = get_shape3d(&margs, &model_params);
//	glLoadIdentity();
//
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	glBindTexture(GL_TEXTURE_2D, face_texture);
//	glBegin(GL_TRIANGLES);
//	for (i = 0; i < model3D.mesh.rows(); ++i)
//	{
//		Eigen::Vector3i triangle = model3D.mesh.row(i);
//		for (j = 0; j < 3; ++j)
//		{
//			GLfloat coord[2];
//			GLfloat fv[3];
//			coord[0] = textureCoords(0, triangle(j));
//			coord[1] = textureCoords(1, triangle(j));
//			fv[0] = shape3d(0, triangle(j));
//			fv[1] = shape3d(1, triangle(j));
//			fv[2] = shape3d(2, triangle(j));
//			glTexCoord2fv(coord);
//			glVertex3fv(fv);
//		}
//	}
//	glEnd();
//	
//	//glfwSwapBuffers(window);
//	glfwPollEvents();
//
//	glReadPixels(0, 0, canvas_width, canvas_height, GL_BGR, GL_UNSIGNED_BYTE, data);
//}

void FaceRender::render_roi(Eigen::Vector<double, 20> model_params, unsigned char* data, int *roi)
{
	int i, j;
	shape3d = get_shape3d(&margs, &model_params);
	glLoadIdentity();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindTexture(GL_TEXTURE_2D, face_texture);
	glBegin(GL_TRIANGLES);
	for (i = 0; i < model3D.mesh.rows(); ++i)
	{
		Eigen::Vector3i triangle = model3D.mesh.row(i);
		for (j = 0; j < 3; ++j)
		{
			GLfloat coord[2];
			GLfloat fv[3];
			coord[0] = textureCoords(0, triangle(j));
			coord[1] = textureCoords(1, triangle(j));
			fv[0] = shape3d(0, triangle(j));
			fv[1] = shape3d(1, triangle(j));
			fv[2] = shape3d(2, triangle(j));
			glTexCoord2fv(coord);
			glVertex3fv(fv);
		}
	}
	glEnd();
	//glfwSwapBuffers(window);
	glfwPollEvents();

	glReadPixels(roi[0], roi[1], roi[2], roi[3], GL_BGR, GL_UNSIGNED_BYTE, data);
}