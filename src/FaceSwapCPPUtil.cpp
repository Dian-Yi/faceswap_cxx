#include "pch.h"
#include "FaceSwapCPPUtil.h"
#include "header.h"
#include "util.h"
#include "glad/gl.h"
#include "GLFW/glfw3.h"

#include <thread>

static Eigen::Matrix<double, 2, 113> texture_coords;
static Eigen::Vector<double, 20> model_params;
static cv::VideoCapture capture;
static cv::Mat capture_frame;
static Eigen::MatrixXd shape3d;
static struct model_info model;
static GLuint face_texture, frame_buffer, render_buffer;
static GLFWwindow* window;
int canvas_width;
int canvas_height;

void render(double* params, unsigned char* data)
{
        auto start = std::chrono::high_resolution_clock::now();

        int i, j, k;
        for (i = 0; i < 20; ++i)
        {
                model_params(i) = *(params + i);
        }
        shape3d = get_shape3d(&model, model_params);

        glLoadIdentity();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, face_texture);
        glBegin(GL_TRIANGLES);
        for (i = 0; i < model.mesh.rows(); ++i)
        {
                Eigen::Vector3i triangle = model.mesh.row(i);
                for (j = 0; j < 3; ++j)
                {
                        GLfloat coord[2];
                        GLfloat fv[3];
                        coord[0] = texture_coords(0, triangle(j));
                        coord[1] = texture_coords(1, triangle(j));
                        fv[0] = shape3d(0, triangle(j));
                        fv[1] = shape3d(1, triangle(j));
                        fv[2] = shape3d(2, triangle(j));
                        glTexCoord2fv(coord);
                        glVertex3fv(fv);
                }
        }
        glEnd();
        glfwSwapBuffers(window);
        glfwPollEvents();

        glReadPixels(0, 0, canvas_width, canvas_height, GL_BGR, GL_UNSIGNED_BYTE, data);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli>  elapsed = end - start;
        std::cout << "Waited " << elapsed.count() << " ms\n";
        std::cout << "---------------------------------\n";
}

void calc_coords_data(int width, int height)
{
        texture_coords.row(0) /= width;
        texture_coords.row(1) /= height;
}

void load_texture(const char* file_name)
{
        cv::Mat frame;
        GLint width, height, total_bytes;
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

        calc_coords_data(width, height);
}

void init_texture(const char * file_name)
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
        calc_coords_data(width, height);
}

void init_rbo()
{
        glGenRenderbuffers(1, &render_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, render_buffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8
                , canvas_width, canvas_height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER
                , GL_DEPTH_STENCIL_ATTACHMENT
                , GL_RENDERBUFFER, render_buffer);
}

void init_fbo()
{
        glGenFramebuffers(1, &frame_buffer);
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);
        init_texture("face/1.jpg");
        init_rbo();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
                printf("init frame buffer success\n");
        else
                printf("init frame buffer failed\n");
}

void init_gl()
{
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
        window = glfwCreateWindow(canvas_width, canvas_height, "", NULL, NULL);
        glfwMakeContextCurrent(window);
        gladLoadGL(glfwGetProcAddress);
        init_fbo();
        init_rbo();
        glViewport(0, 0, canvas_width, canvas_height);

        glEnable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, canvas_width, canvas_height, 0, -10000, 10000);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

}

void init_util_dll(char* path, double* coords, char* face_path, int width, int height)
{
        int i, j;

        canvas_width = width;
        canvas_height = height;
        for (i = 0; i < 2; ++i)
        {
                for (j = 0; j < 113; ++j)
                {
                        texture_coords(i, j) = *(coords + (i * 113 + j));
                }
        }

        strcpy(model.path, path);
        load_3d_face_model(&model);

        init_gl();
        //load_texture(face_path);
}
