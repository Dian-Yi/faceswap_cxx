#pragma once

#ifdef FACESWAP_EXPORTS
#define FACESWAP_API __declspec(dllexport)
#else
#define FACESWAP_API __declspec(dllimport)
#endif

extern "C"
{
        FACESWAP_API void init_util_dll(char* path, double* coords, char* face_path, int width, int height);
        FACESWAP_API void render(double* params, unsigned char *data);
}
