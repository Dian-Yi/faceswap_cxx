#pragma once

#define FACESWAP_API __declspec(dllexport)

typedef struct {
	int blend;
	int mutil_people;
	float alpha;
}FWParams;

extern "C"
{
	///*******************************************************/
	///*      textureCoords: 2 x 113                         */
	///*        mesh       : 175 x 3                         */
	///*       return value: true，   successful             */
	///*                     false，   failed                */
	//FACESWAP_API int init3DModel(int width, int height, const char *srcimg, double *&textureCoords, int *&mesh);
	///*************************************************************/
	///*     shape3D : 3 x 113                                     */
	///*       image : w * h *3                                    */
	///*return value : true，  successful                          */
	///*               false，detected no face, only get image data*/
	//FACESWAP_API int render(unsigned char* image, double *&shape3D);

	/*************************************************/
	/*                  return value                 */
	/*       -1:   camera open error                 */
	/*        0:   load onnx model error             */
	/*        1:   successful                        */
	FACESWAP_API int initCnnModel();

	/*************************************************/
	/*                  return value                 */
	/*       -1:   srcimg  error                     */
	/*        0:   no face in srcimg                 */
	/*        1:   successful                        */
	FACESWAP_API int initFace3DModel(int width, int height, const char *srcimg);

	/*************************************************/
	/*                  return value                 */
	/*        0:   camera orign image                */
	/*        1:   swapping face image               */
	/*       -1:   camera error                      */
	//FACESWAP_API int swapface(Mat &image);
	FACESWAP_API int swapface(unsigned char* image, Mat& data);

	/*************************************************/
	/*                  return value                 */
	/*       -1:   srcimg  error                     */
	/*        0:   no face in srcimg                 */
	/*        1:   successful                        */
	FACESWAP_API int changeFace3D(char* srcimg);
	/*************************************************/
	/*                  return value                 */
	/*        0:   failed                            */
	/*        1:   successful                        */
	FACESWAP_API int openCamera(int cameraID, int width, int height);

	FACESWAP_API void setFWParams(FWParams params);

	FACESWAP_API void releaseCamera();
	FACESWAP_API void releaseSwapALL();
}
