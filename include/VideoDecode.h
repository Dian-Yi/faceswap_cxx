// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the VIDEODECODE_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// VIDEODECODE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef VIDEODECODE_EXPORTS
#define VIDEODECODE_API __declspec(dllexport)
#else
#define VIDEODECODE_API __declspec(dllimport)
#endif

extern VIDEODECODE_API int nVideoDecode;

extern "C"
{
	//VIDEODECODE_API int fnVideoDecode(void);
	//VIDEODECODE_API void Decode(BYTE* film, int n, bool isLoop);
	// , 0, false, true, 1280
	VIDEODECODE_API void DecodeFilm(char* films, int id, bool isTransparent,  bool isLoop,  int dstWidth);

	VIDEODECODE_API void GetFrame(int& width, int& height, int& size, unsigned char * buf, int id);
	//VIDEODECODE_API bool IsFinished();
	VIDEODECODE_API void QuitDecode(int);
	VIDEODECODE_API void Init();
	VIDEODECODE_API void Quit();
}
