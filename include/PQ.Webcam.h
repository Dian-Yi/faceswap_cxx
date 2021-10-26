#pragma once 

#include <memory> 

class Webcam
{
public:
	Webcam();
	~Webcam();

	int				open(int CamIdx, int width, int height);
	void			close();
	
	// fmt: "BGR", "BGRX", "RGB", "RGBX", etc  (case in-sensitive)
	int				setTargetColorFmt(const char* fmt);	

	// return NULL if device is disconnected, error occurred, etc
	unsigned char*	getFrame(); 

	// return NULL if device is disconnected, error occurred, etc
	//int				getFrame(unsigned char &*p);

	//unsigned char*	getFrame(); 

	__declspec( property( get=get_width ) )		int	width;
	__declspec( property( get=get_height) )		int	height;
	__declspec( property( get=get_pitch ) )		int	pitch;

	int		get_width();
	int		get_height();
	int		get_pitch();

private: 
	class Impl; 
	std::unique_ptr<Impl> impl; 

};
