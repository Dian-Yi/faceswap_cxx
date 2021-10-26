#pragma once
#include <vector>

/************************************************************************************
*	                   "one euro filter" for smooth face landmarks
*   The 1€ Filter is a low pass filter for filtering noisy signals in real-time
*	about url: https://jaantollander.com/post/noise-filtering-using-one-euro-filter/
			   http://www.lifl.fr/~casiez/1euro/InteractiveDemo/
*	
*************************************************************************************/

class Filter {
private:
	std::vector<float> prev_lmks;
	std::vector<std::vector<float>> prev_all_lmks;
	// exponential smoothing parameters
	float thresh1 = 1920 * 0.005; //  using prev_lmks for now_lmks
	float thresh2 = 1920 * 0.01;  //  using exponential smoothing

	float alpha = 0.90;

	//  one euro filter parameters
	float min_cutoff = 1.0;
	float beta = 0.001;
	float d_cutoff = 1.0;
	std::vector<float> dx_prev;
public:
	Filter(int w, float alpha = 0.8, float min_cutoff = 1.0, float beta = 0.001, float d_cutoff = 1.0) {
		thresh1 = w * 0.002;
		thresh2 = w * 0.01;
		Filter::alpha = alpha;
		Filter::min_cutoff = min_cutoff;
		Filter::beta = beta;
		Filter::d_cutoff = d_cutoff;
		dx_prev = std::vector<float>(136, 0.0);
	};
	~Filter() {};
	float oneEuroFilter(float x, float x_prev, int i);
	std::vector<float> calculate(std::vector<float> &now_lmks);
	std::vector<std::vector<float>> calculate(std::vector<std::vector<float>>& now_lmks);
};