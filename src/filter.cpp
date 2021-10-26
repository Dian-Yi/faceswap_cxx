#include <math.h>
#include "filter.h"
# define M_PI 3.14159265358979323846 

float smoothing_factor(float t_e, float cutoff)
{
	float r;
	r = 2 * M_PI * cutoff * t_e;
	return r / (r + 1);
}

float exponential_smoothing(float a, float x, float x_prev)
{
	return a * x + (1 - a) * x_prev;
}

float Filter::oneEuroFilter(float x, float x_prev, int i)
{
	float t_e = 1.0;
	float a_d = smoothing_factor(t_e, d_cutoff);
	float dx = (x - x_prev) / t_e;
	float dx_hat = exponential_smoothing(a_d, dx, dx_prev[i]);
	float cutoff = min_cutoff + beta * abs(dx_hat);
	float a = smoothing_factor(t_e, cutoff);
	dx_prev[i] = dx_hat;
	return exponential_smoothing(a, x, x_prev);
}

std::vector<float> Filter::calculate(std::vector<float> &now_lmks)
{
	if (prev_lmks.size() == 0) {
		prev_lmks = now_lmks;
		return now_lmks;
	}
	std::vector<float> result;
	int len = now_lmks.size();

	float distance = 0;
	for (int i = 0; i < len / 2; ++i) {
		distance += sqrt(pow((now_lmks[i * 2] - prev_lmks[i * 2]), 2) + pow((now_lmks[i * 2 + 1] - prev_lmks[i * 2 + 1]), 2));
	}

	if (thresh1 * 68 > distance) {
		result = prev_lmks;
	}
	else {
		for (int i = 0; i < len / 2; ++i) {
			float dist = 0;
			dist = sqrt(pow((now_lmks[i * 2] - prev_lmks[i * 2]), 2) + pow((now_lmks[i * 2 + 1] - prev_lmks[i * 2 + 1]), 2));

			//if (dist < thresh1)  // maybe  it's most useful when you're still
			//{
			//	result.push_back(prev_lmks[i * 2]);
			//	result.push_back(prev_lmks[i * 2 + 1]);
			//}
			//else 
			//if (dist < thresh2) {
			//	result.push_back(exponential_smoothing(alpha, now_lmks[i * 2], prev_lmks[i * 2]));
			//	result.push_back(exponential_smoothing(alpha, now_lmks[i * 2 + 1], prev_lmks[i * 2 + 1]));
			//	dx_prev[i * 2] = 0.0;
			//	dx_prev[i * 2 + 1] = 0.0;
			//}
			//else          // maybe unuseful
			{
				result.push_back(oneEuroFilter(now_lmks[i * 2], prev_lmks[i * 2], i * 2));
				result.push_back(oneEuroFilter(now_lmks[i * 2 + 1], prev_lmks[i * 2 + 1], i * 2 + 1));
			}
		}
	}

	prev_lmks = result;
	return result;
}

std::vector<std::vector<float>> Filter::calculate(std::vector<std::vector<float>>& now_lmks)
{
	if (prev_all_lmks.size() == 0) {
		prev_all_lmks = now_lmks;
		return now_lmks;
	}
	std::vector<std::vector<float>> result;
	

	for (int i = 0; i < now_lmks.size(); ++i) {
		int key = -1;
		float min_dis = FLT_MAX;
		int len = now_lmks[i].size();

		for (int j = 0; j < prev_all_lmks.size(); ++j) {
			float distance = 0;
			for (int k = 0; k < len / 2; ++k) {
				distance += sqrt(pow((now_lmks[i][k * 2] - prev_all_lmks[j][k * 2]), 2) + 
					pow((now_lmks[i][k * 2 + 1] - prev_all_lmks[j][k * 2 + 1]), 2));
			}
			if (distance < min_dis) {
				min_dis = distance;
				key = j;
			}
		}

		if (thresh1 * 68 > min_dis) {
			result.push_back(prev_all_lmks[key]);
			//printf("static landmarks !!!\n");
		}
		else {
			std::vector<float> lmk;
			for (int k = 0; k < len / 2; ++k) {
				lmk.push_back(oneEuroFilter(now_lmks[i][k * 2], prev_all_lmks[key][k * 2], i * 2));
				lmk.push_back(oneEuroFilter(now_lmks[i][k * 2 + 1], prev_all_lmks[key][k * 2 + 1], i * 2));
			}
			result.push_back(lmk);
		}
	}
	prev_all_lmks = result;
	return result;
}