#pragma once
#include "OrthographicProjectionBlendshapes.h"
# include <cmath>


Eigen::Vector<double, 20> GaussNewton(const Eigen::Vector<double, 20> *params, const funArgs *args, const OrthographicParams *ograpParams);
// scipy.optimize.minimize_scalar 
class Brent
{
private:
	double tol = 1.48e-8;
	int maxiter = 500;
	double _mintol = 1.0e-11;
	double _cg = 0.3819660;
	int _funcalls = 0;
	double bounds[3];  // xa, xb, xc

	const Eigen::Vector<double, 20> *params;
	const funArgs *fargs;
	const Eigen::VectorXd *direction;


public:
	Brent(const Eigen::Vector<double, 20>*p, const Eigen::VectorXd *d, const funArgs *args) {
		params = p;
		fargs = args;
		direction = d;
		_funcalls = 0;
	};
	~Brent() {};
	/**************************************/
	/**     minimize_scalar  function *****/
	double func(double x);
	double optimize();
	void bracket(double xa = 0.0, double xb = 1.0, double grow_limit = 110, int maxiter = 1000);
};