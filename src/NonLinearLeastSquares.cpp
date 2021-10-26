#include <math.h>
#include "NonLinearLeastSquares.h"
void swap(double *a, double *b)
{
	double tmp = *a;
	*a = *b;
	*b = tmp;
}

double Brent::func(double x)
{
	Eigen::Vector<double, 20> x0 = *params + *direction * x;
	Eigen::VectorXd r = residual(&x0, fargs);
	double res = r.array().pow(2).sum();
	return res;
}


void Brent::bracket(double xa, double xb, double grow_limit, int maxiter)
{
	double _gold = 1.618034;  // golden ratio : (1.0 + sqrt(5.0)) / 2.0
	double _verysmall_num = 1e-21;
	double fa = func(xa);
	double fb = func(xb);
	if (fa < fb) {
		swap(&xa, &xb);
		swap(&fa, &fb);
	}
	double xc = xb + _gold * (xb - xa);
	double fc = func(xc);
	int funcalls = 3;
	int iter = 0;

	double tmp1, tmp2, val, w, wlim, fw, denom;
	while (fc < fb) {
		tmp1 = (xb - xa) * (fb - fc);
		tmp2 = (xb - xc) * (fb - fa);
		val = tmp2 - tmp1;

		if (abs(val) < _verysmall_num) denom = 2.0 * _verysmall_num;
		else denom = 2.0 * val;
		w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
		wlim = xb + grow_limit * (xc - xb);
		if (iter > maxiter) {
			printf("error:breant.cpp in 41");
			break;
		}

		iter += 1;
		if ((w - xc) * (xb - w) > 0.0) {
			fw = func(w);
			funcalls += 1;
			if (fw < fc) {
				xa = xb;
				xb = w;
				fa = fb;
				fb = fw;

				bounds[0] = xa;
				bounds[1] = xb;
				bounds[2] = xc;
				return;
				// /return xa, xb, xc, fa, fb, fc, funcalls
			}
			else if (fw > fb) {
				xc = w;
				fc = fw;
				//return xa, xb, xc, fa, fb, fc, funcalls
				bounds[0] = xa;
				bounds[1] = xb;
				bounds[2] = xc;
				return;
			}
			w = xc + _gold * (xc - xb);
			fw = func(w);
			funcalls += 1;
		}
		else if ((w - wlim)*(wlim - xc) >= 0.0) {
			w = wlim;
			fw = func(w);
			funcalls += 1;
		}
		else if ((w - wlim)*(xc - w) > 0.0) {
			fw = func(w);
			funcalls += 1;
			if (fw < fc) {
				xb = xc;
				xc = w;
				w = xc + _gold * (xc - xb);
				fb = fc;
				fc = fw;
				fw = func(w);
				funcalls += 1;
			}
		}
		else {
			w = xc + _gold * (xc - xb);
			fw = func(w);
			funcalls += 1;
		}
	}

	bounds[0] = xa;
	bounds[1] = xb;
	bounds[2] = xc;
	//return xa, xb, xc, fa, fb, fc, funcalls
}

double Brent::optimize()
{
	double xa, xb, xc, x, w, v;
	double fx, fw, fv;
	double a, b, deltax;
	int iter = 0;

	bracket();

	xa = bounds[0];
	xb = bounds[1];
	xc = bounds[2];
	x = w = v = xb;
	fx = fw = fv = func(x);
	if (xa < xc) {
		a = xa;
		b = xc;
	}
	else {
		a = xc;
		b = xa;
	}
	_funcalls += 1;
	deltax = 0.0;

	double tol1, tol2, xmid, rat, tmp1, tmp2, p, dx_temp, u, fu;
	while (iter < maxiter) {
		tol1 = tol * abs(x) + _mintol;
		tol2 = 2.0 * tol1;
		xmid = 0.5 * (a + b);
		if (abs(x - xmid) < (tol2 - 0.5*(b - 1))) {
			break;
		}

		if (abs(deltax) <= tol1) {
			if (x >= xmid) deltax = a - x;
			else deltax = b - x;
			rat = _cg * deltax;
		}
		else {
			tmp1 = (x - w) * (fx - fv);
			tmp2 = (x - v) * (fx - fw);
			p = (x - v) * tmp2 - (x - w) * tmp1;
			tmp2 = 2.0 * (tmp2 - tmp1);
			if (tmp2 > 0.0) p = -p;
			tmp2 = abs(tmp2);
			dx_temp = deltax;
			deltax = rat;

			if ((p > tmp2 * (a - x)) && (p < tmp2 * (b - x)) && (abs(p) < abs(0.5 * tmp2 * dx_temp))) {
				rat = p * 1.0 / tmp2;        // if parabolic step is useful.
				u = x + rat;
				if ((u - a) < tol2 || (b - u) < tol2) {
					if (xmid - x >= 0) rat = tol1;
					else rat = -tol1;
				}
			}
			else {
				if (x >= xmid) deltax = a - x;
				else deltax = b - x;
				rat = _cg * deltax;
			}
		}

		if (abs(rat) < tol1) {
			if (rat >= 0) u = x + tol1;
			else u = x - tol1;
		}
		else u = x + rat;

		fu = func(u);
		_funcalls += 1;

		if (fu > fx) {
			if (u < x) a = u;
			else b = u;
			if (fu <= fw || w == x) {
				v = w;
				w = u;
				fv = fw;
				fw = fu;
			}
			else if ((fu <= fv) || (v == x) || (v == w)) {
				v = u;
				fv = fu;
			}
		}
		else {
			if (u >= x) a = x;
			else b = x;
			v = w;
			w = x;
			x = u;
			fv = fw;
			fw = fx;
			fx = fu;
		}
		iter += 1;
	}
	return x;
}

Eigen::Vector<double, 20> GaussNewton(const Eigen::Vector<double, 20> *params, const funArgs *args, const OrthographicParams *ograpParams)
{
	//Brent breat(params, direction, args);
	Eigen::VectorXd direction;
	Eigen::VectorXd r;
	Eigen::MatrixXd J, grad, H;
	double eps = 10e-7;
	int maxIter = 1;
	double oldCost = -1;
	double cost;
	double alpha;
	for (int i = 0; i < maxIter; ++i) {
		r = residual(params, args);
		cost = r.array().pow(2).sum();
		if (cost < eps || abs(cost - oldCost) < eps)
			// return value
			return (*params);
		J = jacobian(params, args, ograpParams);
		grad = J.transpose() * r;
		//std::cout << grad << std::endl;
		H = J.transpose() * J;
		direction = H.colPivHouseholderQr().solve(grad);
		//std::cout << direction << std::endl;
		Brent breat(params, &direction, args);
		alpha = breat.optimize();
	}
	Eigen::Vector<double, 20> x;
	x = (*params) + alpha * direction;
	return x;
}