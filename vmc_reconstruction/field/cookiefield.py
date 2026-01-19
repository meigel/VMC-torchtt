from __future__ import division
from dolfin import *
import numpy as np


DEBUG = False


code = '''
class CookieField : public Expression {
    public:

    const double eps = 1e-10;
    double kappa;
    std::vector<double> rdata;

    CookieField() : Expression() {}

    void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const {
        values[0] = kappa;
        double dist2;
        for(uint i=0; i < rdata.size()/4; ++i) {
            dist2 = (rdata[i*4]-x[0])*(rdata[i*4]-x[0]) + (rdata[i*4+1]-x[1])*(rdata[i*4+1]-x[1]);
            if(dist2 <= rdata[i*4+3]*rdata[i*4+3] + eps*eps) {
                // std::cout << "    (" << dist2 << " " << rdata[i*4+2] << ")";
                values[0] = rdata[i*4+2];
                break;
            }
        }
    }

    void update(const Array<double>& values, double kappa) {
        this->kappa = kappa;
        rdata.clear();
        for(int i=0; i<values.size(); ++i){
            // if(i < 8) std::cout << "(" << i << "): " << values[i];
            rdata.push_back(values[i]);
        }
    }
};
'''


class CookieField(object):
    def __init__(self, D, prepare_cookie_subdomain_data, kappa=1):
        self.D = D
        self.prepare_cookie_subdomain_data = prepare_cookie_subdomain_data
        self.kappa = kappa
        self.ex = ex = Expression(code, degree=1)

    def realisation(self, y, V):
        if DEBUG: print('CookieField realization')
        # setup subdomains and coefficient
        subdomain_data = self.prepare_cookie_subdomain_data(y, cookie_D=self.D)
        self.ex.update(subdomain_data.ravel(), self.kappa)
        ex = interpolate(self.ex, V)                            # interpolate onto reference space

        if DEBUG:
            # import matplotlib.pyplot as plt
            # fig1 = plt.figure()
            plot(ex) # dolfin function
            # plt.gca().set_aspect('equal')
            # fig1.savefig('mu1-%i.png' % np.random.randint(0, 10000))

        return ex
