/* Batched float64 held-thrust defender forecast used by the research solver.
 * Physical constants, update order and four 0.05 s substeps match physics.py.
 * Build on the remote Linux host:
 * cc -O3 -ffp-contract=off -fPIC -shared projection_dynamics.c -lm
 *    -o projection_dynamics_v1.so
 * No fast-math: approximate gradients and resulting trajectories are evaluated
 * as their own numerical implementation, rather than asserted equivalent.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#ifdef _WIN32
#define CITF_API __declspec(dllexport)
#else
#define CITF_API
#endif

CITF_API int citf_projection_distances(const double *initial, const double *normalized,
                             int candidates, int defenders, int steps,
                             double *output) {
    if (!initial || !normalized || !output || candidates < 1 || defenders < 2 || steps < 1
        || defenders > INT_MAX/6) return -1;
    const size_t pairs = (size_t)defenders * (defenders - 1) / 2;
    if ((size_t)defenders > SIZE_MAX/(6*sizeof(double))
        || pairs > SIZE_MAX/sizeof(double)/(size_t)steps/(size_t)candidates
        || (size_t)defenders > SIZE_MAX/sizeof(double)/2/(size_t)candidates) return -1;
    double *state = malloc((size_t)defenders * 6 * sizeof(double));
    if (!state) return -2;
    const double inner = 0.05;
    const double tau = 6.283185307179586476925286766559;
    for (int batch = 0; batch < candidates; ++batch) {
        memcpy(state, initial, (size_t)defenders * 6 * sizeof(double));
        const double *action = normalized + (size_t)batch * defenders * 2;
        for (int tick = 0; tick < steps; ++tick) {
            for (int substep = 0; substep < 4; ++substep) {
                for (int boat = 0; boat < defenders; ++boat) {
                    double *x = state + boat * 6;
                    const double left = 250.0 + 750.0 * action[boat * 2];
                    const double right = 250.0 + 750.0 * action[boat * 2 + 1];
                    const double total_thrust = left + right;
                    const double yaw_moment = 1.25 * (left-right);
                    x[0] = x[0] + inner * x[3];
                    x[1] = x[1] + inner * x[4];
                    double yaw = fmod(x[2] + inner * x[5], tau);
                    if (yaw < 0.0) yaw += tau;
                    x[2] = yaw;
                    const double c = cos(yaw), s = sin(yaw);
                    double u = c * x[3] + s * x[4];
                    double v = -s * x[3] + c * x[4];
                    double r = x[5];
                    const double du = (400.0*r*v - (100.0+150.0*fabs(u))*u + total_thrust)/380.0;
                    const double dv = (-380.0*r*u - (100.0+150.0*fabs(v))*v)/400.0;
                    const double dr = (-20.0*u*v - (980.0+950.0*fabs(r))*r + yaw_moment)/1430.0;
                    u = u + inner*du;
                    v = v + inner*dv;
                    r = r + inner*dr;
                    x[3] = c*u-s*v;
                    x[4] = s*u+c*v;
                    x[5] = r;
                }
            }
            size_t pair = 0;
            for (int i = 0; i < defenders; ++i) {
                for (int j = i+1; j < defenders; ++j) {
                    const double dx = state[6*i]-state[6*j];
                    const double dy = state[6*i+1]-state[6*j+1];
                    output[((size_t)batch*pairs+pair)*steps+tick] = sqrt(dx*dx+dy*dy);
                    ++pair;
                }
            }
        }
    }
    free(state);
    return 0;
}
