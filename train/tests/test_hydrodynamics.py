"""Check the faster integrator against the original matrix implementation."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.modules import WAMV


class MatrixReference(WAMV):
    def compute_motion(self):
        ur, vr = self.project_to_robot_frame(self.velocity_r[:2])
        u, v = self.project_to_robot_frame(self.velocity[:2])
        r = self.velocity[2]
        crb = np.matrix([[0., -self.m*r, 0.], [self.m*r, 0., 0.], [0., 0., 0.]])
        ca = np.matrix([[0., 0., self.yDotV*vr+self.yDotR*r], [0., 0., -self.xDotU*ur],
                        [-self.yDotV*vr-self.yDotR*r, self.xDotU*ur, 0.]])
        dn = -np.matrix([[self.xUU*abs(ur), 0., 0.],
                         [0., self.yVV*abs(vr)+self.yRV*abs(r), self.yVR*abs(vr)+self.yRR*abs(r)],
                         [0., self.nVV*abs(vr)+self.nRV*abs(r), self.nVR*abs(vr)+self.nRR*abs(r)]])
        fx_l, fy_l = self.left_thrust*np.cos(self.left_pos), self.left_thrust*np.sin(self.left_pos)
        fx_r, fy_r = self.right_thrust*np.cos(self.right_pos), self.right_thrust*np.sin(self.right_pos)
        moment = fx_l*self.width/2-fy_l*self.length/2-fx_r*self.width/2-fy_r*self.length/2
        thrust = np.matrix([[fx_l+fx_r], [fy_l+fy_r], [moment]])
        mass = self.M_RB+self.M_A
        velocity = np.matrix([[u, v, r]]).T
        relative = np.matrix([[ur, vr, r]]).T
        rhs = -crb*velocity-(ca+self.D+dn)*relative+thrust
        acceleration = np.linalg.inv(mass.T*mass)*mass.T*rhs
        relative += acceleration*self.dt
        rotation, _ = self.get_robot_transform()
        relative[:2, :] = rotation*relative[:2, :]
        self.velocity_r = np.squeeze(np.array(relative))


class HydrodynamicsTests(unittest.TestCase):
    def test_trajectory_matches_original_with_currents_clipping_and_coupling(self):
        rng = np.random.default_rng(2841)
        for agility in (1., 2.75):
            original, optimized = MatrixReference(agility), WAMV(agility)
            for boat in (original, optimized):
                # Exercise off-diagonal inertia/damping as well as defaults.
                boat.yDotR, boat.nDotV = .3, -.2
                boat.yR, boat.nV, boat.yVR = -.4, .1, -.15
                boat.compute_constant_matrices()
                boat.reset(np.array([13., -21.]), 5.9, np.array([.1, -.2, .01]))
                boat.left_pos, boat.right_pos = .12, -.08
            for _ in range(400):
                thrust = rng.uniform(-900., 1900., 2) * agility
                current = rng.normal(0., [.15, .15, .005])
                original.step(thrust, current)
                optimized.step(thrust, current)
                np.testing.assert_allclose(optimized.pos, original.pos, atol=1e-9, rtol=1e-10)
                np.testing.assert_allclose(optimized.velocity_r, original.velocity_r, atol=1e-10, rtol=1e-10)
                self.assertAlmostEqual(optimized.theta, original.theta, places=9)


if __name__ == "__main__":
    unittest.main()
