"""Contrôles algébriques ciblés des nouvelles analyses CPU."""
import unittest

import numpy as np

from probe_information import efficient_signal
from t3_repair_head import solve_correction, stats


class Calculs(unittest.TestCase):
    def test_repair_orthogonal_features(self):
        # Deux sous-espaces distincts : préserver forget et réparer retain est possible.
        xf=np.array([[1.,0.],[1.,0.]])
        xr=np.array([[0.,1.],[0.,1.]])
        delta=np.array([[2.,-2.],[3.,-3.]])
        c,_=solve_correction(xf,xr,delta,ridge_relative=1e-9)
        np.testing.assert_allclose(xf@c,0,atol=1e-7)
        np.testing.assert_allclose(xr@(delta+c),0,atol=1e-7)

    def test_normal_equations(self):
        rng=np.random.default_rng(7)
        xf,xr=rng.normal(size=(30,6)),rng.normal(size=(25,6))
        delta=rng.normal(size=(6,4));delta-=delta.mean(1,keepdims=True)
        c,rho=solve_correction(xf,xr,delta)
        gradient=(xf.T@xf/len(xf))@c+(xr.T@xr/len(xr))@(delta+c)+rho*c
        np.testing.assert_allclose(gradient,0,atol=1e-10)

    def test_softmax_gauge(self):
        rng=np.random.default_rng(9)
        a,b=rng.normal(size=(20,5)),rng.normal(size=(20,5))
        shifted=b+rng.normal(size=(20,1))*100
        self.assertAlmostEqual(stats(a,b)['KL_original_candidate'],stats(a,shifted)['KL_original_candidate'],places=12)

    def test_nuisance_can_absorb_whole_signal(self):
        signal=np.array([1.,2.,3.])
        raw,eff,rank=efficient_signal(signal,signal[:,None],np.ones(3))
        self.assertAlmostEqual(raw,14.)
        self.assertLess(eff,1e-25)
        self.assertEqual(rank,1)


if __name__=='__main__':
    unittest.main()
