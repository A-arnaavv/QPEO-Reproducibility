import numpy as np, pytest
from qpeo.operators import *

def test_probability_shape():
    theta=np.array([.1,.2,.3,.4])
    omega=np.ones((2,2))
    P=assignment_probabilities(theta,omega)
    assert P.shape==(2,2)
    assert np.all(P>=0)

def test_density_proxy_trace_one():
    rho=candidate_density_proxy(np.array([.2,.4,.6]))
    assert np.trace(rho)==pytest.approx(1.0)

def test_entropy_identical_candidates_zeroish():
    r=candidate_density_proxy(np.array([.2,.4,.6]))
    assert von_neumann_entropy([r,r,r])==pytest.approx(0.0,abs=1e-10)

def test_kappa_bounds():
    assert kappa_schedule(0)==pytest.approx(.2)
    assert .2<kappa_schedule(500)<2.0

def test_caqc_one_assignment_per_workload():
    P=np.array([[.8,.2],[.1,.9]])
    X=caqc_unconstrained_collapse(P,np.array([200,400]),kappa=.5)
    assert np.all(X.sum(1)==1)

def test_ebws_moves_target_toward_source():
    theta=np.array([[1.0,1.0],[0.0,0.0]])
    dep=np.array([[0,.9],[0,0]])
    out=ebws_update(theta,dep,delta_dep=.5,phi=.5)
    assert np.all(out[1]>0)

def test_bro_shapes():
    rng=np.random.default_rng(1)
    pop=np.ones((3,4))*.5
    out=biodiversity_restore(pop,pop[0],np.array([False,True,False]),rng)
    assert out.shape==pop.shape
    assert np.all((out>=0)&(out<=np.pi/2))
