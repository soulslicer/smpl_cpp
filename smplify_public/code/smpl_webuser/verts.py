'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


About this file:
================
This file defines the basic skinning modules for the SMPL loader which 
defines the effect of bones and blendshapes on the vertices of the template mesh.

Modules included:
- verts_decorated: 
  creates an instance of the SMPL model which inherits model attributes from another 
  SMPL model.
- verts_core: [overloaded function inherited by lbs.verts_core]
  computes the blending of joint-influences for each vertex based on type of skinning

'''

import chumpy
import lbs
from posemapper import posemap
import scipy.sparse as sp
from chumpy.ch import MatVecMult
import numpy as np
import time

def ischumpy(x): return hasattr(x, 'dterms')

def verts_decorated(trans, pose, 
    v_template, J, weights, kintree_table, bs_style, f,
    bs_type=None, posedirs=None, betas=None, shapedirs=None, want_Jtr=False):

    for which in [trans, pose, v_template, weights, posedirs, betas, shapedirs]:
        if which is not None:
            assert ischumpy(which)

    v = v_template

    print "********"
    print "betas : " + str(betas.shape)
    print "posedirs : " + str(posedirs.shape)
    print "shapedirs : " + str(shapedirs.shape)

    if shapedirs is not None:
        if betas is None:
            betas = chumpy.zeros(shapedirs.shape[-1])
        v_shaped = v + shapedirs.dot(betas)
        #print shapedirs.dot(betas).shape

        #v_shaped = np.copy(v)
        #for i in range(0,shapedirs.shape[0]):
        #    v_shaped[i,:] = v[i,:] + shapedirs[i,:,:].dot(betas)

        #print v_shaped[1,:]
        #xx = np.random.rand(4,4,4)
        #print xx
        #print xx[0,:,1]
        #print (np.array(xx)).tolist()
        #print (np.array(xx)).tolist()[1][1][1]

    else:
        v_shaped = v
        
    if posedirs is not None:
        print pose.shape
        print posemap(bs_type)(pose).shape
        v_posed = v_shaped + posedirs.dot(posemap(bs_type)(pose))
        #v_posed = v_shaped
    else:
        v_posed = v_shaped
        
    v = v_posed

    # IS this needed
    if sp.issparse(J):
        regressor = J

        print regressor.shape
        J_tmpx = MatVecMult(regressor, v_shaped[:,0])
        J_tmpy = MatVecMult(regressor, v_shaped[:,1])
        J_tmpz = MatVecMult(regressor, v_shaped[:,2])
        J = chumpy.vstack((J_tmpx, J_tmpy, J_tmpz)).T

        #XX = np.array(regressor) * np.array(v_shaped[:,0])
        #print "--"
        #print XX.shape
        #print XX
        #print v_shaped[200,2]
        #print XX[1]

    else:
        assert(ischumpy(J))
        
    assert(bs_style=='lbs')
    result, Jtr = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr=True, xp=chumpy)
     
    tr = trans.reshape((1,3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.weights = weights
    result.kintree_table = kintree_table
    result.bs_style = bs_style
    result.bs_type =bs_type
    if posedirs is not None:
        result.posedirs = posedirs
        result.v_posed = v_posed
    if shapedirs is not None:
        result.shapedirs = shapedirs
        result.betas = betas
        result.v_shaped = v_shaped
    if want_Jtr:
        result.J_transformed = Jtr
        #print Jtr
    return result

def verts_core(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=chumpy):
    
    if xp == chumpy:
        assert(hasattr(pose, 'dterms'))
        assert(hasattr(v, 'dterms'))
        assert(hasattr(J, 'dterms'))
        assert(hasattr(weights, 'dterms'))
     
    assert(bs_style=='lbs')
    result = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr, xp)

    return result
