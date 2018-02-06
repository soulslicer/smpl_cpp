'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


About this file:
================
This file defines the serialization functions of the SMPL model. 

Modules included:
- save_model:
  saves the SMPL model to a given file location as a .pkl file
- load_model:
  loads the SMPL model from a given file location (i.e. a .pkl file location), 
  or a dictionary object.

'''

__all__ = ['load_model', 'save_model']

import numpy as np
import cPickle as pickle
import chumpy as ch
from chumpy.ch import MatVecMult
from posemapper import posemap
from verts import verts_core
    
def save_model(model, fname):
    m0 = model
    trainer_dict = {'v_template': np.asarray(m0.v_template),'J': np.asarray(m0.J),'weights': np.asarray(m0.weights),'kintree_table': m0.kintree_table,'f': m0.f, 'bs_type': m0.bs_type, 'posedirs': np.asarray(m0.posedirs)}    
    if hasattr(model, 'J_regressor'):
        trainer_dict['J_regressor'] = m0.J_regressor
    if hasattr(model, 'J_regressor_prior'):
        trainer_dict['J_regressor_prior'] = m0.J_regressor_prior
    if hasattr(model, 'weights_prior'):
        trainer_dict['weights_prior'] = m0.weights_prior
    if hasattr(model, 'shapedirs'):
        trainer_dict['shapedirs'] = m0.shapedirs
    if hasattr(model, 'vert_sym_idxs'):
        trainer_dict['vert_sym_idxs'] = m0.vert_sym_idxs
    if hasattr(model, 'bs_style'):
        trainer_dict['bs_style'] = model.bs_style
    else:
        trainer_dict['bs_style'] = 'lbs'
    pickle.dump(trainer_dict, open(fname, 'w'), -1)


def backwards_compatibility_replacements(dd):

    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'



def ready_arguments(fname_or_dict):

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict))
    else:
        dd = fname_or_dict
        
    backwards_compatibility_replacements(dd)
        
    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:,0])        
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:,1])        
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:,2])        
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T    
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:    
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
            
    return dd



def load_model(fname_or_dict, params = None):
    dd = ready_arguments(fname_or_dict)

    dd['pose'][3] = 0.78;

    if params is not None:


        # left leg
        dd['pose'][3] = np.deg2rad(-45);
        dd['pose'][4] = 0;
        dd['pose'][5] = 0;

    #    # right leg
    #    dd['pose'][6] = -1;
    #    dd['pose'][7] = 0;
    #    dd['pose'][8] = 0;

    #    # torso
    #    dd['pose'][9] = -1;
    #    dd['pose'][10] = 0;
    #    dd['pose'][11] = 0;

        # left knee
#        dd['pose'][12] = params["left_knee_x"];
#        dd['pose'][13] = 0;
#        dd['pose'][14] = 0;

    
    args = {
        'pose': dd['pose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style']
    }

    import json
    customDict = dict()
    for k, v in dd.items():
        #print "KEY: " + k
        #print "TYPE: " + v.__class__.__name__

        if k == "J_regressor_prior":
            pass
            #print "\t" + str(v.shape)
            # No save
        if k == "pose":
            #print "\t" + str(v.shape)
            customDict[k] = (np.array(v)).tolist()
        if k == "f":
            #print "\t" + str(v.shape)
            customDict[k] = (np.array(v)).tolist()
            #print v
        if k == "J_regressor":
            pass
            #print "\t" + str(v.shape)
            # No save
        if k == "betas":
            pass
            #print "\t" + str(v.shape)
            #print v
            # No save
        if k == "kintree_table":
            #print "\t" + str(v.shape)
            #print v
            customDict[k] = (np.array(v)).tolist()
        if k == "J":
            #print "\t" + str(v.shape)
            #print v
            customDict[k] = (np.array(v)).tolist()
        if k == "v_shaped":
            pass
            #print "\t" + str(v.shape)
            #print v
            # No save
        if k == "weights_prior":
            pass
            #print "\t" + str(v.shape)
            #print v
            # No save
        if k == "trans":
            #print "\t" + str(v.shape)
            #print v
            customDict[k] = (np.array(v)).tolist()
        if k == "v_posed":
            #print "\t" + str(v.shape)
            #print v
            customDict[k] = (np.array(v)).tolist()
        if k == "weights":
            #print "\t" + str(v.shape)
            #print v
            customDict[k] = (np.array(v)).tolist()
        if k == "vert_sym_idxs":
            #print "\t" + str(v.shape)
            #print v
            customDict[k] = (np.array(v)).tolist()
        if k == "posedirs":
            pass
            #print "\t" + str(v.shape)
            # print v
            # No save
        if k == "pose_training_info":
            #print v
            customDict[k] = v
        if k == "bs_style":
            #print v
            customDict[k] = v
        if k == "v_template":
            pass
            #print "\t" + str(v.shape)
            #print v
            # No save
        if k == "shapedirs":
            pass
            #print "\t" + str(v.shape)
            # print v
            # No save
        if k == "bs_type":
            #print v
            customDict[k] = (np.array(v)).tolist()
    print fname_or_dict
    #with open("/home/raaj/project/"+'model.json', 'w') as outfile:
    #    json.dump(customDict, outfile)

#    print "*********"
#    import json
#    customDict = dict()
#    for key in args.keys():
#        obj = args[key]
#        type = args[key].__class__.__name__;
#        print key + " : " + type + " :"

#        if key == "J":
#            (rows, cols) = obj.shape
#            print obj.shape
#            print obj
#            customDict["J"] = (np.array(obj)).tolist()
#        elif key == "weights":
#            (rows, cols) = obj.shape
#            print obj.shape
#            #customDict["weights"] = (np.array(obj)).tolist()
#        elif key == "v":
#            (rows, cols) = obj.shape
#            print obj.shape
#            #customDict["v"] = (np.array(obj)).tolist()
#        elif key == "bs_style":
#            print obj
#        elif key == "kintree_table":
#            (rows, cols) = obj.shape
#            print obj.shape
#            print obj
#        elif key == "pose":
#            print obj
#            print len(obj)
#        elif key == "xp":
#            obj = np
#            print obj
#        elif key == "want_Jtr":
#            print obj
##        elif key == "xp":
##            (rows, cols) = obj.shape
##            print obj.shape
##        elif key == "bs_style":
##            (rows, cols) = obj.shape
##            print obj.shape
##        if dataType == "<class 'scipy.sparse.csc.csc_matrix'>":
##            print "XXXXXXXXXX"
##            print dd[key]
##        else:
##            print "BO"
#    with open('data.txt', 'w') as outfile:
#        json.dump(customDict, outfile)

#    print "*********"
#    # SHOULD YOU USE THE FINAL OUTPUT INSTEAD

#    for k, v in dd.items():
#        print k

#    print "*********"

    # pose, v, J, weights, kintree_table, want_Jtr, xp
    result, Jtr = verts_core(**args)
    result = result + dd['trans'].reshape((1,3))
    result.J_transformed = Jtr + dd['trans'].reshape((1,3))

    for k, v in dd.items():
        setattr(result, k, v)

    return result

