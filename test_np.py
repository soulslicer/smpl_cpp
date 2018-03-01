import numpy as np
import pickle
import json
import cv2

dir = "./data/"
id = "00001"

joints = np.load(dir+id+"_joints.npy")
data = pickle.load( open(dir+id+"_body.pkl", "rb" ) )
img = cv2.imread(dir+id+"_image.png")

#print joints
#print data

output = dict()
output["pose"] = data["pose"].reshape(24,3).tolist()
output["betas"] = data["betas"].tolist()
output["resolution"] = img.shape
output["trans"] = data["t"].tolist()
output["f"] = data["f"]

#str
#print json.dumps(output)

print data.keys()

with open(dir+id+"_body.json", 'w') as outfile:
    json.dump(output, outfile)


#tensorA = np.array([[[1,2,3],
#                  [4,5,6],
#                  [7,8,9]],
#                 [[10,11,12],
#                  [13,14,15],
#                  [16,17,18]],
#                 [[19,20,21],
#                  [22,23,24],
#                  [25,26,27]]])

#tensorB = np.array([[1,3],
#                       [2,3],
#                       [3,3]])


## print tensorA[0,:,:].dot(tensorB)
## print "**"
## print np.tensordot(tensorA, tensorB, axes=([2,0]))



#print np.tensordot(tensorA, tensorB, axes=(2,0))


