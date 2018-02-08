import numpy as np

tensorA = np.array([[[1,2,3],
                  [4,5,6],
                  [7,8,9]],
                 [[10,11,12],
                  [13,14,15],
                  [16,17,18]],
                 [[19,20,21],
                  [22,23,24],
                  [25,26,27]]])

tensorB = np.array([[1,3],
                       [2,3],
                       [3,3]])


# print tensorA[0,:,:].dot(tensorB)
# print "**"
# print np.tensordot(tensorA, tensorB, axes=([2,0]))


print np.tensordot(tensorA, tensorB, axes=(2,0))


