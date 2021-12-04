import os
import numpy as np

from proj6_code.utils import *
from proj6_code.student_code import *


def test_pose_estimate(test_img='data/player1.jpg'):
  '''
  Tests the pose estimate
  '''
  if os.path.exists('data/player1.jpg'):
      test_img = 'data/player1.jpg'
  else:
      test_img = '../data/player1.jpg'

  land_mark, annotated_image = hand_pose_img(test_img)
  expected_left_toe = np.array([119.4369, 627.3868])
  detected_left_toe = np.array(sorted(land_mark, key=lambda x: x[0])[0])
  assert(np.allclose(expected_left_toe, detected_left_toe, atol=50))


def test_projection_2d_to_3d():
  '''
  Test projection_2d_to_3d
  '''
  K = np.array([[ 500,   0, 535],
                            [   0, 500, 390],
                            [   0,   0,  -1]])

  R = np.array([[ 0.5,   -1,  0],
                            [   0,    0, -1],
                            [   1,  0.5,  0]])

  t = np.array([[   1,    0, 0, 300],
                              [   0,    1, 0, 300],
                              [   0,    0, 1,  30]])
  P = np.matmul(K, np.matmul(R, t))
  pose2d = np.array([[100,200],[200,300],[300,400]])
  depth = 1
  n=len(pose2d)

  pose3d_detected = projection_2d_to_3d(P, depth, pose2d)
  pose2d_reconstruct = P.dot(np.hstack((pose3d_detected, np.ones((n,1)))).T)[:2,:].T

  assert(np.allclose(pose2d, pose2d_reconstruct, atol=0.1))


if __name__=="__main__":
  test_pose_estimate()
  # test_projection_2d_to_3d()

