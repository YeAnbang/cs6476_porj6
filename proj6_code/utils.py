import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import mediapipe as mp
from PIL import Image
from scipy.ndimage.filters import maximum_filter
import tensorflow as tf


def imshow1(img):
    fig, axs = plt.subplots()
    axs.imshow(img)
    axs.axis('off')
    plt.show()


def calibrate(path='../data/cali2/example/', m=4, n=7):
    '''
    Use opencv to calibrate the camera

    Args:
        path: the folder to the calibration pictures
        m,n: the number of grids used for calibration
        (which is not the number of total grids in the cheesebord, but two grids smaller than it)


    Returns: intrinsic_matrix
    '''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((n*m,3), np.float32)
    objp[:,:2] = np.mgrid[0:n,0:m].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path += '*.jpg'
    print(path)
    images = glob.glob(path)
    # print(images)
    h, w = 0, 0
    gray = None
    for fname in images:
        # img = cv2.imread(fname)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (n,m),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (n,m), corners2,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, _, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    intrinsic_matrix, _=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    print(intrinsic_matrix)
    return intrinsic_matrix


def draw_box(image, pts):
    '''
    Drawing bounding box in the image
    Args:
        image: image array
        pts: bounding box vertices

    Returns:

    '''
    scaleX = image.shape[1]
    scaleY = image.shape[0]

    lines = [(0,1), (1,3), (0,2), (3,2), (1,5), (0,4), (2,6), (3,7), (5,7), (6,7), (6,4), (4,5)]
    for line in lines:
        pt0 = pts[line[0]]
        pt1 = pts[line[1]]
        pt0 = (int(pt0[0]*scaleX), int(pt0[1]*scaleY))
        pt1 = (int(pt1[0]*scaleX), int(pt1[1]*scaleY))
        cv2.line(image, pt0, pt1, (255,0,0), thickness=10)

    for i in range(8):
        pt = pts[i]
        pt = (int(pt[0]*scaleX), int(pt[1]*scaleY))
        cv2.circle(image, pt, 8, (0,255,0), -1)
        cv2.putText(image, str(i), pt,  cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)


def inference(img, model_path):
    """
    Running inference given the image model, and generate heatmap and displacements.
    If you don't know what is heatmap and displacement fields, you should go to read the objectron paper.
    (https://arxiv.org/pdf/2003.03522.pdf);
    Besides, the `objectron.py` in the repo
    Args:
        img: image file
        model_path: .tflite weights file

    Returns: heatmap and displacement files

    """
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']

    input_data = np.zeros(input_shape)
    input_data[0,:,:,0]=img[0,:,:]
    input_data[0,:,:,1]=img[1,:,:]
    input_data[0,:,:,2]=img[2,:,:]
    input_data = np.array(input_data, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data2 = interpreter.get_tensor(output_details[1]['index'])
    # print(output_data)
    output_data_reshape=np.zeros([1,1,40,30])
    output_data_reshape[0,0,:,:]=output_data[0,:,:,0]

    output_data2_reshape=np.zeros([1,16, 40,30])
    for i in range(16):
        output_data2_reshape[0, i,:,:]=output_data2[0,:,:,i]
    return output_data_reshape, output_data2_reshape


def plot_box_and_camera(points_3d, camera_center, R):
    """
    Visualize the actual 3D points and the estimated 3D camera center.

    """

    print("The camera center is at: \n", camera_center)

    v1 = R[:, 0]
    v2 = R[:, 1]
    v3 = R[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue',
               marker='o', s=10, depthshade=0)
    camera_center = camera_center.squeeze()
    ax.scatter(camera_center[0],  camera_center[1], camera_center[2], c='red',
               marker='x', s=20, depthshade=0)

    cc0, cc1, cc2 = camera_center

    point0 = points_3d[0]
    ax.plot3D([point0[0], point0[0]+2], [point0[1], point0[1]], [point0[2], point0[2]], c='r')
    ax.plot3D([point0[0], point0[0]], [point0[1], point0[1]+2], [point0[2], point0[2]], c='g')
    ax.plot3D([point0[0], point0[0]], [point0[1], point0[1]], [point0[2], point0[2]+2], c='b')

    ax.plot3D([cc0, cc0+v1[0]], [cc1, cc1+v1[1]], [cc2, cc2+v1[2]], c='r')
    ax.plot3D([cc0, cc0+v2[0]], [cc1, cc1+v2[1]], [cc2, cc2+v2[2]], c='g')
    ax.plot3D([cc0, cc0+v3[0]], [cc1, cc1+v3[1]], [cc2, cc2+v3[2]], c='b')


    # draw edges of the box

    min_z = min(points_3d[:, 2])
    min_x = min(points_3d[:, 0])
    min_y = min(points_3d[:, 1])
    for p in points_3d:
        x, y, z = p
        ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)
        ax.plot3D(xs=[x, min_x], ys=[y, y], zs=[z, z], c='black', linewidth=1)
        ax.plot3D(xs=[x, x], ys=[y, min_y], zs=[z, z], c='black', linewidth=1)
    x, y, z = camera_center
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)


def aa_perspective_n_points():
  '''
  Test perspective_n_points
  '''
  K = np.array([[1.11573975e+03, 0.00000000e+00 ,7.22275004e+02],
 [0.00000000e+00, 1.06794751e+03 ,5.31184727e+02],
 [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])


  box_2d = np.array([
                    [152.71358728, 407.62768173],
                    [126.6287899,  543.62051392],
                    [116.25713825,  68.85868454],
                    [ 62.28662968,  17.68793488],
                    [304.58366632, 395.42098427],
                    [365.79633236, 525.12167358],
                    [320.96123457,  60.44083023],
                    [409.12731171,  14.99428177]])

  box_3d = np.array([
                    [0.0, 0.0, 0.0, 1],
                    [0.5, 0.0, 0.0, 1],
                    [0.0, 0.0, 1.5, 1],
                    [0.5, 0.0, 1.5, 1],
                    [0.0, 0.5, 0.0, 1],
                    [0.5, 0.5, 0.0, 1],
                    [0.0, 0.5, 1.5, 1],
                    [0.5, 0.5, 1.5, 1]])

  wRc= np.array([[ 0, 0, 1],
                  [ 1,0,0],
                  [0,1,0]])

  wtc = np.array([[1],
                           [ 2],
                           [ 3]])
  P=K.dot(np.hstack((wRc.T, -wRc.T.dot(wtc))))
  box_2d_exp = P.dot(box_3d.T)
  # print(box_2d_exp.shape)
  box_2d_exp2=np.zeros((8,3))
  for i in range(8):
      box_2d_exp2[i,:] = box_2d_exp[:,i]/box_2d_exp[2,i]

  actual_rot, actual_trans = perspective_n_points(box_3d[:,:3], box_2d_exp2[:,:2], K)
  print(actual_rot)
  print(actual_trans)
  assert(np.allclose(wRc.T, actual_rot, atol=0.1))
  assert(np.allclose(wtc, actual_trans, atol=0.1))


def detect_peak(image, filter_size=3, order=0.5):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image,mask=~(image == local_max))

    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index


def decode(hm, displacements):
    '''
    Decode the heatmap and displacement feilds from the encoder.
    Args:
        hm: heatmap
        displacements: displacement fields

    Returns:
        normalized vertices coordinates in 2D image
    '''
    hm = hm.reshape(hm.shape[2:])     # (40,30)

    peaks=hm.argmax()
    peakX = [peaks%30]
    peakY = [peaks//30]


    peaks=hm.argmax()
    peakX = [peaks%30]
    peakY = [peaks//30]

    scaleX = hm.shape[1]
    scaleY = hm.shape[0]
    objs = []
    for x,y in zip(peakX, peakY):
        conf = hm[y,x]
        print(conf)
        points=[]
        for i in range(8):
            dx = displacements[0, i*2  , y, x]
            dy = displacements[0, i*2+1, y, x]
            points.append((x/scaleX+dx, y/scaleY+dy))
        objs.append(points)
    return objs


def get_world_vertices(width, height, depth):
    """
    Given the real size of the chair, return the real-world coordinates of the eight vertices
    in the same order as the detected bounding box from part 1
    Args:
        width: width of the chair, from vertex 0 to vertex 1
        height: height of the chair, from vertex 0 to vertex 4
        depth: depth of the chair, from vertex 0 to vertex 2
    Returns:
        vertices_world: (8,3), 8 vertices' real-world coordinates (x,y,z)
    """
    vertices_world=np.zeros((8,3))

    vertices_world[0,:]=np.array([0,0,0])
    vertices_world[1,:]=np.array([width,0,0])
    vertices_world[2,:]=np.array([0,0,depth])
    vertices_world[3,:]=np.array([width,0,depth])
    vertices_world[4,:]=np.array([0,height,0])
    vertices_world[5,:]=np.array([width,height,0])
    vertices_world[6,:]=np.array([0,height,depth])
    vertices_world[7,:]=np.array([width,height,depth])

    return vertices_world


def perspective_n_points(initial_box_points_3d, box_points_2d, intrinsic_matrix):
    """
    This function calculates the camera pose given 2D feature points
    with its 3D real world coordiantes matching.

    Args:
    -    initial_box_points_3d: N x 3 array, vertices 3D points in world coordinate
    -    box_points_2d: N x 2 array, vertices 2D points in image coordinate
    -    intrinsic_matrix, 3 x 3 array, the intrinsic matrix of your camera

    Returns:
    -    wRc_T: 3 x 3 array, the rotation matrix that transform from world to camera
    -    camera_center: 3 x 1 array, then camera center in world coordinate
    -    P: 3x4 projection matrix

    Hints: you can use cv2.solvePnP and cv2.Rodrigues in this function

    """
    wRc_T = None
    camera_center = None
    P = None

    initial_box_points_3d = np.array(initial_box_points_3d, dtype='float32')
    box_points_2d  = np.array(box_points_2d, dtype='float32')

    distCoeffs = np.zeros((8, 1), dtype='float32')


    _, rotation_vec, translation_vec = cv2.solvePnP(initial_box_points_3d,
                                            box_points_2d, intrinsic_matrix,
                                            distCoeffs)

    wRc_T, _ = cv2.Rodrigues(rotation_vec)

    P = intrinsic_matrix @ np.hstack((wRc_T, translation_vec))

    wRc = wRc_T.T
    camera_center = -wRc@translation_vec

    return wRc_T, camera_center, P


def projection_2d_to_3d(P, depth, pose2d):
    """
    This function calculates the inverse projection from 2D feature points to 3D real world coordiantes.
    Args:
    -    P: size is (3,4), camera projection matrix which combines both camera pose and intrinsic matrix, and it is not normalized
    -    depth: scalar, which provides the depth information (physica distance between you and camera in real world), in meter
    -    pose2d, size (n,2), where n is the number of 2D pose feature points in (x,y) image coordinates
    Returns:
    -    pose3d, size (n,3), where n is the number of 2D pose points. These are the 3D real-world
    coordiantes of human pose in the chair frame
    Hints:
    When only one 2D point is considered, one very easy way to solve this is treating it
    as three equations with three unknowns. However, since this is a linear system,
    it can also be solve via matrix manipulation. You can try to treat the P as a 3*3 matrix plus a 3*1 column vector,
    and see if it helps
    """
    pose3d = None

    n=len(pose2d)
    pose2d_h = np.hstack((pose2d, np.ones((n,1))))*depth
    p1=P[:3,:3]
    p2=P[:3,3]

    pose3d = np.linalg.inv(p1).dot((pose2d_h-p2).T).T

    # for verification
    # pose2d_reconstruct = P.dot(np.hstack((pose3d, np.ones((n,1)))).T)
    return pose3d