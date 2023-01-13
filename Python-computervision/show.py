import cv2
import numpy as np
import glob
import os

    # TRY TO MODIFY X, SAVE THE CHANGE AND LAUNCH AGAIN THE COMMAND TO SEE THAT THE VOLUME WORK WELL
    # there are 3 value for x, which will run different program available on my gihub
    # for value 1 and 2 it saves new file in the folder named "result"
x=2


if x==1 :
###############################################################################
    #Keypoints
    img1 = cv2.imread('templates/p (1).jpg', 0)
    img2 = cv2.imread('templates/new p (1).png', 0)
    
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)# draw first 50 matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    path = 'result/keypoint/'
    cv2.imwrite(os.path.join(path , 'keypointmatching.jpg'), match_img)
    
    cv2.imshow('Matches', match_img)
    cv2.waitKey(100)

if x==2 :
###############################################################################
    #IMGcalibration
    #termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0),(6,5,0)
    objp = np.zeros((6*7,3), np. float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    
    #Arrays to store object points and image points from all the images. 
    objpoints = [] #3d point in real world space 
    imgpoints = [] #2d points in image plane.
    
    images = glob.glob('resize/*.jpg')
    path = 'result/calibration/'
    for fname in images:
        img = cv2.imread(fname)
        #print (fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners (gray, (7,6),None)
    
        # If found, add object points, image points (after refining them) 
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1),criteria) 
            imgpoints.append(corners2)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            cv2.imshow('img', img) 
            print (fname)
            cv2.waitKey(1000)
    
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\n")
    print("Here the matrix of intrinsic parameters")
    print(mtx)
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )


if x==3 :
###############################################################################
    #Fuse video
    def movingAverage(curve, radius): 
      window_size = 2 * radius + 1
      # Define the filter 
      f = np.ones(window_size)/window_size 
      # Add padding to the boundaries 
      curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
      # Apply convolution 
      curve_smoothed = np.convolve(curve_pad, f, mode='same') 
      # Remove padding 
      curve_smoothed = curve_smoothed[radius:-radius]
      # return smoothed curve
      return curve_smoothed 
    
    def smooth(trajectory): 
      smoothed_trajectory = np.copy(trajectory) 
      # Filter the x, y and angle curves
      for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
    
      return smoothed_trajectory
    
    def fixBorder(frame):
      s = frame.shape
      
      # Scale the image number% without moving the center
      
      T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.10)
      frame = cv2.warpAffine(frame, T, (s[1], s[0]))
      return frame
    
    
    # The larger the more stable the video, but less reactive to sudden panning
    SMOOTHING_RADIUS=50 
    
    # Read input video
    cap = cv2.VideoCapture('stabvid/tab.mp4')
    
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
     
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
     
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
     
    # Set up output video
    path = 'result/stabilization/'
    out = cv2.VideoWriter(os.path.join(path , 'video_out.avi'), fourcc, fps, (2 * w, h))
    
    # Read first frame
    _, prev = cap.read() 
     
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 
    
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32) 
    
    for i in range(n_frames-2):
      # Detect feature points in previous frame
      prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                         maxCorners=200,
                                         qualityLevel=0.01,
                                         minDistance=30,
                                         blockSize=3)
       
      # Read next frame
      success, curr = cap.read() 
      if not success: 
        break 
    
      # Convert to grayscale
      curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
    
      # Calculate optical flow (i.e. track feature points)
      curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
    
      # Sanity check
      assert prev_pts.shape == curr_pts.shape 
    
      # Filter only valid points
      idx = np.where(status==1)[0]
      prev_pts = prev_pts[idx]
      curr_pts = curr_pts[idx]
    
      #Find transformation matrix
      m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
      # Extract traslation
      dx = m[0,2]
      dy = m[1,2]
    
      # Extract rotation angle
      da = np.arctan2(m[1,0], m[0,0])
       
      # Store transformation
      transforms[i] = [dx,dy,da]
       
      # Move to next frame
      prev_gray = curr_gray
    
      print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
    
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0) 
     
    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory) 
    
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
     
    # Calculate newer transformation array
    transforms_smooth = transforms + difference
    
    # Reset stream to first frame 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
     
    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
      # Read next frame
      success, frame = cap.read() 
      if not success:
        break
    
      # Extract transformations from the new transformation array
      dx = transforms_smooth[i,0]
      dy = transforms_smooth[i,1]
      da = transforms_smooth[i,2]
    
      # Reconstruct transformation matrix accordingly to new values
      m = np.zeros((2,3), np.float32)
      m[0,0] = np.cos(da)
      m[0,1] = -np.sin(da)
      m[1,0] = np.sin(da)
      m[1,1] = np.cos(da)
      m[0,2] = dx
      m[1,2] = dy
    
      # Apply affine wrapping to the given frame
      frame_stabilized = cv2.warpAffine(frame, m, (w,h))
    
      # Fix border artifacts
      frame_stabilized = fixBorder(frame_stabilized) 
    
      # Write the frame to the file
      frame_out = cv2.hconcat([frame, frame_stabilized])
    
      # If the image is too big, resize it.
      if(frame_out.shape[1] > 1920): 
        frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2));
      
      cv2.imshow("Before and After", frame_out)
      cv2.waitKey(10)
      out.write(frame_out)
    
    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()