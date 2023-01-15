# Docker computer vision show

# Prerequisite
Install Docker

# Execution 
  Create the container : 
    docker-compose up
    
 Create the container with all the permissions : 
 
    sh rundocker.sh
  --> It will hopen the container in the prompt
  
  Run the python app in the container : 
  
    python show.py

# Test the environment
  show.py contains 3 distinct apps! 
  
  ## To run the wished one : 
  
   -create the container
   
   -change the value of the var x in show.py :
   
   if x=1 : compute keypoints between two images
        The second image was created from the first 
        one following some transformations to obtain 
        a front view of the main object of the first image.
        You will find saved files in the folder named "result"
    
   x=2 : compute camera calibration from a s20 5g
   
   x=3 : stabilize a video.
   You will find saved files in the folder named "result"
   
   As it's a development stack,
   after changing the value of x and saving the file .py
   you don't need to recreate the container!!
   (Here the code is in the container : due to permission troubles; but the volume works)
   
   Run again 
   
    python show.py
   And see the change!
