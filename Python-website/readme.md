# Docker website stack

# Prerequisite
Install Docker

# Execution 
There are 2 containers : a flask one (allow to create light web apps) & redis one (an open source, in-memory data store used by millions of developers as a database, cache, streaming engine, and message broker)

  Create the container : 
  
    docker-compose up
  
 Open the browser and type
     
     http:://127.0.0.1:8080/

# Test the environment
  Each time you refresh the pages app.py creates a new files named index+"n+1"+.html,
  in the folder templates and render it on flask
   
   As it's a development stack,
   you can change some part of the code like the html part by example,
   then save the file .py
   you don't need to recreate the container!!
   
   Refresh the page and see the chages!
