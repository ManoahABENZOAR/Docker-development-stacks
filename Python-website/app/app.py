#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 18:20:07 2022

@author: cytech
"""

import time

import redis
from flask import Flask, render_template

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route('/')
def home():
    count = get_hit_count()
    
    pagename='templates/home'+str(count)+'.html'
    f = open(pagename, 'w')
    # the html code which will go in the file GFG.html
    html_template = """
    <html>
    <head></head>
    <body>
    <p> I have been seen {} times. </p>
    <p>This using redis!</p>
    </body>
    </html>
    """.format(count)
      
    # writing the code into the file
    f.write(html_template)
      
    # close the file
    f.close()
    
    prefix = "templates/"
    pagen = pagename[len(prefix):] if pagename.startswith(prefix) else pagename
    return render_template(pagen)

