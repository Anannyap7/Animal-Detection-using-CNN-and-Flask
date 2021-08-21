# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:31:59 2021

@author: anann
"""

import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from flask import Flask , request, render_template
# secure_filename will ensure the images uploaded will get saved in the uploads folder
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("animal.h5")

@app.route('/')
def index():
    return render_template('cnn.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        # This extracts the filepath of the image uploaded
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        # This appends the original filepath to that of uploads
        filepath = os.path.join(basepath,'static/uploads',f.filename)
        print("upload folder is ", filepath)
        # This saves the filepath of the image
        f.save(filepath)
        
        file = "/static/uploads/" + f.filename
        
        # Testing the model
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        preds = model.predict(x)
        
        print("prediction",preds)
            
        index = ['bear','crow','elephant','racoon','rat']
        
        print(np.argmax(preds))
        
        result = "The predicted animal is : " + str(index[np.argmax(preds)])
        
    return render_template("cnn.html", result=result, uploaded_image=file)

if __name__ == '__main__':
    app.run(debug = True, threaded = False)