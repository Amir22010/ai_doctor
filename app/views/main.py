from flask import render_template, jsonify, Flask, redirect, url_for, request
from app import app
import random
import os
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras import backend
backend.set_image_data_format('channels_first')
import tensorflow as tf
import cv2
import base64

dir_name = r"D:/StartupAI/AI_prototype/AI_Startup_Prototype/flaskSaaS-master/app/"
dirs = os.listdir(dir_name)
cur_dir =dir_name+dirs[8]+"/"
model_names = os.listdir(cur_dir)
model_file = cur_dir+model_names[0]
model = keras.models.load_model(model_file)
graph = tf.get_default_graph()

@app.route('/')

#disease_list = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', \
                  # 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', \
                  # 'Hernia']

@app.route('/upload')
def upload_file2():
   return render_template('index.html')
	
@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      path = os.path.join(dir_name+dirs[6]+"/", f.filename)
      print(path)
      f.save(path)
      frame = cv2.imread(path)
      img = image.load_img(path, target_size=(150,150))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      with graph.as_default():
          preds = model.predict(x)
          y_pred = preds.argmax(axis=-1)
          print(y_pred)
      if y_pred[0] == 0:
          result = "NORMAL"
      else:
          result="PNEUMONIA"
      image_content = cv2.imencode('.jpg', frame)[1].tostring()
      encoded_image = base64.encodestring(image_content)
      to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
      
      return render_template('uploaded.html', title='Success', predictions=result, user_image=to_send)

@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/map')
def map():
    return render_template('map.html', title='Map')


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')