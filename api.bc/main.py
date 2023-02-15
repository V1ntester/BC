import logging
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from flask import Flask, request


app = Flask(__name__,)

@app.route('/', methods=['POST'])
def upload():
   try:
      if request.files:
         imageFile = request.files['file']

         with open(f'temp/img.png', 'wb') as fp:
            for itm in imageFile:
               fp.write(itm) 

         model = tf.keras.models.load_model('model')

         img = image.img_to_array(image.load_img(f'temp/img.png', target_size=(150, 150)))
         img = preprocess_input(np.expand_dims(img, axis=0))

         prediction = model.predict(img)[0].tolist()

         if prediction[0] == 1:
            text = 'Covid'
         elif prediction[1] == 1:
            text = 'Lung Opacity'
         elif prediction[2] == 1:
            text = 'Normal'
         elif prediction[3] == 1:
            text = 'Viral pneumonia'

         return text
      else:
         return "no files"
   except:
      logging.exception('')
      return ''

if __name__ == '__main__':
    app.run(debug=True)
