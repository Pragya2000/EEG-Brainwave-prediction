from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
app = Flask(__name__)

MODEL_PATH='eeg-brainwave-pred.h5'
model =tf.keras.models.load_model(MODEL_PATH)

@app.route('/index.html')
def home_page():
  return render_template('index.html')

@app.route('/about.html')
def about_page():
  return render_template('about.html')

@app.route('/uploadx.html')
def upload_file1():
   return render_template('uploadx.html')


def predict(df):
    X= df.drop('2548', axis=1)
    A=model.predict(X)
    num=np.argmax(A)
    if(num==0):
      return 'NEGATIVE'
    elif(num==1):
      return 'NEUTRAL'
    elif(num==2):
      return 'POSITIVE'

import requests
from random import randint


@app.route('/result.html', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
      name=request.form.get('first_name')
      mail=request.form.get('mail_id')
      f = request.files['file']
      df=pd.read_csv(f)
      emotion=predict(df)
      #if(emotion=='NEGATIVE'):
      num=randint(100001,999999)
      response = requests.post('https://events-api.notivize.com/applications/d5f045ef-7789-400f-a2f6-20a086899fb4/event_flows/c8fe106a-69db-4195-afe3-7c8f4dcedb9b/events', json = {
      'email': mail,
      'first_name': name,
      'result': '0',
      'unique_id': str(num),
      })
      print(response)
      
      return render_template('result.html',emotion=emotion)
        
if __name__ == '__main__':
   app.run(debug = True)
