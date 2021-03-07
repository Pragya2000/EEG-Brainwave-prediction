from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
app = Flask(__name__)

MODEL_PATH='eeg-brainwave-pred.h5'
model =tf.keras.models.load_model(MODEL_PATH)
@app.route('/upload')
def upload_file1():
   return render_template('upload.html')

#@app.route('/predict',methods=['POST'])
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

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
      f = request.files['file']
      df=pd.read_csv(f)
      return predict(df)
        
if __name__ == '__main__':
   app.run(debug = True)