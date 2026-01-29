from flask import Flask,render_template,request,url_for,redirect
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import pickle

Ridge_regression=pickle.load(open('models/ridge.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        new_data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=Ridge_regression.predict(new_data)

        return render_template('home.html',results=result[0])


    else:
        return render_template('home.html')
if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
