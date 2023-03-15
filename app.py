from flask import Flask,request,render_template
import logging, os, sys, traceback,os.path,gc,random,requests, json
import pandas
import json
import numpy as np

from MLP_algoritm import *

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'

port = 1000

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'GET':
        train_val = 'Accuracy train: 0%'
        evaluate_val = 'Accuracy evaluate: 0%'
        hasil_predict = 'Belum ada hasil prediksi'
        return render_template("home.html", trainaccure=train_val, evaluateaccure=evaluate_val, hasil_predict=hasil_predict)
    elif request.method == 'POST':
        file_train = request.files['train']
        file_evaluate = request.files['evaluate']
        inp_predict = (request.form['predict'])
        
        if file_train.filename == '':
            train_val = 'Using Current Save Model'
        else:
            d = json.load(file_train)
            df_t = d["dataset"]
            dataset_t = pandas.DataFrame(df_t)

            # Slice data set in data and labels (2D-array)
            X_train = dataset_t.values[:,0:64].astype(float) # Data
            Y_train = dataset_t.values[:,64].astype(int) # Labels
            train_val = train(X_train, Y_train)
        
        if file_evaluate.filename == '':			
            evaluate_val= 'Please Insert Test Dataset'
            hasil_predict = 'No Prediction Please insert Data'
        else:
            #file_evaluate.json()	
            f = json.load(file_evaluate)
            df_e = f["dataset"]
            dataset_e = pandas.DataFrame(df_e)
            
            # Slice data set in data and labels (2D-array)
            X_test = dataset_e.values[:,0:64].astype(float) # Data
            Y_test = dataset_e.values[:,64].astype(int) # Labels
            evaluate_val = evaluate(X_test, Y_test)
            
            if inp_predict == '':
                hasil_predict = 'No Prediction Please insert Test Dataset'
            else:
                hasil_predict = predict(X_test, inp_predict)

        return render_template("home.html", trainaccure=train_val, evaluateaccure=evaluate_val, hasil_predict=hasil_predict)

    
if __name__ == '__main__':
    app.run(debug=True, port=port, host='0.0.0.0')
