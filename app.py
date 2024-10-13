from flask import Flask, request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

## route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('Home.html')
    else:
        data=CustomData(
            GENDER=request.form.get('GENDER'),
            AGE=float(request.form.get('AGE')),
            SMOKING=request.form.get('SMOKING'),
            ANXIETY=request.form.get('ANXIETY'),
            PEER_PRESSURE=request.form.get('PEER PRESSURE'),
            FATIGUE=request.form.get('FATIGUE'),
            WHEEZING=request.form.get('WHEEZING'),
            ALLERGY=request.form.get('ALLERGY'),
            ALCOHOL_CONSUMING=request.form.get('ALCOHOL CONSUMING'),
            COUGHING=request.form.get('COUGHING'),
            SWALLOWING_DIFFICULTY=request.form.get('SWALLOWING DIFFICULTY'),
            SHORTNESS_OF_BREATH=request.form.get('SHORTNESS OF BREATH'),
            CHEST_PAIN=request.form.get('CHEST PAIN'),
            YELLOW_FINGERS=request.form.get('YELLOW FINGERS'),
            CHRONIC_DISEASE=request.form.get('CHRONIC DISEASE'),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("before prediction")
        predict_pipeline=PredictPipeline()
        print("MId prediction")
        results=predict_pipeline.predict(pred_df)
        print("After prediction")
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)

