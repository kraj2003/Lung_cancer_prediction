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
            GENDER=request.form.get('Gender'),
            AGE=float(request.form.get('Age')),
            SMOKING=request.form.get('Smoking'),
            YELLOW_FINGERS=request.form.get('Yellow fingers'),
            ANXIETY=request.form.get('Anxiety'),
            PEER_PRESSURE=request.form.get('Peer_pressure'),
            CHRONIC_DISEASE=request.form.get('Chronic Disease'),
            FATIGUE=request.form.get('Fatigue'),
            ALLERGY=request.form.get('Allergy'),
            WHEEZING=request.form.get('Wheezing'),
            ALCOHOL_CONSUMING=request.form.get('Alcohol'),
            COUGHING=request.form.get('Coughing'),
            SHORTNESS_OF_BREATH=request.form.get('Shortness of Breath'),
            SWALLOWING_DIFFICULTY=request.form.get('Swallowing Difficulty'),
            CHEST_PAIN=request.form.get('Chest pain'),
        # GENDER=float(request.form.get('Gender'))
        # AGE=float(request.form.get('Age'))
        # SMOKING=float(request.form.get('Smoking'))
        # YELLOW_FINGERS=float(request.form.get('Yellow fingers'))
        # ANXIETY=float(request.form.get('Anxiety'))
        # PEER_PRESSURE=float(request.form.get('Peer_pressure'))
        # CHRONIC_DISEASE=float(request.form.get('Chronic Disease'))
        # FATIGUE=float(request.form.get('Fatigue'))
        # ALLERGY=float(request.form.get('Allergy'))
        # WHEEZING=float(request.form.get('Wheezing'))
        # ALCOHOL_CONSUMING=float(request.form.get('Alcohol'))
        # COUGHING=float(request.form.get('Coughing'))
        # SHORTNESS_OF_BREATH=float(request.form.get('Shortness of Breath'))
        # SWALLOWING_DIFFICULTY=float(request.form.get('Swallowing Difficulty'))
        # CHEST_PAIN=float(request.form.get('Chest pain'))
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

