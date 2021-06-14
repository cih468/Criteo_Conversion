from flask import Flask,render_template  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
import pandas as pd
import tensorflow as tf
from flask import request
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import preprocess
import os
from threading import Thread
from datetime import datetime
import time

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

model = preprocess.model
model_path = preprocess.model_path
model_update_time = preprocess.model_update_time

@app.route('/html/cvr.html')
def basic():
    return render_template("./cvr.html")

@app.route('/update/model')
def updateModel():
    global l1_reg
    preprocess.model = tf.keras.models.load_model(model_path,custom_objects={'l1_reg': l1_reg})
    preprocess.model_update_time=os.path.getmtime(model_path)

    return render_template("./cvr.html")

@api.route('/get/cvr',methods=['GET'])
class getCvr(Resource):
    def get(self):
        cvr = 0
        line = request.args.get('line','')
        data = line.split('\\t')
        df = preprocess.makeDataFrame(data)
        df = preprocess.makeTimeFeature(df)

        df_preprocess = preprocess.totalProcess(df)
        # df = preprocess.missingValueProcessCategorical(df)
        # df = preprocess.missingValueProcessInteger(df)

        # df_preprocess = pd.DataFrame()
        # df_preprocess = preprocess.ohe(df,df_preprocess,use_saved_columns=True)
        # df_preprocess = pd.concat([df['nb_clicks_1week'],df_preprocess],axis=1)
        # df_preprocess = df_preprocess.astype(float)
        # df_preprocess = preprocess.normalization(df_preprocess)
        
        cvr = preprocess.model.predict(df_preprocess)[0][0]

        return {
            'cvr': str(cvr)
        }

## threading.Thread를 상속받는 클래스를 만들어서 run하여 객체를 생성한다.
def check_model_change(val):
    global l1_reg

    while True:
        if preprocess.model_update_time!=os.path.getmtime(model_path) :
            print("update")
            tf.keras.models.load_model(model_path,custom_objects={'l1_reg': l1_reg})
            preprocess.model_update_time=os.path.getmtime(model_path)
            time.sleep(1)
        time.sleep(1)
    

if __name__ == '__main__':
    global l1_reg
    preprocess.model = model = tf.keras.models.load_model(model_path,custom_objects={'l1_reg': preprocess.l1_reg})
    print('model_update_time : ',model_update_time)

    t1 = Thread(target = check_model_change, args=(1,))
    t1.start()

    app.run(debug=True,port=5000)