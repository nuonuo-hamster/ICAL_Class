import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.externals import joblib

from . import load_data, ssp


def predict(models, inputs, THRES):
    pred_svm = []

    for i in range(len(models)):
        max_prob = 0
        max_label = 0
        pred = models[i].predict(inputs)[0]
        prob = models[i].predict_proba(inputs)[:, 1]
        for n, pd in enumerate(prob):
            if pd > max_prob:
                max_prob = pd
                max_label = n
        if max_prob > THRES:
            pred_svm.append({'ss':max_label, 'label':i, 'prob':max_prob})
    return pred_svm

# 定義精確度計算函數 get_acc(模型，資料集屬性，資料集類別)
def get_acc(model, atts, labels):
    # 初始化得分數
    score = 0    
    # 比對每筆資料集預測結果與類別是否吻合
    print('label \t pred \t prob')
    for i in range(atts.shape[0]):
        # 獲取預測結果
        response = model.predict(atts[i])[0]
        prob = model.predict_proba(atts[i])[0, response]
        # 印出部份資料集查看比對狀況
        if i % int(atts.shape[0] / 10.0) == 0: print(labels[i], '\t', response, '\t', prob)            
        # 若預測結果與類別吻合則累加得分數
        if model.predict(atts[i])[0] == labels[i]:
            score = score + 1
    
    # 將總得分數除於總數量獲得精確度
    acc = score / atts.shape[0]
    print('==================')        
    print('accuracy: ', acc)