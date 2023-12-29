import numpy as np
from tensorflow.keras.models import Model
from .ssp import ssp
from . import svm



def predict(input_x, THRES, cnn_model, svm_model, bbox_regressor):
    ret = []
    objectP = input_x[np.newaxis, :].copy()
    ssp_x, ssp_rects = ssp(objectP, input_x.shape, input_x.shape)
    
    cnn_model_svm = Model(inputs = cnn_model.input, outputs = cnn_model.layers[4].output)
    cnn_model_box = Model(inputs = cnn_model.input, outputs = cnn_model.layers[3].output)
    flatten_svm = cnn_model_svm.predict(ssp_x)
    flatten_box = cnn_model_box.predict(ssp_x)
    
    svm_pred = svm.predict(svm_model, flatten_svm, THRES)
    
    for pred in svm_pred:
        ss = pred['ss']
        x, y, w, h = ssp_rects[ss,:]
        box_input = flatten_box[ss].reshape(-1)
        x = int(x + np.matmul(box_input, bbox_regressor['x']) * input_x.shape[0])
        y = int(y + np.matmul(box_input, bbox_regressor['y']) * input_x.shape[1])
        
        ret.append({'svm':pred, 'x':x, 'y':y, 'w':w, 'h':h})
        
    return ret

