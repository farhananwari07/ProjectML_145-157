import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
datahasil = os.listdir('static/result/')

print("Init Flask App")
app = Flask(__name__)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/")
def f_074_compare():
    return render_template('/compare/compare.html', )

@app.route('/1/pred_comp', methods=['POST'])
def f_074_predict_compare():

    respon_model = []
    running_time = []
    chosen_model = request.form.getlist('select_model')
    filename = request.form.get('input_image')
    img = cv2.cvtColor(np.array(np.array(Image.open(filename))), cv2.COLOR_BGR2RGB)
    model_dict = {'maxpol' :   'static/model/model/CNNModelMax.h5',
                  'avpol'   :  'static/model/model/CNNModelAvgV2.h5',
                  'avpoladel'   :   'static/model/model/CNNModelAvg.h5',
                  'ResNet'   :   'static/model/model/ModelResNet50.h5'
                  }

    for m in chosen_model:
        if "_js" in m:
            json_file = open(model_dict[m][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[m][1])
        else:
            model = load_model(model_dict[m])
        
        imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
        start = time.time()
        pred = model.predict(imgs)[0]
        running_time.append(round(time.time()-start,4))
        respon_model.append([round(elem * 100, 2) for elem in pred])    
    
    return f_074_predict_result_compare(respon_model, chosen_model, running_time, filename[7:])

@app.route('/1/pred_comps', methods=['POST'])
def f_074_predicts_compare():

    respon_model = []
    running_time = []
    chosen_model = request.form.getlist('select_model')
    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))
    img = cv2.cvtColor(np.array(np.array(Image.open(file))), cv2.COLOR_BGR2RGB)
    model_dict = {'maxpol' :   'static/model/model/CNNModelMax.h5',
                  'avpol'   :  'static/model/model/CNNModelAvgV2.h5',
                  'avpoladel'   :   'static/model/model/CNNModelAvg.h5',
                  'ResNet'   :   'static/model/model/ModelResNet50.h5' 
                  }
    for m in chosen_model:
        if "_js" in m:
            json_file = open(model_dict[m][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[m][1])
        else:
            model = load_model(model_dict[m])

        imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
        start = time.time()
        pred = model.predict(imgs)[0]  
        running_time.append(round(time.time()-start,4))
        respon_model.append([round(elem * 100, 2) for elem in pred])  

    return f_074_predict_result_compare(respon_model, chosen_model, running_time, 'temp.jpg')

def f_074_predict_result_compare(probs, mdl, run_time, img):
    class_list = {'Positif':'Positif','Negatif':'Negatif'} 
    idx_pred = [i.index(max(i)) for i in probs]
    labels = list(class_list.keys())
    return render_template('/compare/result_compare.html', labels=labels, 
                            probs=probs, mdl=mdl, run_time=run_time, pred=idx_pred, img=img)

# Select Model f_201710370311285 Brain Tumor Disease
@app.route("/")
def f_074_select():
    return render_template('/compare/select.html', )

@app.route('/1/pred_select', methods=['POST'])
def f_074_predict_select():
    
    chosen_model = request.form['select_model']
    model_dict = {'maxpol' :   'static/model/model/CNNModelMax.h5',
                  'avpol'   :  'static/model/model/CNNModelAvgV2.h5',
                  'avpoladel'   :   'static/model/model/CNNModelAvg.h5',
                  'ResNet'   :   'static/model/model/ModelResNet50.h5'
                  }

    if chosen_model in model_dict:
        if "_js" in chosen_model:
            json_file = open(model_dict[chosen_model][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[chosen_model][1])
        else:
            model = load_model(model_dict[chosen_model]) 
    else:
        model = load_model(model_dict[0])
    
    filename = request.form.get('input_image')
    img = cv2.cvtColor(np.array(Image.open(filename)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(img)[0]
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]

    return f_074_predict_result_select(chosen_model, runtimes, respon_model, filename[7:])

@app.route('/1/pred_selects', methods=['POST'])
def f_074_predicts_select():

    chosen_model = request.form['select_model']
    model_dict = {'maxpol' :   'static/model/model/CNNModelMax.h5',
                  'avpol'   :  'static/model/model/CNNModelAvgV2.h5',
                  'avpoladel'   :   'static/model/model/CNNModelAvg.h5',
                  'ResNet'   :   'static/model/model/ModelResNet50.h5'
                  }

    if chosen_model in model_dict:
        if "_js" in chosen_model:
            json_file = open(model_dict[chosen_model][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[chosen_model][1])
        else:
            model = load_model(model_dict[chosen_model]) 
    else:
        model = load_model(model_dict[0])

    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(img)[0]
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]

    return f_074_predict_result_select(chosen_model, runtimes, respon_model, 'temp.jpg')

def f_074_predict_result_select(model, run_time, probs, img):
    class_list = {'Positif':'Positif','Negatif':'Negatif'}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/compare/result_compare.html', labels=labels, 
                            probs=probs, model=model, pred=idx_pred, 
                            run_time=run_time, img=img)

if __name__ == "__main__": 
        app.run(debug=True, host='0.0.0.0', port=2000)