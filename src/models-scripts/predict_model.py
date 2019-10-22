# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:25:50 2019
@author: Wukong
"""

import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd
from pathlib import Path

PATH_MODELS = Path("../../models")

with open(PATH_MODELS/'rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict')
def predict_iris():
    """ Retorna a Classe que a Iris Pertence
    ---
    parameters:
        - name: s_length
          in: query
          type: number
          required: true
        - name: s_width
          in: query
          type: number
          required: true
        - name: p_lenght
          in: query
          type: number
          required: true
        - name: p_width
          in: query
          type: number
          required: true
    """
    s_lenght = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_lenght = request.args.get("p_lenght")
    p_width = request.args.get("p_width")

    prediction = model.predict(np.array([[s_lenght, s_width, p_lenght, p_width]]))
    return str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    """ Retorna a Classe que a Iris Pertence
    ---
    parameters:
        - name: input_file
          in: formData
          type: file
          required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)