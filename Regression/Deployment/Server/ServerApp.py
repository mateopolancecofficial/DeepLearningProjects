from typing import Dict
from flask import Flask, request, Response
from joblib import load
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from Deployment.BaseStructures import Model


# set models paths
BASE_MODEL_PATH = '../../TensorflowModels/BaseModel'
EXTENDED_MODEL_PATH = '../../TensorflowModels/ExtendModel'
BASE_MODEL_FT = '../../Transformers/BaseModelFT.pkl'
BASE_MODEL_TT = '../../Transformers/BaseModelTT.pkl'
EXTEND_MODEL_FT = '../../Transformers/ExtendModelFT.pkl'
EXTEND_MODEL_TT = '../../Transformers/ExtendModelTT.pkl'

app = Flask(__name__)


@app.route('/model/score', methods=['POST'])
def score() -> Response:
    """Base model vineyard predictions API endpoint. """
    request_data = request.json
    X = make_features_from_request_data(request_data)
    model = request_data["model"]
    model_output = model_predictions(X, model).tolist()
    response_data = json.dumps({'base_model_predictions': model_output})
    return response_data


def make_features_from_request_data(
    request_data: Dict[str, float]
) -> pd.DataFrame:
    """Create feature array from JSON data parsed as dictionary."""
    X = pd.DataFrame.from_dict(request_data["data"])
    return X


def model_predictions(features: pd.DataFrame, model_p: bool):
    """Return model score for a single instance of feature data."""
    input_dataset = tf.convert_to_tensor(base_ft.transform(features), dtype="float32")
    if model_p == Model.EXTEND.value:
        prediction = model_extend.predict(input_dataset)
    else:
        prediction = model.predict(input_dataset)
    return base_tt.inverse_transform(prediction)


if __name__ == '__main__':
    model = keras.models.load_model(BASE_MODEL_PATH)
    model_extend = keras.models.load_model(EXTENDED_MODEL_PATH)
    base_ft = load(BASE_MODEL_FT)
    base_tt = load(BASE_MODEL_TT)
    extend_ft = load(EXTEND_MODEL_FT)
    extend_tt = load(EXTEND_MODEL_TT)
    print('Models and transformers successfully loaded. ')
    print(f'starting API server')
    app.run(host='0.0.0.0', port=5000)
