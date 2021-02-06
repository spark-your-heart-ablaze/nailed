from flask import Flask, request
from model import predict
import tensorflow
# create the flask object
app = Flask(__name__)


@app.route('/')
def index():
    return "Index Page" + tensorflow.keras.__version__


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    data = request.form.get('data')
    if data == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = predict.predict(data)
    return str(prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
