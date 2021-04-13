from flask import Flask, request

from model import equip_color, equip_stamping
from model.predict import predict
# from model import equip
from pixellib.instance import custom_segmentation

import tensorflow


# create the flask object
app = Flask(__name__)

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=1, class_names=["BG", "nail"])
segment_image.load_model("model/mask_rcnn_model.067-0.335795.h5")


@app.route('/')
def index():
    return "Index Page" + tensorflow.keras.__version__


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    data = request.form.get('data')
    user_id = request.form.get('user_id')
    counter = request.form.get('counter')
    segment_image.load_model("model/mask_rcnn_model.067-0.335795.h5")
    if data == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = predict(segment_image, data, user_id, counter)
    return str(prediction)


@app.route('/equip', methods=['GET', 'POST'])
def equip():
    name = request.form.get('data')
    template_number = request.form.get('color')
    user_id = request.form.get('user_id')
    counter = request.form.get('counter')
    stamping = request.form.get('stamping_condition')
    color = request.form.get('color_condition')
    if name == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = equip_color.equip(name, template_number, user_id, counter, stamping, color)
    return str(prediction)


@app.route('/equip_stamp', methods=['GET', 'POST'])
def equip_stamp():
    name = request.form.get('data')
    stamping_name = request.form.get('stamping')
    stamping = request.form.get('stamping_condition')
    color = request.form.get('color_condition')
    user_id = request.form.get('user_id')
    counter = request.form.get('counter')
    if name == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = equip_stamping.equip_template(name, stamping_name, stamping, color, user_id, counter)
    return str(prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
