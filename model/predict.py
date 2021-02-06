import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import base64
import requests
from matplotlib import pyplot as plt
import shutil  # to save it locally

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

dependencies = {
     'mean_iou': mean_iou
}

model = tf.keras.models.load_model('model/nailed.h5', custom_objects=dependencies)


def predict(data):
    ###  IMAGE PREPROCESSING  ###
    #|
    #|
    #  data-->preprocessed_data #
    #|
    #|
    ###          END          ###
    #example "<a href=\"https://cp.puzzlebot.top\">Ссылка</a>"
    #%3Ca%20href%3D%22 https%3A%2F%2Fcp.puzzlebot.top%2Fchat_statistics%2Ftg_get_file.php%3Fbot_username%3Dnailed_bot%26answer_id%3D1143899%26form_id%3DWH98ARN1P7MJ8VH0%26person_id%3D359082325%26file_id%3DAgACAgIAAxkBAAICUWAenbrBERFv_39gtHOCMrKLHM2dAAJisTEbHRzwSLGUjJ8Xkb4AAaQHbZcuAAMBAAMCAAN5AAOixAYAAR4E%26gr%3D1535990968% 22%3E%D0%A1%D1%81%D1%8B%D0%BB%D0%BA%D0%B0%3C%2Fa%3E
    print(data)
    start = '<a href=\"'
    end = "\">Ссылка</a>"
    data = data[len(start):-len(end)]
    print(data)

    r = requests.get(data, stream=True)
    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
    r.raw.decode_content = True

    # Open a local file with wb ( write binary ) permission.
    with open("raw_image.jpg", 'wb') as f:
        shutil.copyfileobj(r.raw, f)

    img = Image.open("raw_image.jpg")
    img_array = np.array(img.resize((192, 160)))

    background = Image.fromarray(img_array)

    array = []
    array.append(img_array)

    X_f = np.array(array).astype('float32')
    X_f /= 255
    mask = model.predict(X_f, batch_size=4, verbose=0)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_array)
    plt.imshow(mask[0, :, :, 0], alpha=0.3)
    plt.savefig('geeks.png')

    #mask_image = Image.fromarray(mask[0,:,:,0]).convert('RGB')
    #new_img = Image.blend(background, mask_image, 0.5)
    #new_img.save("geeks.png")

    with open("geeks.png", "rb") as file:
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": '25502214673ff70e70aacc9157b3084e',
            "image": base64.b64encode(file.read()),
        }
        res = requests.post(url, payload)



    # You may want to further format the prediction to make it more
    # human readable
    return res.json()['data']['url']