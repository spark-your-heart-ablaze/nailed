import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import base64
import requests
from matplotlib import pyplot as plt
import shutil  # to save it locally
import cv2
import glob



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
    print(data.split(';'))
    data = data.split(';')
    start = '<a href=\"'
    end = "\">Ссылка</a>"
    template_number = data[1]
    data = data[0][len(start):-len(end)]

    print(data)

    r = requests.get(data, stream=True)
    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
    r.raw.decode_content = True

    # Open a local file with wb ( write binary ) permission.
    with open("raw_image.jpg", 'wb') as f:
        shutil.copyfileobj(r.raw, f)

    img = Image.open("raw_image.jpg")
    width, height = img.size

    img_array = np.array(img.resize((192, 160)))

    array = []
    array.append(img_array)

    X_f = np.array(array).astype('float32')
    X_f /= 255
    mask = model.predict(X_f, batch_size=4, verbose=0)

    mask = np.uint8(mask[0, :, :, 0] * 255)

    img_mask = Image.fromarray(np.uint8(mask), 'L')
    img_mask = img_mask.resize((width, height), Image.BICUBIC)
    img_mask = np.array(img_mask)
    img_mask[img_mask < 10] = 0
    img_mask[img_mask >= 10] = 255
    img_mask = Image.fromarray(np.uint8(img_mask), 'L')
    img_mask = img_mask.convert("RGBA")

    pixdata = img_mask.load()

    width, height = img_mask.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (0, 0, 0, 255):
                pixdata[x, y] = (0, 0, 0, 0)

    # Overlay foreground onto background at top right corner, using transparency of foreground as mask
    #img.paste(img_mask, mask=img_mask)
    img_mask.save('mask_image.png', 'PNG')

    template_name = glob.glob('model/nail_templates/LUXIO_'+template_number+'*')[0]
    img_with_template = equip_template(template_name, 'raw_image.jpg', 'mask_image.png')
    img_with_template.save('geeks.png', 'PNG')


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


def equip_template(template_path,raw_path,mask_image):
    #Read template image
    template = Image.open(template_path)

    # Read color image
    img = cv2.imread(raw_path)
    img_pil = Image.open(raw_path)
    # Read mask; OpenCV can't handle indexed images, so we need Pillow here
    # for that, see also: https://stackoverflow.com/q/59839709/11089932
    mask = np.array(Image.open(mask_image))
    mask = mask.mean(axis=2)
    mask[mask < 254] = 0
    mask[mask > 0] = 1

    binary_map = (mask > 0).astype(np.uint8)

    connectivity = 4 # or whatever you prefer

    output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)

    mask = np.array(output[1]).astype('float32')

    ## Iterate all colors in mask
    for color in np.unique(mask):
        if color == np.unique(mask)[-1]:
            continue
        # Color 0 is assumed to be background or artifacts
        if color == 0:
            continue

        # Determine bounding rectangle w.r.t. all pixels of the mask with
        # the current color
        x, y, w, h = cv2.boundingRect(np.uint8(mask == color))

        template_resized = template.resize((w, h), Image.BICUBIC)
        img_pil.paste(template_resized, (x,y), mask=template_resized)

    return img_pil