import tensorflow as tf
from pixellib.instance import custom_segmentation
from PIL import Image
import numpy as np
import urllib.request
import base64
import requests
import shutil  # to save it locally
import cv2
import glob
import math


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

    create_mask('raw_image.jpg')

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

    img_pil = Image.open(raw_path)
    img_array = np.array(img_pil)
    # Read mask; OpenCV can't handle indexed images, so we need Pillow here
    # for that, see also: https://stackoverflow.com/q/59839709/11089932
    mask = np.array(Image.open(mask_image))
    mask[mask < 254] = 0
    mask[mask > 0] = 1

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:

        ## create rotated rectangle
        rect = cv2.minAreaRect(c)
        h, w = np.int0(rect[1])
        theta = np.int0(rect[2])
        print(theta)
        print(h,w)
        w_b = w
        h_b = h


        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = sorted(box, key=lambda k: [k[1], k[0]])
        x, y = box[0]

        theta_photo = 0

        if theta != 90:
            theta_photo = theta
            if theta > 45:
                theta_photo = 90 - theta
            length = w * math.cos(math.radians(90- abs(theta)))
            x = x-length

            length_2 = h * math.cos(math.radians(abs(theta)))
            w_b = length + length_2

            length = w * math.sin(math.radians(90 - abs(theta)))
            length_2 = h * math.cos(math.radians(90 -abs(theta)))
            h_b = length + length_2

        if h<w:
            theta_photo = -theta_photo

        im1 = template.rotate(theta_photo, Image.NEAREST, expand=1)

        template_resized = im1.resize((int(w_b), int(h_b)), Image.BICUBIC)
        img_pil.paste(template_resized, (int(x),int(y)), mask=template_resized)

    return img_pil


def create_mask(raw_image):
    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes= 1, class_names= ["BG", "nail"])
    segment_image.load_model("model/mask_rcnn_model.084-0.348825.h5")
    segmask, output = segment_image.segmentImage(raw_image)

    img = Image.open(raw_image)
    width, height = img.size

    image = np.zeros((height, width), dtype=np.uint8)

    for i in range(width):
        for j in range(height):
            if True in segmask['masks'][j][i]:
                image[j][i] = 1

    print(image)

    img = Image.fromarray(np.uint8(image * 255) , 'L')
    img = img.convert("RGB")

    pixdata = img.load()

    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (0, 0, 0, 255):
                pixdata[x, y] = (0, 0, 0, 0)
    img.save("mask_image.png", "PNG")