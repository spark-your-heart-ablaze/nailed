import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import base64
import requests
import shutil  # to save it locally
import cv2
import glob
import math



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

        #image = cv2.circle(img_array, (x,y), 10, color=(255, 255, 255), thickness=-10)

        #cv2.imshow('image', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #theta = theta - 90
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


            # Start coordinate, here (100, 50)
            # represents the top left corner of rectangle
            start_point = (int(x), int(y))


            # Ending coordinate, here (125, 80)
            # represents the bottom right corner of rectangle
            end_point = (int(x+w_b), int(y+h_b))

            # Black color in BGR
            color = (0, 0, 0)

            # Line thickness of -1 px
            # Thickness of -1 will fill the entire shape
            thickness = 0

            # Using cv2.rectangle() method
            # Draw a rectangle of black color of thickness -1 px
            #image = cv2.rectangle(img_array, start_point, end_point, color, thickness)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)



        if h<w:
            theta_photo = -theta_photo



        im1 = template.rotate(theta_photo, Image.NEAREST, expand=1)
        #im1.show()

        template_resized = im1.resize((int(w_b), int(h_b)), Image.BICUBIC)
        img_pil.paste(template_resized, (int(x),int(y)), mask=template_resized)
        #img_pil.show()

    return img_pil