from PIL import Image
import numpy as np
import shutil  # to save it locally
import cv2
from pixellib.instance import custom_segmentation
from scipy.interpolate import splprep, splev
import os
import cloudinary.uploader
import requests
import tensorflow as tf

import PIL.Image
import PIL.ImageOps
import numpy as np


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

class Segment_model(object):
    def __init__(self, filepath="model/mask_rcnn_model.067-0.335795.h5"):
        self.segment_image = custom_segmentation()
        self.segment_image.inferConfig(num_classes=1, class_names=["BG", "nail"])

        graph = tf.compat.v1.get_default_graph()
        self.segment_image.load_model(filepath)
        self.segment_image.model.keras_model._make_predict_function()

def create_mask(segment_image, raw_image):
    #self.segment_image.model.keras_model._make_predict_function()
    segmask, _ = segment_image.segmentImage(raw_image)

    # create image from array
    img = Image.open(raw_image)
    width, height = img.size

    image = np.zeros((height, width), dtype=np.uint8)

    for i in range(width):
        for j in range(height):
            if True in segmask['masks'][j][i]:
                image[j][i] = 1

    mask = image.astype(np.uint8)
    # Finds contours
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    smoothened = []
    for contour in cnts:
        cnt = contour
        hull = cv2.convexHull(cnt)

        # print(hull)
        x, y = hull.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x, y], u=None, s=1, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))

        # cv2.drawContours(image, [hull], -1, (255, 0, 255), thickness=-1)

    cv2.drawContours(mask, smoothened, -1, (1, 1, 1), -1)

    # Enlarge the mask
    dilatation_size = 5
    # Options: cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE
    dilatation_type = cv2.MORPH_CROSS
    element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    mask = cv2.dilate(mask, element)

    img = Image.fromarray(np.uint8(mask * 255), 'L')
    img = img.convert("RGBA")

    pixdata = img.load()
    for y in range(height):
        for x in range(width):
            # make the background transparent
            if pixdata[x, y] == (0, 0, 0, 255):
                pixdata[x, y] = (0, 0, 0, 0)

            # make mask fully untransparent
            if pixdata[x, y][3] > 10:
                pixdata[x, y] = (pixdata[x, y][0], pixdata[x, y][1], pixdata[x, y][2], 0)

            #if pixdata[x, y][3] != (0,0,0,0):
            #    pixdata[x, y] = (255, 255, 255, 255)

    return img

def predict(segment_image, data, user_id, counter):
    ###  IMAGE PREPROCESSING  ###
    # |
    # |
    #  data-->preprocessed_data #
    # |
    # |
    ###          END          ###
    # print(data)

    r = requests.get(data, stream=True)
    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
    r.raw.decode_content = True
    path_orig_photo = "model/orig/" + os.path.basename(data)
    path_orig_photo_noextension = "model/orig/" + os.path.splitext(os.path.basename(data))[0]
    path_mask_photo = "model/mask/" + os.path.splitext(os.path.basename(data))[0] + '.png'

    # 3print(path_orig_photo)
    # Open a local file with wb ( write binary ) permission.
    with open(path_orig_photo, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    # importing the image
    im = Image.open(path_orig_photo)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        im = PIL.ImageOps.exif_transpose(im)
    else:
        # Otherwise, do the exif transpose ourselves
        im = exif_transpose(im)

    basewidth = 1280
    #img = Image.open('somepic.jpg')
    wpercent = (basewidth / float(im.size[0]))
    hsize = int((float(im.size[1]) * float(wpercent)))
    im = im.resize((basewidth, hsize), Image.ANTIALIAS)
    #img.save('somepic.jpg')

    # converting to jpg
    rgb_im = im.convert("RGB")
    # exporting the image
    rgb_im.save(path_orig_photo_noextension + '.jpg')
    # os.remove(path_orig_photo)
    path_orig_photo = path_orig_photo_noextension + '.jpg'
    print(path_orig_photo)
    img = create_mask(segment_image, path_orig_photo)
    img.save(path_mask_photo, "PNG")
    # Cloudinary settings using python code. Run before pycloudinary is used.
    import cloudinary
    cloudinary.config(
        cloud_name='dg0qyrbbh',
        api_key='193157247241951',
        api_secret='brazq0NfMbQDvVh_y56nb24oY_A'
    )
    result = cloudinary.uploader.upload(
        path_orig_photo,
        upload_preset="ml_default",
        public_id=user_id + "_" + counter)
    # with open("geeks.png", "rb") as file:
    #    url = "https://api.imgbb.com/1/upload"
    #    payload = {
    #        "key": '25502214673ff70e70aacc9157b3084e',
    #        "image": base64.b64encode(file.read()),
    #    }
    #    res = requests.post(url, payload)
    # You may want to further format the prediction to make it more
    # human readable
    return 1



