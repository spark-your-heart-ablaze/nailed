from PIL import Image
import numpy as np
import urllib.request
import base64
import requests
import shutil  # to save it locally
import cv2
import glob
import math
from pixellib.instance import custom_segmentation
from scipy.interpolate import splprep, splev
import os


class Segment_model(object):
    def __init__(self, filepath="model/mask_rcnn_model.067-0.335795.h5"):
        self.segment_image = custom_segmentation()
        self.segment_image.inferConfig(num_classes=1, class_names=["BG", "nail"])
        self.segment_image.load_model(filepath)

    def create_mask(self, raw_image):

        segmask, _ = self.segment_image.segmentImage(raw_image)

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
        dilatation_size = 1
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

        return img




    def predict(self, data):
        ###  IMAGE PREPROCESSING  ###
        # |
        # |
        #  data-->preprocessed_data #
        # |
        # |
        ###          END          ###
        print(data)
        r = requests.get(data, stream=True)
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        path_orig_photo = "model/orig/" + os.path.basename(data)
        path_mask_photo = "model/mask/" + os.path.basename(data)
        # Open a local file with wb ( write binary ) permission.
        with open(path_orig_photo, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        img = self.segment_image.create_mask(path_orig_photo)

        img.save(path_mask_photo, "PNG")

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



