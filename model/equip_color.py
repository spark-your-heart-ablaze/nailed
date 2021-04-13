from PIL import Image
import glob
import numpy as np
import cv2
import base64
import requests
import os
from PIL import Image, ImageFilter

#here we recieve the name as data and number of the color. We can find the mask original file and a color with this function
def equip(name,template_number, user_id, counter, stamping, color):
    if stamping == '0' and (color == '0' or color == '1'):
        #template_number = int(template_number)
        #template_number = str("{0:03}".format(template_number))
        template_name = glob.glob('model/nail_templates/'+template_number+'*')[0]

        with open(template_name) as f:
            color = f.readline().split()

        path_orig_photo = "model/orig/" + os.path.splitext(os.path.basename(name))[0] + '.jpg'
        path_mask_photo = "model/mask/" + os.path.splitext(os.path.basename(name))[0] + '.png'

        img = Image.open(path_mask_photo)
        width, height = img.size
        pixdata = img.load()

        for y in range(height):
            for x in range(width):

                #change the color of the mask
                if pixdata[x,y] == (255,255,255,0):
                    pixdata[x, y] = (int(color[0]), int(color[1]), int(color[2]), 200)

        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img_orig = Image.open(path_orig_photo)

        img.save("model/equiped_color/" + os.path.splitext(os.path.basename(name))[0] + '.png')

        img_orig.paste(img, mask=img)

        # exporting the image
        img_orig.save(user_id + '.jpg')

        import cloudinary
        cloudinary.config(
            cloud_name='dg0qyrbbh',
            api_key='193157247241951',
            api_secret='brazq0NfMbQDvVh_y56nb24oY_A'
        )
        result = cloudinary.uploader.upload(
            user_id + '.jpg',
            upload_preset="ml_default", public_id= user_id + "_" + counter)

        #with open("nail.png", "rb") as file:
        #    url = "https://api.imgbb.com/1/upload"
        #    payload = {
        #        "key": '25502214673ff70e70aacc9157b3084e',
        #        "image": base64.b64encode(file.read()),
        #    }
        #    res = requests.post(url, payload)

        # You may want to further format the prediction to make it more
        # human readable
        #return res.json()['data']['url']
        return 1

    if stamping == '1' and (color == '1' or color == '0'):
        #template_number = int(template_number)
        #template_number = str("{0:03}".format(template_number))
        template_name = glob.glob('model/nail_templates/'+template_number+'*')[0]

        with open(template_name) as f:
            color = f.readline().split()

        path_orig_photo = "model/orig/" + os.path.splitext(os.path.basename(name))[0] + '.jpg'
        path_mask_photo = "model/mask/" + os.path.splitext(os.path.basename(name))[0] + '.png'
        stamping_path = "model/equiped_stamping/" + os.path.splitext(name)[0] + ".png"
        img_stamping = Image.open(stamping_path)

        img = Image.open(path_mask_photo)
        width, height = img.size
        pixdata = img.load()

        for y in range(height):
            for x in range(width):

                #change the color of the mask
                if pixdata[x,y] == (255,255,255,0):
                    pixdata[x, y] = (int(color[0]), int(color[1]), int(color[2]), 200)

        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img_orig = Image.open(path_orig_photo)

        img.save("model/equiped_color/" + os.path.splitext(os.path.basename(name))[0] + '.png')

        img_orig.paste(img, mask=img)
        img_orig.paste(img_stamping, mask=img_stamping)

        img_orig.save(user_id + '.jpg')


        print(user_id)
        print(counter)
        print(name)
        import cloudinary
        cloudinary.config(
            cloud_name='dg0qyrbbh',
            api_key='193157247241951',
            api_secret='brazq0NfMbQDvVh_y56nb24oY_A'
        )
        result = cloudinary.uploader.upload(
            user_id + '.jpg',
            upload_preset="ml_default", public_id= user_id + "_" + counter)

        #with open("nail.png", "rb") as file:
        #    url = "https://api.imgbb.com/1/upload"
        #    payload = {
        #        "key": '25502214673ff70e70aacc9157b3084e',
        #        "image": base64.b64encode(file.read()),
        #    }
        #    res = requests.post(url, payload)

        # You may want to further format the prediction to make it more
        # human readable
        #return res.json()['data']['url']
        return 1

