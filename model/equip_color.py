from PIL import Image
import glob
import numpy as np
import cv2
import base64
import requests
import os

#here we recieve the name as data and number of the color. We can find the mask original file and a color with this function
def equip(name,template_number, user_id, counter):
    template_number = int(template_number)
    template_number = str("{0:03}".format(template_number))
    template_name = glob.glob('model/nail_templates/LUXIO_'+template_number+'*')[0]

    with open(template_name) as f:
        color = f.readline().split()

    path_orig_photo = "model/orig/" + name
    path_mask_photo = "model/mask/" + name

    img = Image.open(path_mask_photo)
    width, height = img.size
    pixdata = img.load()

    for y in range(height):
        for x in range(width):

            #change the color of the mask
            if pixdata[x,y] == (255,255,255,0):
                pixdata[x, y] = (int(color[0]), int(color[1]), int(color[2]), 230)

    img_orig = Image.open(path_orig_photo)
    img_orig.paste(img, mask=img)
    img_orig.save("nail.png")

    im = Image.open("nail.png")
    # converting to jpg
    rgb_im = im.convert("RGB")
    # exporting the image
    rgb_im.save('nail.jpg')

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
        "nail.jpg",
        upload_preset="ml_default", public_id= user_id + "_" + counter + "_" + name)

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

