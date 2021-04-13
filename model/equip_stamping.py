'''

Trying to place the template of the nail to our detected mask

'''

import cv2
import cv2 as cv
import numpy as np
from PIL import Image
import math
import base64
import requests
import os

def equip_template(raw_path,template_path, stamping, color,user_id, counter):
    if (stamping == '0' or stamping == '1') and color == '0':
        path_orig_photo = "model/orig/" + raw_path
        path_mask_photo = "model/mask/" + os.path.splitext(raw_path)[0] + ".png"
        template_path = "model/stamping/" + template_path

        #Read template image
        template = Image.open(template_path)

        # Read color image

        img_pil = Image.open(path_orig_photo)
        img_array = np.array(img_pil)
        # Read mask; OpenCV can't handle indexed images, so we need Pillow here
        # for that, see also: https://stackoverflow.com/q/59839709/11089932
        mask = np.array(Image.open(path_mask_photo))
        mask[mask < 254] = 0
        mask[mask > 0] = 1


        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = Image.open(path_mask_photo)

        width, height = mask.size

        for c in cnts:

            rows, cols = img_array.shape[:2]
            [vx, vy, x, y] = cv.fitLine(c, cv.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv.line(img_array, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)


            ## create rotated rectangle
            rect = cv2.minAreaRect(c)
            h, w = np.int0(rect[1])
            theta = np.int0(rect[2])

            w_b = w
            h_b = h


            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv.drawContours(img_array, [box], 0, (0, 0, 255), 2)

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
                image = cv2.rectangle(img_array, start_point, end_point, color, thickness)
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


            if h<w:
                theta_photo = -theta_photo



            im1 = template.rotate(theta_photo, Image.NEAREST, expand=1)
            #im1.show()

            template_resized = im1.resize((int(w_b), int(h_b)), Image.BICUBIC)

            #im_mask = template_resized
            #im_mask = im_mask.convert("L")
#
            #pixdata = im_mask.load()
            #width, height = im_mask.size
            #for y in range(height):
            #    for x in range(width):
            #        print(pixdata[x, y])
            #        if pixdata[x, y] > 0:
            #            pixdata[x, y] = 200


            mask.paste(template_resized, (int(x),int(y)), mask=template_resized)
            #img_pil.show()

        cutout = Image.open(path_mask_photo)
        pixdata = cutout.load()

        for y in range(height):
            for x in range(width):
                #make the background transparent
                if pixdata[x, y] == (0, 0, 0, 0):
                    pixdata[x, y] = (0, 0, 0, 255)

        mask.paste(cutout,(0,0),mask=cutout)

        pixdata = mask.load()

        for y in range(height):
            for x in range(width):
                #make the background transparent
                if pixdata[x, y] == (0, 0, 0, 255):
                    pixdata[x, y] = (0, 0, 0, 0)

        mask.save("model/equiped_stamping/" + os.path.splitext(raw_path)[0] + ".png")
        img_pil.paste(mask,(0,0), mask=mask)

        img_pil.save(user_id + '.jpg')

        import cloudinary
        cloudinary.config(
            cloud_name='dg0qyrbbh',
            api_key='193157247241951',
            api_secret='brazq0NfMbQDvVh_y56nb24oY_A'
        )
        result = cloudinary.uploader.upload(
            user_id + '.jpg',
            upload_preset="ml_default", public_id=user_id + "_" + counter)
        #with open("nail.png", "rb") as file:
        #    url = "https://api.imgbb.com/1/upload"
        #    payload = {
        #        "key": '25502214673ff70e70aacc9157b3084e',
        #        "image": base64.b64encode(file.read()),
        #    }
        #    res = requests.post(url, payload)
#
        # You may want to further format the prediction to make it more
        # human readable
        return 1

    if (stamping == '0' or stamping == '1')  and color == '1':
        path_orig_photo = "model/orig/" + raw_path
        path_mask_photo = "model/mask/" + os.path.splitext(raw_path)[0] + ".png"
        template_path = "model/stamping/" + template_path
        color_path = "model/equiped_color/" + os.path.splitext(raw_path)[0] + ".png"
        img_color = Image.open(color_path)





        #Read template image
        template = Image.open(template_path)

        # Read color image

        img_pil = Image.open(path_orig_photo)
        img_pil.paste(img_color, (0, 0), mask=img_color)

        img_array = np.array(img_pil)
        # Read mask; OpenCV can't handle indexed images, so we need Pillow here
        # for that, see also: https://stackoverflow.com/q/59839709/11089932
        mask = np.array(Image.open(path_mask_photo))
        mask[mask < 254] = 0
        mask[mask > 0] = 1


        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = Image.open(path_mask_photo)

        width, height = mask.size

        for c in cnts:

            rows, cols = img_array.shape[:2]
            [vx, vy, x, y] = cv.fitLine(c, cv.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv.line(img_array, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)


            ## create rotated rectangle
            rect = cv2.minAreaRect(c)
            h, w = np.int0(rect[1])
            theta = np.int0(rect[2])

            w_b = w
            h_b = h


            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv.drawContours(img_array, [box], 0, (0, 0, 255), 2)

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
                image = cv2.rectangle(img_array, start_point, end_point, color, thickness)
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


            if h<w:
                theta_photo = -theta_photo



            im1 = template.rotate(theta_photo, Image.NEAREST, expand=1)
            #im1.show()

            template_resized = im1.resize((int(w_b), int(h_b)), Image.BICUBIC)

            #im_mask = template_resized
            #im_mask = im_mask.convert("L")
#
            #pixdata = im_mask.load()
            #width, height = im_mask.size
            #for y in range(height):
            #    for x in range(width):
            #        print(pixdata[x, y])
            #        if pixdata[x, y] > 0:
            #            pixdata[x, y] = 200


            mask.paste(template_resized, (int(x),int(y)), mask=template_resized)
            #img_pil.show()

        cutout = Image.open(path_mask_photo)
        pixdata = cutout.load()

        for y in range(height):
            for x in range(width):
                #make the background transparent
                if pixdata[x, y] == (0, 0, 0, 0):
                    pixdata[x, y] = (0, 0, 0, 255)

        mask.paste(cutout,(0,0),mask=cutout)

        pixdata = mask.load()

        for y in range(height):
            for x in range(width):
                #make the background transparent
                if pixdata[x, y] == (0, 0, 0, 255):
                    pixdata[x, y] = (0, 0, 0, 0)


        img_pil.paste(mask,(0,0), mask=mask)

        img_pil.save(user_id + '.jpg')

        import cloudinary
        cloudinary.config(
            cloud_name='dg0qyrbbh',
            api_key='193157247241951',
            api_secret='brazq0NfMbQDvVh_y56nb24oY_A'
        )
        result = cloudinary.uploader.upload(
            user_id + '.jpg',
            upload_preset="ml_default", public_id=user_id + "_" + counter)

        #with open("nail.png", "rb") as file:
        #    url = "https://api.imgbb.com/1/upload"
        #    payload = {
        #        "key": '25502214673ff70e70aacc9157b3084e',
        #        "image": base64.b64encode(file.read()),
        #    }
        #    res = requests.post(url, payload)

        # You may want to further format the prediction to make it more
        # human readable
        return 1

