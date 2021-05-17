from flask import Flask, request

from model import equip_color, equip_stamping
from model.predict import predict
from instance import custom_segmentation
#from pixellib.instance import custom_segmentation
from model import database

import tensorflow

from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from starlette import status
from pydantic import BaseModel, EmailStr
from typing import Optional
import re
import pandas as pd
import uvicorn


#Main app
app = FastAPI(
    title="Nailed IT",
    description="API для фронта")

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create the flask object

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=1, class_names=["BG", "nail"])
segment_image.load_model("model/mask_rcnn_model.067-0.335795.h5")

class mask_out(BaseModel):
    mask_photo: str

@app.post('/prediction', summary="Создание маски ногтей с фотографии, которую загрузил пользователь")
async def prediction(Photo:str = Query(..., description='Оригинальное фото руки с уникальным нэймингом в формате base64')):
    data = Photo
    user_id = ''
    counter = ''
    #segment_image.load_model("model/mask_rcnn_model.067-0.335795.h5")
    if data == None:
        return 'Got None туц'
    else:
        # model.predict.predict returns a dictionary
        prediction = predict(segment_image, data, user_id, counter)
    return str(prediction)


@app.get('/equip', summary="Получение маски нужного цвета", response_model=mask_out)
async def equip(
        photo_name: str,
        stamping_condition: int,
        color_condition: int,
        color: str
):
    name = photo_name
    template_number = color
    user_id = ''
    counter = ''
    stamping = stamping_condition
    color = color_condition
    if name == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = equip_color.equip(name, template_number, user_id, counter, stamping, color)
    return str(prediction)


@app.get(
    '/equip_stamp',
    summary="Получение фотографии с наложенным стэмпингом",
    response_model=mask_out,
    response_description="Возвращает фото маски с нужным цветом в формате base64")
def equip_stamp(
        photo_name: str,
        stamping_condition: int,
        color_condition: int,
        stamping: str
):
    name = photo_name
    stamping_name = stamping
    stamping = stamping_condition
    color = color_condition
    user_id = ''
    counter = ''
    if name == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = equip_stamping.equip_template(name, stamping_name, stamping, color, user_id, counter)
    return str(prediction)

@app.get('/download', summary="Получение всех цветов и стемпинга для выбранного салона")
def download(
        salon_code:int
):
    name = request.form.get('data')

    if name == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = database.database(name)
    return str(prediction)

@app.get('/add_to_csv', summary="Добавить строку в csv")
def add_to_csv(
        csv_parameter:str
):
    name = request.form.get('data')

    if name == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        df = pd.DataFrame([csv_parameter])
        df.to_csv('file.csv', mode='a', header=False)
    return "success"


#if __name__ == "__main__":
#    app.run(host='0.0.0.0', debug=True, threaded=False)

#if __name__=="__main__":
#    uvicorn.run("app:app",host='0.0.0.0', debug=True)
