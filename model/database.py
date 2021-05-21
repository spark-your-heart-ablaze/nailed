import pandas as pd
import json
import sys


def database(data):
    df = pd.read_csv('model/Sheets for Copy of nailedIT - Color number.csv', encoding='utf8')
    df = df.dropna(how='all')
    df = df[df["Салоны"].str.contains(data)]

    df1 = pd.read_csv('model/Sheets for Copy of nailedIT - Стемпинг.csv', encoding='utf8')
    df1 = df1.dropna(how='all')
    df1 = df1[df1["Салоны"].str.contains(data)]
    df = df.append(df1)
    df.reset_index(inplace=True)
    result1 = df.to_json(orient="index", force_ascii=False, indent=4)

    return result1.replace('\/', r'/')