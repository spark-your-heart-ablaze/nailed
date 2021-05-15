import pandas as pd
import json


def database(data):
    df = pd.read_csv('Sheets for Copy of nailedIT - Color number.csv', encoding='utf8')
    df = df.dropna(how='all')
    result = df.to_json(orient="index")
    print(json.dumps(result, indent=4))

    return 1