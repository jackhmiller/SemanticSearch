import pandas as pd
import urllib.request
from google.cloud import storage
import requests
from PIL import Image
from io import BytesIO

def get_pid(urls):
    pids = []
    for i in urls:
        try:
            pids.append(i.split("pid=")[1])
        except:
            pids.append(0)
    return pids

def get_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        # image = Image.open(BytesIO(response.content))
        return response.content
    else:
        print("Failed to retrieve image. Status code:", response.status_code)
        return None

def main():
    df = pd.read_parquet("gs://forecasting-algo-dev/cleaned_data/cleaned_data.parquet")

    urls = df.url.values.tolist()
    pids = get_pid(urls)
    tuples = list(zip(pids, urls))
    files = [x for x in tuples if x[1] is not None]

    client = storage.Client()
    bucket = client.bucket("forecasting-algo-dev")
    for i, (id, url) in enumerate(files):
        try:
            print(i)
            image = get_image_from_url(url)
            blob = bucket.blob(f"images/{id}.jpg")
            blob.upload_from_string(image, content_type='image/jpeg')
        except:
            pass



if __name__ == "__main__":
    main()
