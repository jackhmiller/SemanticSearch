import aiohttp
import asyncio
import pandas as pd
from google.cloud import storage


async def download_image(session, url):
	async with session.get(url) as response:
		return await response.read()


async def upload_image(client, image_data, bucket_name, name):
	bucket = client.bucket(bucket_name)

	blob = bucket.blob(f"images/{name}.jpg".format(name))
	await blob.upload_from_string(image_data, content_type='image/jpeg')


async def process_images(urls, bucket_name):
	client = storage.Client()

	async with aiohttp.ClientSession() as session:
		tasks = []
		for id, url in urls:
			if url is not None:
				try:
					print(id)
					image_data = await download_image(session, url)
					task = upload_image(client, image_data, bucket_name, id)
					tasks.append(task)
				except:
					continue

		await asyncio.gather(*tasks)


if __name__ == "__main__":

	bucket_name = "forecasting-algo-dev"
	df = pd.read_parquet("gs://forecasting-algo-dev/cleaned_data/cleaned_data.parquet")

	tuples = list(zip(df.index.tolist(), df.url.values.tolist()))
	final = [x for x in tuples if x[1] is not None]
	asyncio.run(process_images(final, bucket_name))
