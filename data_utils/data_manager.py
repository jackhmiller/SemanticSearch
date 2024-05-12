import pandas as pd
from google.cloud import storage
import json
import os
import io
import pyarrow.parquet as pq

class DataManager:
	def __init__(self, file: str, features: list[str], embed: list[str]):
		self.raw_file_name = file
		# self.feature_hash = hashlib.sha1(" ".join(features).encode()).hexdigest()
		self.feature_hash = "_".join(features) + '.parquet'
		self.embedding_hash = "_".join(features) + '_' + f"encode_{'_'.join(embed)}" + '.parquet'
		self.gcs = storage.Client()
		self.bucket = self.gcs.bucket(os.getenv("BUCKET_NAME"))
		self.raw_data_path = os.getenv("RAW_DATA_PATH")
		self.cleaned_data_path = os.getenv("CLEANED_DATA_PATH")
		self.embedding_data_path = os.getenv("EMBEDDING_DATA_PATH")


	def read_raw_catalogue(self) -> list:
		blob = self.bucket.blob(os.path.join(self.raw_data_path,
										self.raw_file_name))

		with blob.open("r", encoding='utf-8') as file:
			catalogue = []
			for line in file:
				data = json.loads(line)
				catalogue.append(data)

		return catalogue

	def check_hash(self, phase: str):
		if phase == 'clean':
			blob = self.bucket.blob(os.path.join(self.cleaned_data_path, self.feature_hash))
			return blob.exists()
		if phase == 'embed':
			blob = self.bucket.blob(os.path.join(self.embedding_data_path, self.embedding_hash))
			return blob.exists()

	def read_hash(self, phase: str):
		if phase == 'clean':
			df = self.load_parquet_from_gcs(os.path.join(self.cleaned_data_path, self.feature_hash))
			return df
		if phase == 'embed':
			df = self.load_parquet_from_gcs(os.path.join(self.embedding_data_path, self.embedding_hash))
			return df

	def load_parquet_from_gcs(self, blob_name: str) -> pd.DataFrame:
		blob = self.bucket.blob(blob_name)
		buffer = io.BytesIO()
		blob.download_to_file(buffer)
		buffer.seek(0)
		table = pq.read_table(buffer)
		df = table.to_pandas()
		return df

	def save_parquet_to_gcs(self, df, phase):
		if phase == 'clean':
			path = os.path.join(self.cleaned_data_path, self.feature_hash)
			destination_uri = f'gs://{os.getenv("BUCKET_NAME")}/{path}'
			df.to_parquet(destination_uri)
		if phase == 'embed':
			path = os.path.join(self.embedding_data_path, self.embedding_hash)
			destination_uri = f'gs://{os.getenv("BUCKET_NAME")}/{path}'
			df.to_parquet(destination_uri)
