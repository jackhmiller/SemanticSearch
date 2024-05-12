from google.cloud import storage
import json
import os
import hashlib


class DataManager:
	def __init__(self, file: str, features: list[str], embed: list[str]):
		self.raw_file_name = file
		# self.feature_hash = hashlib.sha1(" ".join(features).encode()).hexdigest()
		self.feature_hash = "_".join(features) + '_.parquet'
		self.embedding_hash = self.feature_hash + '_' + f"encode_{'_'.join(embed)}"
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
			return blob.exists(), blob
		if phase == 'embed':
			blob = self.bucket.blob(os.path.join(self.embedding_data_path, self.embedding_hash))
			return blob.exists(), blob

	def read_hash(self):
		pass
