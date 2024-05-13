import pandas as pd
import os
from gcs import GCSContextManager

class DataManager:
	def __init__(self,
				 features: list[str],
				 embed: list[str]
				 ):
		self.feature_hash = "_".join(features) + '.parquet'
		self.embedding_hash = "_".join(features) + '_' + f"encode_{'_'.join(embed)}" + '.parquet'
		self.bucket_name = os.getenv("BUCKET_NAME")
		self.cleaned_data_path = os.getenv("CLEANED_DATA_PATH")
		self.embedding_data_path = os.getenv("EMBEDDING_DATA_PATH")

	def check_hash(self, phase: str):
		if phase == 'clean':
			name = os.path.join(self.cleaned_data_path, self.feature_hash)
			with GCSContextManager(self.bucket_name) as gcs:
				blob = gcs.check_blob_exits(name)
			return blob.exists()
		if phase == 'embed':
			name = os.path.join(self.embedding_data_path, self.embedding_hash)
			with GCSContextManager(self.bucket_name) as gcs:
				blob = gcs.check_blob_exits(name)
			return blob.exists()

	def read_hash(self, phase: str):
		if phase == 'clean':
			name = os.path.join(self.cleaned_data_path, self.feature_hash)
			with GCSContextManager(self.bucket_name) as gcs:
				df = gcs.load_parquet_from_gcs(name)
			return df
		if phase == 'embed':
			name = os.path.join(self.embedding_data_path, self.embedding_hash)
			with GCSContextManager(self.bucket_name) as gcs:
				df = gcs.load_parquet_from_gcs(name)
			return df

	def save_data(self, phase: str, df: pd.DataFrame):
		if phase == 'clean':
			blob = os.path.join(self.cleaned_data_path, self.feature_hash)
			with GCSContextManager(self.bucket_name) as gcs:
				gcs.save_parquet_to_gcs(df=df,
										blob_name=blob)
		if phase == 'embed':
			blob = os.path.join(self.embedding_data_path, self.embedding_hash)
			with GCSContextManager(self.bucket_name) as gcs:
				gcs.save_parquet_to_gcs(df=df,
										blob_name=blob)
