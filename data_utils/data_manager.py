import pandas as pd
import os
from data_utils.gcs import GCSContextManager

def get_embed_hash(model: str, features: list) -> str:
	if '.' in model:
		model = model.split('.')[0]
	elif '/' in model:
		model = model.split('/')[1]

	return model + '_' + '_'.join(features) + '.parquet'


class DataManager:
	def __init__(self,
				 path: str,
				 hash: str
				 ):
		self.hash = hash
		self.path = path

	def check_hash(self) -> bool:
		name = os.path.join(self.path, self.hash)
		with GCSContextManager() as gcs:
			blob = gcs.check_blob_exists(name)
		return blob

	def read_hash(self) -> pd.DataFrame:
		name = os.path.join(self.path, self.hash)
		with GCSContextManager() as gcs:
			df = gcs.load_parquet_from_gcs(name)
		return df

	def save_data(self, df: pd.DataFrame) -> None:
		print("Saving dataframe")
		blob = os.path.join(self.path, self.hash)
		with GCSContextManager() as gcs:
			gcs.save_parquet_to_gcs(df=df,
									blob_name=blob)

