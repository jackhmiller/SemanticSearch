from google.cloud import storage
import io
import pyarrow.parquet as pq
import pandas as pd
import os

class GCSContextManager:
    def __init__(self):
        self.bucket_name = os.getenv("BUCKET_NAME")
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def check_blob_exits(self, path:str) -> bool:
        blob = self.bucket.blob(path)
        return blob.exists()

    def load_parquet_from_gcs(self, blob_name: str) -> pd.DataFrame:
        blob = self.bucket.blob(blob_name)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        table = pq.read_table(buffer)
        df = table.to_pandas()
        return df

    def save_parquet_to_gcs(self, df: pd.DataFrame, blob_name:str):
        destination_uri = f'gs://{self.bucket_name}/{blob_name}'
        df.to_parquet(destination_uri)