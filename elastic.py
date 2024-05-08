from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
from pprint import pprint
import os
import time
import pandas as pd
from embeddings import EmbeddingModel

PATH = "./data/data_with_embeddings.parquet"

def loadDocs(path: str) -> pd.DataFrame:
	df = pd.read_parquet(path)
	return df

class Search:
	def __init__(self, host, name: str):
		self.es = Elasticsearch('http://localhost:9200')
		self.index_name = name

	def get_client_info(self):
		client_info = self.es.info()
		print('Connected to Elasticsearch')
		pprint(client_info.body)

	def get_embedding(self, text):
		return self.model.encode(text)

	def create_index(self):
		self.es.indecis.create(index=self.index_name)

	def reindex(self):
		df = loadDocs(PATH)
		self.create_index()
		bulk(self.es, self.generate_docs(df))

	def generate_docs(self, df: pd.DataFrame):
		for idx, row in df.iterrows():
			yield {
				'_index': self.index_name,
				'_id': idx,
				'_embedding': row['embeddings']
			}

	def search(self, **query_args):
		return self.es.search(index=self.index, **query_args)


if __name__ == '__main__':
	es = Search(host='http://localhost:9200',
				name='catalogue_embeddings')