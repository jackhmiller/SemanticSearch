from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from data_utils.gcs import GCSContextManager
from pprint import pprint
import pandas as pd
from embeddings import EmbeddingModel
import os


class Search:
	def __init__(self, host, name: str):
		self.es = Elasticsearch(host)
		self.index_name = name
		self.model = EmbeddingModel(inference=True)

	@property
	def client_info(self):
		client_info = self.es.info()
		return pprint(client_info.body)

	@property
	def index_mapping(self):
		return self.es.indices.get_mapping(index=self.index_name)

	def get_embedding(self, text):
		return self.model.run_embedding_model(inference_sample=text)

	def create_index(self):
		self.es.indices.delete(index=self.index_name,
							   ignore_unavailable=True)
		self.es.indices.create(index=self.index_name)

	def reindex_from_df(self, df: pd.DataFrame):
		self.create_index()
		bulk(self.es, self.generate_docs(df))

	def reindex_from_gcs(self):
		with GCSContextManager() as gcs:
			df = gcs.load_parquet_from_gcs(blob_name='None') #todo preplace deployed embeddings
		self.create_index()
		bulk(self.es, self.generate_docs(df))

	def generate_docs(self, df: pd.DataFrame):
		for idx, row in df.iterrows():
			yield {
				'_index': self.index_name,
				'_id': idx,
				'_embedding': row['embeddings'],
				'_url': row['url']
			}

	def insert_document(self, document):
		return self.es.index(index=self.index_name,
							 document=document)

	def insert_documents(self, documents: pd.DataFrame):
		return bulk(self.es, self.generate_docs(documents))

	def knn_search(self, text):
		search_kwargs = {'query':
								{
									"match": {
										"title": {
											"query": text
										}
									}
								},
						'knn':
							{"field": "_embedding",
							"query_vector": self.get_embedding(text),
							"k": 10,
							"num_candidates": 50,
							# **filters,
						},
						'size':10
		}
		return self.es.search(index=self.index_name,
							  **search_kwargs)


if __name__ == '__main__':
	es = Search(host='http://localhost:9200',
				name='catalogue_embeddings')
