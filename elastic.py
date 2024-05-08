from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
from pprint import pprint
import pandas as pd
from embeddings import EmbeddingModel

PATH = "./data/data_with_embeddings.parquet"


def load_docs(path: str) -> pd.DataFrame:
	df = pd.read_parquet(path)
	return df


class Search:
	def __init__(self, host, name: str):
		self.es = Elasticsearch(host)
		self.index_name = name
		self.model = EmbeddingModel(inference=True)

	def get_client_info(self):
		client_info = self.es.info()
		print('Connected to Elasticsearch')
		pprint(client_info.body)

	def get_embedding(self, text):
		return self.model.run_embedding_model(inference_sample=text)

	def create_index(self):
		self.es.indices.delete(index=self.index_name,
							   ignore_unavailable=True)
		self.es.indices.create(index=self.index_name)

	def reindex(self):
		df = load_docs(PATH)
		self.create_index()
		bulk(self.es, self.generate_docs(df))

	def generate_docs(self, df: pd.DataFrame):
		for idx, row in df.iterrows():
			yield {
				'_index': self.index_name,
				'_id': idx,
				'_embedding': row['embeddings']
			}

	def insert_document(self, document):
		return self.es.index(index=self.index_name,
							 document=document)

	def insert_documents(self, documents: pd.DataFrame):
		return bulk(self.es, self.generate_docs(documents))

	def search(self, text):
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
						'size':5
		}
		return self.es.search(index=self.index_name,
							  **search_kwargs)


if __name__ == '__main__':
	es = Search(host='http://localhost:9200',
				name='catalogue_embeddings')

	# es.reindex()

	test_sentence = "Womens sports bra black"
	response = es.search(test_sentence)

	for hit in response["hits"]["hits"]:
		print("Document ID:", hit["_id"])
		print("Similarity Score:", hit["_score"])