from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from data_utils import GCSContextManager, get_embed_hash
from pprint import pprint
import pandas as pd
import os
from embeddings import FinetuneModel, PretrainedModel


class Search:
	def __init__(self, host,
				 name: str,
				 embedding_model: str,
				 feature_set: list[str],
				 ):
		self.es = Elasticsearch(host)
		self.index_name = name
		self.model_name = embedding_model
		self.feature_set = feature_set
		self.model = FinetuneModel(model_name=embedding_model,   #todo pretrained
							  tokenizer=None,
							  features=feature_set)

	@property
	def client_info(self):
		client_info = self.es.info()
		return pprint(client_info.body)

	@property
	def index_mapping(self):
		return self.es.indices.get_mapping(index=self.index_name)

	def retrieve_document(self, id):
		return self.es.get(index=self.index_name, id=id)

	def get_embedding(self, text):
		return self.model.predict(inference_sample=text)

	def create_index(self):
		self.es.indices.delete(index=self.index_name,
							   ignore_unavailable=True)
		self.es.indices.create(index=self.index_name)

	def reindex_from_df(self, df: pd.DataFrame):
		self.create_index()
		bulk(self.es, self.generate_docs(df))

	def reindex_from_gcs(self):
		with GCSContextManager() as gcs:
			hash = get_embed_hash(self.model_name, self.feature_set)
			name = os.path.join(os.getenv("EMBEDDING_DATA_PATH"), hash)
			df = gcs.load_parquet_from_gcs(blob_name=name)
		self.create_index()
		bulk(self.es, self.generate_docs(df))

	def generate_docs(self, df: pd.DataFrame):
		for idx, row in df.iterrows():
			yield {
				'_index': self.index_name,
				'_id': idx,
				'_embedding': row['embeddings'],
				'_url': row['image_url'],
				'_product_page': row['product_page'],
				'_overview': row['overviews'],
				'_tags': row['tags'],
				'_price': row['current_price'],
				'_colors': row['colors'],
				'_price_status': row['price_status'],
				'_name': row['product_title'],
				'_paragraph': row['paragraph']  # unclean overviews
			}

	def insert_document(self, document):
		return self.es.index(index=self.index_name,
							 document=document)

	def insert_documents(self, documents: pd.DataFrame):
		return bulk(self.es, self.generate_docs(documents))

	def knn_search(self, text, filters=None):
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
							"k": 15,
							"num_candidates": 50,

							 # "filter": [
								#  {"range": {"date":}},
								#  {"match": {"_tags":}}
							 # ]
						},
						'size':20
		}
		return self.es.search(index=self.index_name,
							  **search_kwargs)


if __name__ == '__main__':
	es = Search(host='http://localhost:9200',
				name='catalogue_embeddings',
				embedding_model="rotem_model_v1.pkl",
				feature_set=['tags', 'colors'])

	es.reindex_from_gcs()
