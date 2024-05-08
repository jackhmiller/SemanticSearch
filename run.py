from preprocessing import CataloguePreprocessing
from embeddings import EmbeddingModel
from elastic import Search

def main():
	"""
	Run three stages end to end
	1. Parse catalogue
	2. Build embeddings
	3. Load embeddings to elastic
	"""

	########################## OFFLINE
	features = ['style', 'colors', 'fabrics', 'fits', 'tags', 'hierarchys', 'overviews']
	CataloguePreprocessing(data_path="data/athleta_sample.ndjson",
						   features=features,
						   save_to_file=True
						   ).run_preprocessing()

	feature = "overviews"
	model = EmbeddingModel(feature, inference=False)
	##### For debug - read from file or recieve data directly from preprocess
	DATA_IN_PATH = "./data/cleaned_search_data.parquet"
	#####
	model.run_embedding_model(data_path=DATA_IN_PATH)

	es = Search(host='http://localhost:9200',
				name='catalogue_embeddings')

	es.reindex()

	########################## ONLINE

	test_sentence = "Womens sports bra black"
	response = es.search(test_sentence)

	for hit in response["hits"]["hits"]:
		print("Document ID:", hit["_id"])
		print("Similarity Score:", hit["_score"])


if __name__ == '__main__':
	main()