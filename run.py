from preprocessing import CataloguePreprocessing
from embeddings import EmbeddingModel
from elastic import Search
from dotenv import load_dotenv
from data_utils import DataManager


def main(file: str, features: list[str], embedding_features: list[str]):
	"""
	Run three stages end to end
	1. Parse catalogue
	2. Build embeddings
	3. Load embeddings to elastic
	(4. Automate search quality testing)
	"""
	load_dotenv()
	data_manager = DataManager(file=file,
							   features=features,
							   embed=embedding_features)

	if data_manager.check_hash(phase='clean'):
		data_manager.read_hash()
	else:
		raw_catalogue = data_manager.read_raw_catalogue()
		cleaned_data = CataloguePreprocessing(features=features,
											  data=raw_catalogue
											  ).run_preprocessing()


	model = EmbeddingModel(embedding_features,
						   inference=False)
	##### For debug - read from file or recieve data directly from preprocess
	DATA_IN_PATH = "./data/cleaned_search_data.parquet"
	#####
	model.run_embedding_model(data_path=DATA_IN_PATH)

	es = Search(host='http://localhost:9200',
				name='catalogue_embeddings')

	es.reindex()

	########################## ONLINE

	results = {}
	test_sentences = ["Womens sports bra black"]
	for sentence in test_sentences:
		response = es.search(sentence)
		hit_dict ={}
		for hit in response["hits"]["hits"]:
			hit_dict[hit["_id"]] = {'score': hit["_score"],
									'url': hit["_source"]["_url"]}
		results[sentence] = hit_dict


if __name__ == '__main__':
	parse_features = ['style', 'colors', 'fabrics', 'fits', 'tags', 'hierarchys', 'overviews']
	embed = ['overview']
	main(file='athleta_sample.ndjson',
		 features=parse_features,
		 embedding_features=embed)