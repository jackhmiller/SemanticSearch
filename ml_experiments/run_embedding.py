from preprocessing import CataloguePreprocessing
from embeddings import EmbeddingModel
from dotenv import load_dotenv
from data_utils import DataManager
from elastic import Search


def main(file: str,
		 features: list[str],
		 embedding_features: list[str],
		 index_name: str,
		 deploy: bool=False,
		 overwrite: bool=False):
	"""
	Run three stages end to end
	1. Parse catalogue
	2. Build embeddings
	3. Load embeddings to elastic
	(4. Automate knn_search quality testing)
	"""
	load_dotenv()
	data_manager = DataManager(file=file,
							   features=features,
							   embed=embedding_features)

	if (data_manager.check_hash(phase='clean')) & (not overwrite):
		pass
	else:
		raw_catalogue = data_manager.read_raw_catalogue()
		cleaned_data = CataloguePreprocessing(features=features,
											  data=raw_catalogue
											  ).run_preprocessing()
		data_manager.save_parquet_to_gcs(df=cleaned_data,
										 phase='clean')

	if (data_manager.check_hash(phase='embed')) & (not overwrite):
		pass
	else:
		data = data_manager.read_hash('clean')
		model = EmbeddingModel(features=embedding_features,
							   inference=False,
							   )
		data_with_embedding = model.run_embedding_model(data=data)
		data_manager.save_parquet_to_gcs(data_with_embedding,
										 'embed')

		if deploy:
			es = Search(host='http://localhost:9200',
						name=index_name,
						)
			es.reindex_from_df(data_with_embedding)


if __name__ == '__main__':
	parse_features = ['style', 'colors', 'fabrics', 'fits', 'tags', 'hierarchys', 'overviews']
	embed = ['overviews']
	main(file='athleta_sample.ndjson',
		 features=parse_features,
		 embedding_features=embed,
		 index_name='catalogue_embeddings',
		 deploy=False,
		 overwrite=False)