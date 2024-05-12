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
		pass
	else:
		raw_catalogue = data_manager.read_raw_catalogue()
		cleaned_data = CataloguePreprocessing(features=features,
											  data=raw_catalogue
											  ).run_preprocessing()
		data_manager.save_parquet_to_gcs(df=cleaned_data,
										 phase='clean')

	if data_manager.check_hash(phase='embed'):
		pass
	else:
		data = data_manager.read_hash('clean')
		model = EmbeddingModel(features=embedding_features,
							   inference=False,
							   )
		model.run_embedding_model(data=data)
		data_manager.save_parquet_to_gcs(data,
										 'embed')



if __name__ == '__main__':
	parse_features = ['url', 'style', 'colors', 'fabrics', 'fits', 'tags', 'hierarchys', 'overviews']
	embed = ['overviews']
	main(file='athleta_sample.ndjson',
		 features=parse_features,
		 embedding_features=embed)