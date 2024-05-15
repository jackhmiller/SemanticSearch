from preprocessing import CataloguePreprocessing
from embeddings import EmbeddingModel
from dotenv import load_dotenv, find_dotenv
from data_utils import DataManager
from elastic import Search
import os
from ops.run_tracker import Tracker
from data_utils.gbq import GBQContextManager

def get_embed_hash(model: str, features: list) -> str:
	if '/' in model:
		model = model.split('/')[1]
	return model + '_' + '_'.join(features) + '.parquet'


def main(file: str,
		 embedding_features: list[str],
		 embedding_model: str,
		 tokenizer: str,
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
	load_dotenv(find_dotenv())
	tracker = Tracker(model=embedding_model,
					  embedding_features=embedding_features)

	clean_data_manager = DataManager(path=os.getenv("CLEANED_DATA_PATH"),
									 hash=os.getenv("CLEANED_DATA_FILE"))

	if (clean_data_manager.check_hash()) & (not overwrite):
		pass
	else:
		preprocess = CataloguePreprocessing(file=file)
		cleaned_data = preprocess.run_preprocessing()

		clean_data_manager.save_data(df=cleaned_data)


	embed_hash = get_embed_hash(embedding_model, embedding_features)
	embed_data_manager = DataManager(path=os.getenv("EMBEDDING_DATA_PATH"),
									 hash=embed_hash)

	if (embed_data_manager.check_hash()) & (not overwrite):
		pass
	else:
		data = embed_data_manager.read_hash()
		model = EmbeddingModel(model_name=embedding_model,
							   tokenizer=tokenizer,
							   features=embedding_features,
							   inference=False,
							   )
		data_with_embedding = model.run_embedding_model(data=data)
		embed_data_manager.save_data(data_with_embedding)
		#todo eval
		
		with GBQContextManager() as gbq:
			gbq.append_run(tracker)

	if deploy:
		es = Search(host='http://localhost:9200',
					name=index_name,
					)
		es.reindex_from_gcs()


if __name__ == '__main__':
	embed = ['overviews']
	main(file='athleta_sample.ndjson',
		 embedding_features=embed,
		 embedding_model='sentence-transformers/all-mpnet-base-v2',
		 tokenizer='sentence-transformers/all-mpnet-base-v2',
		 index_name='catalogue_embeddings',
		 deploy=False,
		 overwrite=False)