from data_utils import CataloguePreprocessing, DataManager, get_embed_hash, GBQContextManager
from models import FinetuneModel, PretrainedModel
from dotenv import load_dotenv, find_dotenv
import os
from ops.run_tracker import Tracker
from utilities import df_col_to_sentences


def main(file: str,
		 embedding_features: list[str],
		 embedding_model_name: str,
		 tokenizer: str,
		 overwrite: bool=False,
		 finetune: bool= False):
	"""
	Run stages end to end
	1. Parse catalogue
	2. Build embeddings
	"""
	load_dotenv(find_dotenv())
	tracker = Tracker(model=embedding_model_name,
					  embedding_features=embedding_features)

	clean_data_manager = DataManager(path=os.getenv("CLEANED_DATA_PATH"),
									 hash=os.getenv("CLEANED_DATA_FILE"))

	if (clean_data_manager.check_hash()) & (not overwrite):
		print("Cleaned data already exists")
		pass
	else:
		print("Getting data to preprocess")
		preprocess = CataloguePreprocessing(file=file)
		cleaned_data = preprocess.run_preprocessing()

		clean_data_manager.save_data(df=cleaned_data)


	embed_hash = get_embed_hash(embedding_model_name, embedding_features)
	embed_data_manager = DataManager(path=os.getenv("EMBEDDING_DATA_PATH"),
									 hash=embed_hash)

	if finetune:
		print("Running finetuning model")
		model = FinetuneModel(model_name=embedding_model_name,
							  tokenizer=tokenizer)
	else:
		print("Running pretrained model")
		model = PretrainedModel(model_name=embedding_model_name,
								tokenizer=tokenizer)

	if (embed_data_manager.check_hash()) & (not overwrite):
		pass
	else:
		data = clean_data_manager.read_hash()
		sentences = df_col_to_sentences(features=embedding_features,
										data=data)

		data['embeddings'] = model.create_embeddings(sentences=sentences)
		embed_data_manager.save_data(data)
		
		with GBQContextManager() as gbq:
			gbq.append_run(tracker)



if __name__ == '__main__':
	embed = ['tags', 'colors', 'style', 'hierarchys']
	model_params = {
		"file": 'athleta_sample.ndjson',
		"embedding_features": embed,
		"embedding_model_name": "sentence_transformer_model_with_click.pkl",  # for pretrained: 'sentence-transformers/all-mpnet-base-v2'
		"tokenizer": None,  					  # for pretrained: 'sentence-transformers/all-mpnet-base-v2'
		"overwrite": False,
		"finetune": True,
	}


	main(**model_params)