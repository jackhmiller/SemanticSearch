from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from typing import Union, List
from sentence_transformers import SentenceTransformer


MODEL = 'sentence-transformers/all-mpnet-base-v2'
TOKENIZER = 'sentence-transformers/all-mpnet-base-v2'
DATA_IN_PATH = "./data/cleaned_search_data.parquet"
DATA_OUT_PATH = "./data/data_with_embeddings.parquet"


class EmbeddingModel:
	def __init__(self,
				 features: Union[str, List[str]] = None,
				 model_name: str = MODEL,
				 max_length: int = 128,
				 save_model=False,
				 inference=False,
				 ):
		self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
		self.model_name = model_name
		self.model = AutoModel.from_pretrained(model_name)
		self.max_length = max_length
		self.features = features
		self.save_model = save_model
		self.inference = inference

		if not inference:
			assert len(features) > 0


	def run_embedding_model(self,
							inference_sample: str = None,
							data_path: str = None) -> Union[List, None]:
		if self.inference:
			return self.create_embeddings(inference_sample)[0]
		else:
			data = pd.read_parquet(data_path)
			print("Starting to compute embeddings")
			sentences = data[features].to_list()
			embeddings = self.create_embeddings(sentences)
			data['embeddings'] = embeddings
			data.to_parquet(DATA_OUT_PATH)
			print("Done saving embeddings")

	def load_model(self):
		loaded_model = SentenceTransformer('sentence_transformer_model')
		pass

	def save_model(self, model):
		model.save(self.model_name)
		pass

	def create_embeddings(self, sentences):
		encoded_input = self.tokenizer(sentences,
									   padding=True,
									   truncation=True,
									   return_tensors='pt',
									   max_length=self.max_length)

		with torch.no_grad():
			model_output = self.model(**encoded_input)

		final_embeddings = self.mean_pooling(model_output,
											 encoded_input["attention_mask"])

		return [i.numpy().tolist() for i in final_embeddings]


	@staticmethod
	def mean_pooling(embeddings, attention_mask):
		token_embeddings = embeddings[0]
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
		sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
		return sum_embeddings / sum_mask




if __name__ == "__main__":
	features = "overviews"
	model = EmbeddingModel(features, inference=False)
	model.run_embedding_model(data_path=DATA_IN_PATH)