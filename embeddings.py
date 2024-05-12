from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from typing import Union, List
from sentence_transformers import SentenceTransformer
import os
import itertools

TOKENIZER = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
MODEL = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

class EmbeddingModel:
	def __init__(self,
				 features: Union[str, List[str]] = None,
				 model_name: str = os.getenv('MODEL'),
				 max_length: int = 128,
				 save_model=False,
				 inference=False,
				 ):
		self.tokenizer = TOKENIZER
		self.model_name = model_name
		self.model = MODEL
		self.max_length = max_length
		self.features = features
		self.save_model = save_model
		self.inference = inference

		if not inference:
			assert len(features) > 0


	def run_embedding_model(self,
							data: pd.DataFrame = None,
							inference_sample: str = None):
		if self.inference:
			return self.create_embeddings(inference_sample)[0]
		else:

			print("Starting to compute embeddings")
			sentences = self.convert_to_sentences(data)
			embeddings = self.create_embeddings(sentences)
			data['embeddings'] = embeddings
			return data

	def load_model(self):
		loaded_model = SentenceTransformer('sentence_transformer_model')
		pass

	def save_model(self, model):
		model.save(self.model_name)
		pass

	def convert_to_sentences(self, data: pd.DataFrame) ->list:
		if len(self.features) > 1:
			sentence_pairs = data[self.features].values.tolist()
			sentences = [' '.join(i) for i in sentence_pairs]
		else:
			sentences = data[self.features].values.tolist()
			sentences = list(itertools.chain.from_iterable(sentences))

		return sentences

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
	model.run_embedding_model()