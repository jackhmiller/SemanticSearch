from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
import os
import itertools


class EmbeddingModel:
	def __init__(self,
				 model_name: str='sentence-transformers/all-mpnet-base-v2',
				 tokenizer: str='sentence-transformers/all-mpnet-base-v2',
				 max_length: int = 128,
				 features: List[str] = None,
				 pretrained: bool = True,
				 save_model:bool = False,
				 inference:bool = False,
				 ):
		self.pretrained = pretrained
		self.tokenizer = self.load_tokenizer(tokenizer)
		self.model = self.load_model(model_name)
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

	def load_model(self, model: str):
		if self.pretrained:
			loaded_model = AutoModel.from_pretrained(model)
			return loaded_model

	def load_tokenizer(self, model: str):
		if self.pretrained:
			loaded_model = AutoTokenizer.from_pretrained(model)
			return loaded_model

	def save_model(self, model):
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
	features = ["overviews"]
	model = EmbeddingModel(features, inference=False)
	model.run_embedding_model()