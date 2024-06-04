from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from typing import List
import os
import itertools


class EmbeddingModel:
	def __init__(self,
				 features: List[str],
				 ):

		self.features = features

	def convert_to_sentences(self, data: pd.DataFrame) ->list:
		if len(self.features) > 1:
			sentence_pairs = data[self.features].values.tolist()
			sentences = [' '.join(i) for i in sentence_pairs]
		else:
			sentences = data[self.features].values.tolist()
			sentences = list(itertools.chain.from_iterable(sentences))

		return sentences

	@staticmethod
	def mean_pooling(embeddings, attention_mask):
		token_embeddings = embeddings[0]
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
		sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
		return sum_embeddings / sum_mask


class PretrainedModel(EmbeddingModel):
	def __init__(self,
				 model_name: str,
				 tokenizer: str,
				 features: List[str],
				 max_length: int = 128):
		EmbeddingModel.__init__(self, features=features)
		self.tokenizer = self.load_tokenizer(tokenizer)
		self.model = self.load_model(model_name)
		self.max_length = max_length

	def load_model(self, model: str):
		loaded_model = AutoModel.from_pretrained(model)
		return loaded_model

	def load_tokenizer(self, model: str):
		loaded_model = AutoTokenizer.from_pretrained(model)
		return loaded_model

	def run_embedding_model(self,
							data: pd.DataFrame):

		print("Starting to compute embeddings")
		sentences = self.convert_to_sentences(data)
		embeddings = self.create_embeddings(sentences)
		data['embeddings'] = embeddings
		return data

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

	def predict(self, inference_sample):
		return self.create_embeddings(inference_sample)[0]


class FinetuneModel(EmbeddingModel):
	def __init__(self,
				 model_name: str,
				 features: List[str],
				 tokenizer: str = None):
		EmbeddingModel.__init__(self, features=features)
		self.model = self.load_model(model_name)
		self.tokenizer = tokenizer
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def load_model(self, model: str):
		loaded_model = pd.read_pickle(f"gs://{os.getenv('BUCKET_NAME')}/{os.getenv('MODEL_PATH')}/{model}")
		return loaded_model

	def run_embedding_model(self,
							data: pd.DataFrame):

		print("Starting to compute embeddings")
		sentences = self.convert_to_sentences(data)
		embeddings = self.create_embeddings(sentences)
		data['embeddings'] = embeddings
		return data

	def create_embeddings(self, sentences):
		if self.tokenizer:
			encoded_input = self.tokenizer(sentences,
										   padding=True,
										   truncation=True,
										   return_tensors='pt',
										   max_length=self.max_length)

		with torch.no_grad():
			model_output = self.model.encode(sentences,
											 device=self.device)

		print(model_output.shape)

		return model_output.tolist()

	def predict(self, inference_sample):
		with torch.no_grad():
			prediction = self.model.encode(inference_sample,
										   device=self.device)

		return prediction




if __name__ == "__main__":
	features = ["overviews"]
	model = EmbeddingModel(features, inference=False)
	model.run_embedding_model()