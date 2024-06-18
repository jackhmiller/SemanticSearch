from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from typing import List
import os


class EmbeddingModel:
	def __init__(self):
		pass

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
				 max_length: int = 128):
		EmbeddingModel.__init__(self)
		self.tokenizer = self.load_tokenizer(tokenizer)
		self.model = self.load_model(model_name)
		self.max_length = max_length

	def load_model(self, model: str):
		loaded_model = AutoModel.from_pretrained(model)
		return loaded_model

	def load_tokenizer(self, model: str):
		loaded_model = AutoTokenizer.from_pretrained(model)
		return loaded_model

	def create_embeddings(self, sentences: list[str]):
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
				 tokenizer: str = None):
		EmbeddingModel.__init__(self)
		self.model = self.load_model(model_name)
		self.tokenizer = tokenizer
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def load_model(self, model: str):
		loaded_model = pd.read_pickle(f"gs://{os.getenv('BUCKET_NAME')}/{os.getenv('MODEL_PATH')}/{model}")
		return loaded_model

	def create_embeddings(self, sentences: list[str]):
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
	pass