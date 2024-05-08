from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd

MODEL = 'sentence-transformers/all-mpnet-base-v2'
TOKENIZER = 'sentence-transformers/all-mpnet-base-v2'
DATA_IN_PATH = "./data/cleaned_search_data.parquet"
DATA_OUT_PATH = "./data/data_with_embeddings.parquet"


class EmbeddingModel:
	def __init__(self, data_path: str, features: str | list[str], max_length: int = 128):
		self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
		self.model = AutoModel.from_pretrained(MODEL)
		self.data = pd.read_parquet(pd.read_parquet(data_path))
		self.max_length = max_length
		self.features = features


	def run_embedding_model(self):
		embeddings = self.create_embeddings()
		self.data['embeddings'] = embeddings
		self.data.to_parquet(DATA_OUT_PATH)

	def create_embeddings(self):
		sentences = list(self.data[features].values)
		#todo currently only for 1 feature
		encoded_input = self.tokenizer(sentences,
									   padding=True,
									   truncation=True,
									   return_tensors='pt',
									   max_length=self.max_length)

		with torch.no_grad():
			model_output = self.model(**encoded_input)

		final_embeddings = self.mean_pooling(model_output,
											 encoded_input["attention_mask"])

		return [i.numpy() for i in final_embeddings]


	@staticmethod
	def mean_pooling(embeddings, attention_mask):
		token_embeddings = embeddings[0]
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
		sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
		return sum_embeddings / sum_mask








if __name__ == "__main__":
	features = "overviews"
	EmbeddingModel(DATA_IN_PATH, features)