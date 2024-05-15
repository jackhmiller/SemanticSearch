import json
import re
import pandas as pd
import string
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from data_utils.parser import parse_catalogue
import numpy as np
import os
from google.cloud import storage


class TextPreprocessor:
	def __init__(self, text):
		self.text = text
		# self.stemmer = SnowballStemmer('english')
		self.lemmatizer = WordNetLemmatizer()

	def run_preprocessing(self):
		text = self.text.apply(self.preprocess)
		text = text.apply(self.remove_stopwords)
		text = text.apply(self.remove_double_words)
		# text = text.apply(self.stemming)
		text = text.apply(self.lemmatizing)
		return text

	@staticmethod
	def preprocess(text):
		text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
		text = text.lower()
		text = text.strip()
		text = re.compile('<.*?>').sub('', text)
		text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
		text = re.sub('\s+', ' ', text)
		text = re.sub(r'\[[0-9]*\]', ' ', text)  # [0-9]
		text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
		text = re.sub(r'\d', ' ', text)
		text = re.sub(r'\s+', ' ', text)
		return text

	@staticmethod
	def remove_stopwords(text):
		chars = [i for i in text.split() if i not in stopwords.words('english')]
		return ' '.join(chars)

	@staticmethod
	def remove_double_words(text):
		return ' '.join(list(set([i for i in text.split(' ')])))

	def stemming(self, text):
		chars = [self.stemmer.stem(i) for i in word_tokenize(text)]
		return " ".join(chars)

	def lemmatizing(self, text):
		lemmatizer = self.lemmatizer
		words = nltk.word_tokenize(text)
		chars = [lemmatizer.lemmatize(word) for word in words]

		return " ".join(chars)


class CataloguePreprocessing:
	def __init__(self, file: str):
		self.raw_file_name = file
		self.raw_data_path = os.getenv("RAW_DATA_PATH")
		self.gcs = storage.Client()
		self.bucket = self.gcs.bucket(os.getenv("BUCKET_NAME"))
		self.text_features = ['style', 'colors', 'fabrics', 'fits', 'tags', 'hierarchys', 'overviews']
		self.raw_catalogue = None
		self.parsed_catalogue = None

	def run_preprocessing(self) -> pd.DataFrame:
		self.read_raw_catalogue()
		self.get_parse_catalogue()
		df = self.clean_catalogue_text()
		return df

	def get_parse_catalogue(self):
		self.parsed_catalogue = parse_catalogue(self.raw_catalogue)

	def read_raw_catalogue(self) -> list:
		blob = self.bucket.blob(os.path.join(self.raw_data_path,
										self.raw_file_name))

		with blob.open("r", encoding='utf-8') as file:
			catalogue = []
			for line in file:
				data = json.loads(line)
				catalogue.append(data)

		return catalogue

	@staticmethod
	def clean_price(prices: list) ->list:
		cleaned_prices = []
		for i in prices:
			if i:
				if not np.isnan(i).all():
					p = list(set(i))
					if len(p) > 1:
						cleaned_prices.append([min(p), max(p)])
					elif len(p) == 1:
						cleaned_prices.append(p)
				else:
					cleaned_prices.append(None)
			else:
				cleaned_prices.append(None)

		return cleaned_prices

	def clean_catalogue_text(self):
		df = pd.DataFrame(self.parsed_catalogue).T
		df_clean = pd.DataFrame()
		df_clean['current_price'] = self.clean_price(df['current_price'].to_list())
		df_clean['url'] = df['url'].astype(str)
		for col in self.text_features:
			df[col] = df[col].astype(str)
			df_clean[col] = df[col].apply(TextPreprocessor.preprocess)

		return df_clean

	def enrich_text(self, feature):
		#todo
		pass


if __name__ == "__main__":
	pass