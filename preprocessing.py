import json
import pprint
import os
import re
import pandas as pd
import string
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
from parser import parse_catalogue


class TextPreprocessor:
	def __init__(self, text):
		self.text = text
		# self.stemmer = SnowballStemmer('english')
		self.lemmatizer = WordNetLemmatizer()

	def run_preprocessing(self):
		text = self.text.apply(self.preprocess)
		text = text.apply(self.remove_stopwords)
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

	def stemming(self, text):
		chars = [self.stemmer.stem(i) for i in word_tokenize(text)]
		return " ".join(chars)

	def lemmatizing(self, text):
		lemmatizer = self.lemmatizer
		words = nltk.word_tokenize(text)
		chars = [lemmatizer.lemmatize(word) for word in words]

		return " ".join(chars)


class CataloguePreprocessing:
	def __init__(self, data_path, features, save_to_file=False):
		self.file_path = data_path
		self.features = features
		self.save_to_file = save_to_file
		self.raw_catalogue = None
		self.parsed_catalogue = None

	def run_preprocessing(self):
		self.load_data()
		self.get_parse_catalogue()

	def load_data(self):
		with open("athleta_sample.ndjson", 'r', encoding='utf-8') as file:
			raw_catalogue = []
			for line in file:
				data = json.loads(line)
				raw_catalogue.append(data)

		self.raw_catalogue = raw_catalogue

	def get_parse_catalogue(self):
		self.parsed_catalogue = parse_catalogue(self.raw_catalogue)

	def clean_catalogue_text(self):
		df = pd.DataFrame(self.parsed_catalogue).T
		df = df.astype(str)
		df_clean = pd.DataFrame()
		for col in self.features:
			df_clean[col] = df[col].apply(TextPreprocessor.preprocess)

		if self.save_to_file:
			df_clean.to_parquet("cleaned_search_data.parquet",
								index=True)

		return df_clean



if __name__ == "__main__":
	features = ['style', 'colors', 'fabrics', 'fits', 'tags', 'hierarchys', 'overviews']
	_ = CataloguePreprocessing(data_path="data/athleta_sample.ndjson",
							   features=features)