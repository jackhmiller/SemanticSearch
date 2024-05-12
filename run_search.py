from preprocessing import CataloguePreprocessing
from embeddings import EmbeddingModel
from elastic import Search
from dotenv import load_dotenv
from data_utils import DataManager


def main():

	es = Search(host='http://localhost:9200',
				name='catalogue_embeddings',
				data_loader=data_manager)

	es.reindex()

	########################## ONLINE

	results = {}
	test_sentences = ["Womens sports bra black"]
	for sentence in test_sentences:
		response = es.search(sentence)
		hit_dict ={}
		for hit in response["hits"]["hits"]:
			hit_dict[hit["_id"]] = {'score': hit["_score"],
									'url': hit["_source"]["_url"]}
		results[sentence] = hit_dict


if __name__ == '__main__':
	parse_features = ['url', 'style', 'colors', 'fabrics', 'fits', 'tags', 'hierarchys', 'overviews']
	embed = ['overviews']
	main(file='athleta_sample.ndjson',
		 features=parse_features,
		 embedding_features=embed)