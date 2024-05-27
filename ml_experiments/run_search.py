from elastic import Search
import json


def main(index_name: str,
		 reindex: bool = False):

	es = Search(host='http://localhost:9200',
				name=index_name,
				)

	if reindex:
		es.reindex_from_gcs()


	results = {}
	test_sentences = ["red dress", 'shorts', 'linen', 'linen pants', 'brooklyn ankle pants']
	for sentence in test_sentences:
		response = es.knn_search(sentence)
		hit_dict ={}
		for hit in response["hits"]["hits"]:
			hit_dict[hit["_id"]] = {'score': hit["_score"]}
		results[sentence] = hit_dict
	with open("ran_test.json", 'w') as json_file:
		json.dump(results, json_file)


if __name__ == '__main__':
	main(index_name='catalogue_embeddings',
		 reindex=False)