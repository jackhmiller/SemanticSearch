from elastic import Search



def main(index_name: str,
		 reindex: bool = False):

	es = Search(host='http://localhost:9200',
				name=index_name,
				)

	if reindex:
		es.reindex_from_gcs()


	results = {}
	test_sentences = ["Womens sports bra black"]
	for sentence in test_sentences:
		response = es.knn_search(sentence)
		hit_dict ={}
		for hit in response["hits"]["hits"]:
			hit_dict[hit["_id"]] = {'score': hit["_score"],
									'url': hit["_source"]["_url"]}
		results[sentence] = hit_dict


if __name__ == '__main__':
	main(index_name='catalogue_embeddings',
		 reindex=True)