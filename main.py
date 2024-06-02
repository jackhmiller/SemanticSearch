from elastic import Search
from flask import Flask, render_template, request
import re

app = Flask(__name__)

es = Search(host='http://localhost:9200',
            name='catalogue_embeddings',
            )

def extract_filters(query):
    filters = []

    filter_regex = r': ([^\s]+)\s*'
    m = re.search(filter_regex, query)
    if m:
        filters.append({
            'term': {
                '_tags.keyword': {
                    'value': m.group(1)
                }
            }
        })
        query = re.sub(filter_regex, '', query).strip()

    return {'filter': filters}, query

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/")
def handle_search():
    raw_query = request.form.get("query", "")
    filters, parsed_query = extract_filters(raw_query)
    from_ = request.form.get("from_", type=int, default=0)
    result = es.knn_search(parsed_query)

    return render_template(
        "index.html",
        results=result["hits"]["hits"],
        query=parsed_query,
        from_=from_,
        total=result["hits"]["total"]["value"],
        # aggs=aggs,
    )


@app.get("/document/<id>")
def get_document(id):
    document = es.retrieve_document(id)
    title = document["_source"]["_name"]
    paragraphs = document["_source"]["_paragraph"]
    return render_template("document.html",
                           title=title,
                           paragraphs=paragraphs
                           )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)