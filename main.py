from elastic import Search
from flask import Flask, render_template, request

app = Flask(__name__)

es = Search(host='http://localhost:9200',
            name='catalogue_embeddings',
            )

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/")
def handle_search():
    query = request.form.get("query", "")
    from_ = request.form.get("from_", type=int, default=0)
    result = es.knn_search(query)

    return render_template(
        "index.html",
        results=result["hits"]["hits"],
        query=query,
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
    app.run(host='0.0.0.0', port=5001)