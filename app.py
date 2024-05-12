import re
from flask import Flask, render_template, request
from elastic import Search

app = Flask(__name__)
es = Search()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/")
def handle_search():
    query = request.form.get("query", "")
    #filters, parsed_query = extract_filters(query)
    from_ = request.form.get("from_", type=int, default=0)

    if parsed_query:
        search_query = {
            "must": {
                "multi_match": {
                    "query": parsed_query,
                    "fields": ["name", "summary", "content"],
                }
            }
        }
    else:
        search_query = {"must": {"match_all": {}}}

    results = es.search()
    return render_template(
        "index.html",
        results=results["hits"]["hits"],
        query=query,
        from_=from_,
        total=results["hits"]["total"]["value"],
        aggs=aggs,
    )