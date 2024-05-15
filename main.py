from fastapi import FastAPI, HTTPException
from elastic import Search
from starlette.templating import Jinja2Templates
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

es = Search(host='http://localhost:9200',
            name='catalogue_embeddings',
            )

# @app.get("/")
# def index():
#     return templates.TemplateResponse("index.html")


@app.post("/")
async def search(query: str):
    response = es.knn_search(query)
    if response['hits']['total']['value'] == 0:
        raise HTTPException(status_code=404, detail="No results found")
    return templates.TemplateResponse(name="index.html",
                                      context={
                                          "results":response["hits"]["hits"],
                                          "query":query,
                                          "total":response["hits"]["total"]["value"]
                                      })



if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000)