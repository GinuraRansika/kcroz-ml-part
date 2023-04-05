from fastapi import FastAPI, Request
# from pydantic import BaseModel
import uvicorn
# import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from newsapi import NewsApiClient
import newspaper


app = FastAPI()
gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}


def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = "Ginura/kcroz-summerization-model"
    tokenizer = AutoTokenizer.from_pretrained(model)
    kcrozModel= AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    pipe = pipeline("summarization", model=model)
    return tokenizer,kcrozModel, pipe

tokenizer, model, pipe = get_model()


@app.post("/summerize")
async def read_root(request: Request):
    data = await request.json()
    print(data)
    if 'text'in data:
        user_interest = data['text']
        newspaper_url = get_article_url(user_interest)
        article_content = get_newspaper_content(newspaper_url)
        output= pipe(article_content, **gen_kwargs)[0]['summary_text']
        response = output
    else:
        response = {"Sorry!, Something went wrong"}
    return response

def get_article_url(user_interest):
    newsapi = NewsApiClient(api_key='ad2c008f0e354dd38de6c27d44d057eb')
    sources = newsapi.get_sources()
    top_headlines = newsapi.get_everything(q='${user_interest}',language='en', sort_by='relevancy')
    articles = top_headlines['articles']
    articles_dict = articles[0]
    url = (articles_dict['url'])
    return url

def get_newspaper_content(url):
    article_object = newspaper.Article(url)
    article_object.download()
    article_object.parse()
    article_content = article_object.text
    return article_content

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080, reload=True)