from fastapi import FastAPI, Request
# from pydantic import BaseModel
import uvicorn
# import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch


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
        user_input = data['text']
        output= pipe(user_input, **gen_kwargs)[0]['summary_text']
        response = {"recieved text": user_input, "summary text": output}
    else:
        response = {"Nothing"}
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)