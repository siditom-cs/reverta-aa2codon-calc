from typing import Union

from fastapi import FastAPI, Form
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from model import predict, load_model

app = FastAPI()

# Mount the "static" directory to serve static files (e.g., index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define a route for your SPA
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Read and return the HTML file
    #load_model('mimic', 10)

    with open("static/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


# Your text processing function
def mask_inference_prediction(info):
    # Your text processing logic here
    # For this example, let's return the reversed text
    config = {
        'special_token_th': 31,
        'sw_aa_size': info['winsize'],
        'calc_stats': False,
        'mimic': False

    }

    example = {
        'qseq': info['aaseq'],
        'query_species': info['species'],
        'expr': info['expr']
    }
    model_info = load_model(model_type='mimic', winsize=info['winsize'])
    res = predict(config, example, model_info['restrict_dict'], model_info['tokenizer'], model_info['model'])

    #return len(aaseq)#res
    return res


# Define an endpoint to process text
@app.post("/pred/")
async def process_text_endpoint(aaseq: str = Form(...), winsize: int = Form(...), species: str = Form(...), expr: str = Form(...)):
    info = {"aaseq":aaseq}
    #print(winsize)
    info = {"aaseq": aaseq, "winsize": winsize, "species": species, "expr": expr}
    predicted_codons = mask_inference_prediction(info)
    return predicted_codons
