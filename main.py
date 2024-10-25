from typing import Union

from fastapi import FastAPI, Form
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from model import predict, load_model
from Bio.Seq import translate

app = FastAPI()

# Mount the "static" directory to serve static files (e.g., index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Define a route for your SPA
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Read and return the HTML file
    load_model('mimic', 10)

    with open("static/out_of_host_model.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# Define a route for your SPA
@app.get("/DEMO", response_class=HTMLResponse)
async def read_root():
    # Read and return the HTML file
    load_model('mimic', 10)

    with open("static/paper_demo.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# Your text processing function
def mask_inference_prediction(info):
    # Your text processing logic here
    # For this example, let's return the reversed text
    config = {
        'special_token_th': 41,
        'sw_aa_size': info['winsize'],
        'calc_stats': info['calc_stats'],
        'inference_type': info['inference_type']

    }
    aaseq = ""

    example = {
        'qseq': info['aaseq'],
        'query_species': info['species'],
        'expr': info['expr']

    }
    if info['calc_stats']:
        example['query_dna_seq'] = info['query_dna_seq']
        example['qseq'] = ''.join([translate(c) if c!='<gap>' else '-' for c in example['query_dna_seq'].split(" ")])
    if info['inference_type']=='mimic':
        example['subject_dna_seq'] = ' '.join([c if c != '<gap>' else '<mask_'+aa+'>'for c,aa in zip(info['subject_dna_seq'].split(" "),example['qseq'])])
        example['subject_species'] = info['subject_species']


    model_info = load_model(model_type='mimic', winsize=info['winsize'])
    #return {"error": "DEBUG " + str(example) + " " + str(config) }
    res = predict(config, example, model_info['restrict_dict'], model_info['tokenizer'], model_info['model'])


    return res


def validate_stats_flag(stats_flag: str):
    if stats_flag == 'yes':
        return True
    return False

def validate_inference_flag(infer_flag: str):
    if (infer_flag == 'mask') or (infer_flag == 'mimic'):
        return True
    return False

def validate_len(target_codon_sequence: str, target_aaseq: str):
    return len(target_codon_sequence.split(" "))==len(target_aaseq)

def validate_codons(codon_sequence):
    codons = ['AAA', 'AAC', 'AAT', 'AAG', 'ACA', 'ACC', 'ACT', 'ACG', 'ATA', 'ATC', 'ATT', 'ATG', 'AGA', 'AGC', 'AGT', 'AGG', 'CAA', 'CAC', 'CAT', 'CAG', 'CCA', 'CCC', 'CCT', 'CCG', 'CTA', 'CTC', 'CTT', 'CTG', 'CGA', 'CGC', 'CGT', 'CGG', 'TAA', 'TAC', 'TAT', 'TAG', 'TCA', 'TCC', 'TCT', 'TCG', 'TTA', 'TTC', 'TTT', 'TTG', 'TGA', 'TGC', 'TGT', 'TGG', 'GAA', 'GAC', 'GAT', 'GAG', 'GCA', 'GCC', 'GCT', 'GCG', 'GTA', 'GTC', 'GTT', 'GTG', 'GGA', 'GGC', 'GGT', 'GGG', '<gap>']
    token_list = codon_sequence.split(" ")
    for token in token_list:
        if not token in codons:
            return False, token
    return True, None

def validate_mimic_codons(codon_sequence):
    codons = ['AAA', 'AAC', 'AAT', 'AAG', 'ACA', 'ACC', 'ACT', 'ACG', 'ATA', 'ATC', 'ATT', 'ATG', 'AGA', 'AGC', 'AGT', 'AGG', 'CAA', 'CAC', 'CAT', 'CAG', 'CCA', 'CCC', 'CCT', 'CCG', 'CTA', 'CTC', 'CTT', 'CTG', 'CGA', 'CGC', 'CGT', 'CGG', 'TAA', 'TAC', 'TAT', 'TAG', 'TCA', 'TCC', 'TCT', 'TCG', 'TTA', 'TTC', 'TTT', 'TTG', 'TGA', 'TGC', 'TGT', 'TGG', 'GAA', 'GAC', 'GAT', 'GAG', 'GCA', 'GCC', 'GCT', 'GCG', 'GTA', 'GTC', 'GTT', 'GTG', 'GGA', 'GGC', 'GGT', 'GGG', '<gap>']
    aas = "YMRS*WILNQFPHDCAGTEKV"
    aa_masks = ['<mask_'+aa+'>' for aa in aas]
    available_tokens = [*codons, *aa_masks]
    token_list = codon_sequence.split(" ")

    for token in token_list:
        if not token in available_tokens:
            return False, token
    return True, None

def validate_aas(aaseq: str):
    aaseq = aaseq.upper()
    aas = "YMRS*WILNQFPHDCAGTEKV-"
    if aaseq == "":
        return False
    for i in aaseq:
        if not i in aas:
            return False, i
    return True, None


# Define an endpoint to process text
@app.post("/pred/")
async def process_text_endpoint(aaseq: str = Form(...), winsize: int = Form(...), species: str = Form(...), expr: str = Form(...),
                                stats_flag: str=Form("no"), infer_flag: str=Form("mask"), mimic_species: str=Form("S_cerevisiae"),mimic_codon_sequence: str=Form("")):


    if not validate_inference_flag(infer_flag):
        return {'error': 'Problem with the value of inference_type: '+str(infer_flag)}
    stats_flag = validate_stats_flag(stats_flag)

    info = {"aaseq": aaseq, "winsize": winsize, "species": species, "expr": expr, 'calc_stats': stats_flag,
            'inference_type': infer_flag}

    targetlen=-1
    if stats_flag: # stats == yes; the target sequence is codons.

        flag, token = validate_codons(aaseq)
        if not flag:
            return {
                'error': "Token " + token + " in target sequence is not a codon. Please insert a codon only sequence (or gap as <gap>)."}
        info['query_dna_seq'] = aaseq
        targetlen = len(aaseq.split(" "))
    else:
        targetlen = len(aaseq)
        flag, token = validate_aas(aaseq)
        if not flag:
            return {
                'error': "Token " + token + " in target sequence is not an amino-acid character (or gap). Please insert amino-acid chars or 'gap' without spaces."}

    if infer_flag == 'mimic':
        flag, token = validate_mimic_codons(mimic_codon_sequence)
        if not flag:
            return {
                'error': "Token " + token + " in mimic codon sequence is not a codon. Please insert a codon only sequence (or gap as <gap>)."}
        if len(mimic_codon_sequence.split(" ")) != targetlen:
            return {'error': "Target and mimic sequences are not of the same length. target length = " + str(targetlen) + "; mimic length = " + str(len(mimic_codon_sequence.split(" "))) + ";"}
        info['subject_dna_seq'] = mimic_codon_sequence
        info['subject_species'] = mimic_species

    predicted_codons = mask_inference_prediction(info)
    return predicted_codons

