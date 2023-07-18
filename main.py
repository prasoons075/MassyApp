import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from starlette.requests import Request
from builtins import enumerate
from config import *
import ast
import re

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import pipeline

# read the Excel file into a Pandas dataframe
df_pandas = pd.read_excel("super small excel.xlsx", sheet_name='All Data')

label_to_id = {
    'Roofing Issues': 0,
    'Utility closet water leak': 1,
    'Clog (shower / toilet/ sink)': 2,
    'Shower, sink, toilet, or tub leak': 3,
    'Toilet': 4,
    'Smell': 5,
    'Water in walls or ceiling': 6,
    'water heater leaks': 7,
    'Fixtures poor condition (loose/broken faucets, tubs, toilets)': 8,
    'Hot water complaints': 9,
    'Low water pressure /no water at all complaints': 10,
    'Humidity/Mold': 11,
    'Heating': 12,
    'Air conditioning not working/broken': 13,
    'Thermostat issues': 14,
    'Smell.1': 15,
    'Fan issues': 16,
    'leaking device': 17,
    'Inadequate system': 18,
    'Electrical': 19,
    'Fire': 20,
    'Window issues': 21,
    'Window Leak': 22,
    'Window Condensation': 23,
    'Washer/Dryer Issues': 24,
    'Dishwasher Issues': 25
}

# Labels
labels = list(label_to_id.keys())

# Text correction pipeline
fix_spelling_pipeline = pipeline("text2text-generation", model="oliverguhr/spelling-correction-multilingual-base")

# Zero-shot classification pipeline
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Ensure PyTorch is using CPU
device = torch.device("cpu")

# Load the model
model = BertForSequenceClassification.from_pretrained('./model')
model.to(device)  # Ensure model is using the correct device

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def create_table(df, page, page_size=30):
    df_paginated = df.iloc[(page-1)*page_size:page*page_size]
    num_of_rows = len(df.index)
    num_of_pages = num_of_rows // page_size + 1 if num_of_rows > 0 else 1
    rows = df_paginated.values.tolist()
    return {
        'headers': df.columns.tolist(),
        'rows': rows,
        'num_of_pages': num_of_pages
    }

def correct_text(text, max_length=2048):
    # Check for empty or None text
    if not text:
        raise ValueError("Input text cannot be empty or None")

    corrected_text = fix_spelling_pipeline("fix:" + text, max_length=max_length)[0]['generated_text']
    return corrected_text

def predict(text, threshold):
    # Tokenize the text
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Move tensors to correct device
    encoding = {key: tensor.to(device) for key, tensor in encoding.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**encoding)

    # Apply sigmoid to the output logits
    probabilities = torch.sigmoid(outputs.logits)

    # Convert the tensor to a numpy array
    probabilities = probabilities.cpu().numpy()[0]

    # Prepare a dictionary mapping labels to their probabilities
    label_probabilities = {label: prob for label, prob in zip(labels, probabilities) if prob >= threshold}

    # Sort the dictionary by value in descending order and return it as a list of tuples
    label_probabilities_sorted = sorted(label_probabilities.items(), key=lambda item: item[1], reverse=True)

    return label_probabilities_sorted

def classify_and_predict(text, correct_text_enabled=True, threshold=0.5):
    # Check for empty or None text
    if not text:
        raise ValueError("Input text cannot be empty or None")

    # Correct the text if enabled
    corrected_text = correct_text(text) if correct_text_enabled else text

    # Classify the text using the zero-shot model
    zero_shot_result = pipe(corrected_text, candidate_labels=['Home Infrastructure Problems','Others'])

    # If the result is 'Home Infrastructure Problems', perform further classification
    if zero_shot_result['labels'][0] == 'Home Infrastructure Problems':
        predictions = predict(corrected_text, threshold)
    else:
        predictions = {}

    # Prepare the output dictionary
    output = {
        "corrected_text": corrected_text,
        "predictions": predictions
    }

    return output    

# @app.on_event("startup")    
# async def startup_event():
#     return 


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, page: int = 1):
    page_data = create_table(df_pandas, page)  # call this after updating the table
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page_data": page_data,
        "page": page,
        "enumerate": enumerate,
        "DEFAULT_COLUMNS": DEFAULT_COLUMNS  # pass the list to the template
    })

@app.get("/predict", response_class=HTMLResponse)
async def read_filters(request: Request, page: int = 1):
    page_data = create_table(df_pandas, page)  # call this after updating the table
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "page_data": page_data,
        "page": page,
        "enumerate": enumerate,
        "DEFAULT_COLUMNS": DEFAULT_COLUMNS  # pass the list to the template
    })
    

@app.get("/filters", response_class=HTMLResponse)
async def read_filters(request: Request):
    return templates.TemplateResponse("filters.html", {
        "request": request,
        "enumerate": enumerate,
        "keywords": KEYWORDS,
    })

import ast

@app.post("/filters")
async def update_keywords(request: Request):
    # Load existing keywords
    with open("keywords.json") as f:
        existing_keywords = json.load(f)

    form_data = await request.form()

    # Append new keywords to existing ones
    for category, keywords_str in form_data.items():
        # Wrap the keywords_str in square brackets to make it a list
        if not (keywords_str.startswith('[') and keywords_str.endswith(']')):
            keywords_str = '[' + keywords_str + ']'

        try:
            keywords = eval(keywords_str)
        except ValueError as e:
            return {"message": f"Error parsing keywords: {e}"}

        # Continue if the counts are equal
        if len(keywords) == len(existing_keywords[category]):
            continue

        for keyword in keywords:
            # If keyword is a list, append it directly
            if isinstance(keyword, list):
                if keyword not in existing_keywords[category]:
                    existing_keywords[category].append(keyword)
            # If keyword is a single string, check if it exists in the category already
            else:
                if [keyword] not in existing_keywords[category] and keyword not in existing_keywords[category]:
                    existing_keywords[category].append(keyword)

    # Save updated keywords
    with open("keywords.json", "w") as f:
        json.dump(existing_keywords, f)
    
    return {"message": "Keywords updated"}
    
from fuzzywuzzy import fuzz
@app.post("/")
async def rules_engine_update():
    # iterate through dataframe
    for index, row in df_pandas.iterrows():
        # iterate through keys and values in the keywords dictionary
        for key, values in keywords.items():
            column_name = key.replace("Not_", "").replace("_", " ") if key.startswith('Not_') else key.replace("_", " ")
            # only process if column already exists in the DataFrame
            if column_name in df_pandas.columns:
                # check if key starts with 'Not_'
                if key.startswith('Not_'):
                    for value in values:
                        if isinstance(value, list):
                            if all(fuzz.partial_ratio(word, row['Description']) >= 85 for word in value):
                                df_pandas.loc[index, column_name] = 0
                                break
                        else:  # if values is a simple list
                            if fuzz.partial_ratio(value, row['Description']) >= 85:
                                df_pandas.loc[index, column_name] = 0
                                break
                            
                else:
                    for value in values:
                        # check if values is list of lists
                        if isinstance(value, list):
                            if all(fuzz.partial_ratio(word, row['Description']) >= 85 for word in value):
                                df_pandas.loc[index, column_name] = 1
                                break  # no need to check further lists if already matched
                        else:  # if values is a simple list
                            if fuzz.partial_ratio(value, row['Description']) >= 85:
                                df_pandas.loc[index, column_name] = 1
                                break

    # Save the updated DataFrame back to the Excel file
    df_pandas.to_excel("super small excel.xlsx", sheet_name='All Data', index=False)

    return RedirectResponse(url='/', status_code=303)




@app.post("/predict")
async def update_model_predictions():

    # Add new columns to the DataFrame for the predicted category and score
    df_pandas['predicted_category'] = ''
    df_pandas['predicted_score'] = 0.0

    # Define the threshold for the prediction
    threshold = 0.5

    # Apply the classify_and_predict method to each row of the 'Description' column
    df_pandas[['predicted_category', 'predicted_score']] = df_pandas['Description'].apply(lambda x: pd.Series(classify_and_predict(x)['predictions'][0] if classify_and_predict(x)['predictions'] and classify_and_predict(x)['predictions'][0][1] >= threshold else ['', 0.0]))
    
    
    # Save the updated DataFrame back to the Excel file
    # df_pandas.to_excel("small excel.xlsx", sheet_name='All Data', index=False)

    return RedirectResponse(url='/predict', status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
