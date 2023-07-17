import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from starlette.requests import Request
from builtins import enumerate
from config import *

# read the Excel file into a Pandas dataframe
df_pandas = pd.read_excel("super small excel.xlsx", sheet_name='All Data')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def create_table(df, page, page_size=50):
    df_paginated = df.iloc[(page-1)*page_size:page*page_size]
    num_of_rows = len(df.index)
    num_of_pages = num_of_rows // page_size + 1 if num_of_rows > 0 else 1
    rows = df_paginated.values.tolist()
    return {
        'headers': df.columns.tolist(),
        'rows': rows,
        'num_of_pages': num_of_pages
    }

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

@app.get("/filters", response_class=HTMLResponse)
async def read_filters(request: Request):
    return templates.TemplateResponse("filters.html", {
        "request": request,
        "enumerate": enumerate,
        "keywords": KEYWORDS,
    })

@app.post("/filters")
async def update_keywords(request: Request):
    # Load existing keywords
    with open("keywords.json") as f:
        existing_keywords = json.load(f)

    form_data = await request.form()

    # Append new keywords to existing ones
    for category, keywords in form_data.items():
        new_keywords = [x.strip() for x in keywords.split(",")]
        for keyword in new_keywords:
            if keyword not in existing_keywords[category]:
                existing_keywords[category].append(keyword)

    # Save updated keywords
    with open("keywords.json", "w") as f:
        json.dump(existing_keywords, f)
    
    return {"message": "Keywords updated"}

    
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
                            if all(word in row['Description'] for word in value):
                                df_pandas.loc[index, column_name] = 0
                                break
                        else:  # if values is a simple list
                            if value in row['Description']:
                                df_pandas.loc[index, column_name] = 0
                                break
                            
                else:
                    for value in values:
                        # check if values is list of lists
                        if isinstance(value, list):
                            if all(word in row['Description'] for word in value):
                                df_pandas.loc[index, column_name] = 1
                                break  # no need to check further lists if already matched
                        else:  # if values is a simple list
                            if value in row['Description']:
                                df_pandas.loc[index, column_name] = 1
                                break

    # Save the updated DataFrame back to the Excel file
    df_pandas.to_excel("super small excel.xlsx", sheet_name='All Data', index=False)

    return RedirectResponse(url='/', status_code=303)




async def update_model_predictions():
    print("model dta updated")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
