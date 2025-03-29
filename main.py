
from fastapi import FastAPI,status, UploadFile, File, Query
import weaviate
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2, os
from sentence_transformers import SentenceTransformer
from weaviate.auth import AuthApiKey 

app=FastAPI()
WEAVIATE_URL = "https://dirgrgv7qrgqc27ecn23dg.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "LWR7rlfqKkMRg5lstKNLJUsa2gr0O0Udjy0T"
WCS_USER="dheeraj.nemalikanti@gmail.com"
WCS_PASSWORD="nuvPi7-tomreb-fiqjyd"


client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),  # Use AuthApiKey for API key authentication
)




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
def hello():
    return {"message":"hello world"}


def extract_text(pdf):
    try:
        with open(pdf,'rb') as pdf_file:
            reader=PyPDF2.PdfReader(pdf_file)
            extract_text=""
            for page in reader.pages:
                extract_text+=page.extract_text()
            with open('output.txt','w',encoding='utf-8') as txt_file:
                txt_file.write(extract_text)
            return extract_text
    except Exception as e:
        print(f'an error occured: {e}')

@app.post("/api/v1/upload",status_code=status.HTTP_200_OK)
async def upload(file: UploadFile = File(...)):
    #for each document, generate embeddings using hugging face
    # store those embeddings in weaviate db
    #if new doc is uploaded with same name, then previously stored embeddings 
    # should be overwritten
    file_content=await file.read()
    temp_file_path=f'temp_{file.filename}'
    try:
        with open(temp_file_path,'wb') as temp_file:
            temp_file.write(file_content)
        text=extract_text(temp_file_path)
        
        if not text.strip():
            return {"error":"No extractable text found in the pdf"}
        #generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text).tolist()

        schema = {
            "classes": [
                {
                    "class": "Document",
                    "vectorizer": "none",  # We are providing our own embeddings
                    "properties": [
                        {"name": "filename", "dataType": ["string"]},
                        {"name": "content", "dataType": ["string"]}
                    ],
                }
            ]
        }
        existing_classes=client.schema.get()["classes"]

        if not any(cls["class"] == "Document" for cls in existing_classes):
            client.schema.create(schema)
            print("Schema created successfully!")

        #check if the document with the same name already exists or not
        result = client.query.get("Document", ["filename"]).with_where({
            "path": ["filename"],
            "operator": "Equal",
            "valueString": file.filename
        }).do()
        if result["data"]["Get"]["Document"]:
            # If the document exists, delete it
            existing_id = result["data"]["Get"]["Document"][0]["_additional"]["id"]
            client.data_object.delete(existing_id)
            print(f"Existing document with filename '{file.filename}' deleted.")

        # Add the new document with embeddings
        client.data_object.create(
            data_object={
                "filename": file.filename,
                "content": text,
            },
            class_name="Document",
            vector=embedding  # Attach the embedding vector
        )

    except Exception as e:
        print(f'error occured: {e}')
        return {"error": "failed to process the file"}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    file_name=file.filename
    print(f"Received file: {file_name}, Size: {len(file_content)} bytes")
    return {"message": f"File '{file_name}' uploaded successfully","filename":file_name,"embedding": embedding}


@app.get('/api/v1/search',status_code=status.HTTP_200_OK)
async def search_documents(query: str = Query(...,description="User query")):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()
    response = client.query.get(
        "Document",
        ["filename","content"]
    ).with_near_vector({"vector":query_embedding}).with_limit(3).do()
    results=[]
    for doc in response["data"]["Get"]["Document"]:
        results.append({
            "id": doc["_additional"]["id"],
            "filename": doc["filename"],
            "snippet": doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"]
        })

    return {"query": query, "results": results}