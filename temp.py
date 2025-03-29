import os
import weaviate
from weaviate.auth import AuthApiKey  # Correct import for API key authentication
from sentence_transformers import SentenceTransformer

# Best practice: store your credentials in environment variables
weaviate_url = "https://dirgrgv7qrgqc27ecn23dg.c0.asia-southeast1.gcp.weaviate.cloud"
weaviate_api_key = "LWR7rlfqKkMRg5lstKNLJUsa2gr0O0Udjy0T"

# Connect to Weaviate Cloud
client = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=AuthApiKey(api_key=weaviate_api_key),  # Use AuthApiKey for API key authentication
)

print(client.is_ready())

if not client.is_ready():
    print("failed to connect to weaviate")
    exit()

schema = {
    "classes": [
        {
            "class": "DemoEmbedding",
            "vectorizer": "none",  # We are providing our own embeddings
            "properties": [
                {"name": "description", "dataType": ["string"]},
            ],
        }
    ]
}
existing_classes=client.schema.get()["classes"]

if not any(cls["class"] == "DemoEmbedding" for cls in existing_classes):
    client.schema.create(schema)
    print("Schema created successfully!")
else:
    print("Schema already exists.")

# client.schema.create(schema)

# model = SentenceTransformer('all-MiniLM-L6-v2')
# sample_text = "This is a demo embedding for Weaviate."
# embedding = model.encode(sample_text).tolist()

# # Store the embedding in Weaviate
# client.data_object.create(
#     data_object={
#         "description": sample_text,
#     },
#     class_name="DemoEmbedding",
#     vector=embedding  # Attach the embedding vector
# )

# print("Demo embedding stored successfully!")