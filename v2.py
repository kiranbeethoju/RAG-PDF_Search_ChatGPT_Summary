from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import re
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Initialize Qdrant client
client = QdrantClient(host='localhost', port=6333)
vector_size = 1024  # Replace with the appropriate dimensionality for your embeddings

# Load the embedding model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

# Function to get embeddings
def bge_m3_embed(query: str):
    embeddings = model.encode([query])['dense_vecs'][0]
    return embeddings

# Function to split text into chunks
def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []
    end_of_sentence_pattern = r'[.!?]'
    text_length = len(text)

    sentences = re.split(end_of_sentence_pattern, text)

    current_chunk = ''
    for sentence in sentences:
        sentence += re.search(end_of_sentence_pattern, text[len(current_chunk):]).group()

        if len(current_chunk) + len(sentence) > chunk_size:
            split_index = current_chunk.rfind('.') + 1
            if split_index == 0:
                split_index = current_chunk.rfind('!') + 1
            if split_index == 0:
                split_index = current_chunk.rfind('?') + 1
            if split_index == 0:
                split_index = len(current_chunk) - overlap

            chunks.append(current_chunk[:split_index])
            current_chunk = current_chunk[split_index:]

        current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Function to convert PDF to text using pytesseract
def convert_pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    full_text = ""
    for image in images:
        text = pytesseract.image_to_string(image)
        full_text += text + "\n"
    return full_text

# Function to push data to Qdrant
def push_to_qdrant(docs, collection_name, batch_size=10):
    batch = []
    for i, doc in enumerate(docs):
        embedding = bge_m3_embed(doc)
        payload = PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"text": doc, "date": datetime.now().strftime("%Y-%m-%d")}
        )
        batch.append(payload)

        if len(batch) == batch_size or i == len(docs) - 1:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            batch = []

# Function to search Qdrant
def search_qdrant(query, collection_name, top_k=2):
    query_embeddings = model.encode([query])['dense_vecs'][0]
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embeddings,
        limit=top_k
    )

    results = []
    for item in search_result:
        payload = item.payload
        text = payload.get('text', '')
        date = payload.get('date', '')
        score = item.score
        results.append((text, date, score))

    return results
from flask import jsonify

# Global variable to store progress
progress = 0
# Route for the upload page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global progress
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return 'No PDF file provided', 400

        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return 'No file selected', 400

        filename = secure_filename(pdf_file.filename)
        collection_name = os.path.splitext(filename)[0]  # Use the file name as the collection name
        pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Function to create a collection with specified vector configuration
        def create_collection(collection_name, vector_size):
            try:
                vectors_config = {
                    "size": vector_size,
                    "distance": "Cosine"
                }
                client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
                print(f"Collection '{collection_name}' created successfully with vector dimension {vector_size}")
            except Exception as e:
                print(f"Error occurred while creating collection '{collection_name}': {str(e)}")

        # Get the list of collection names
        collections = client.get_collections()
        collection_names = [collection[0] for collection in collections]  # Assuming collections is a list of tuples

        # Create the collection if it doesn't exist
        if collection_name not in collection_names:
            create_collection(collection_name, vector_size)

        # Convert PDF to text
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_text = convert_pdf_to_text(pdf_path)



        # Chunk the text and push to Qdrant
        chunks = chunk_text(pdf_text)
        total_chunks = len(chunks)
        print(total_chunks)
        chunk_size = 10  # Number of chunks to upload together (you can adjust this based on your needs)
        for i in range(0, total_chunks, chunk_size):
            chunk_batch = chunks[i:i+chunk_size]
            push_to_qdrant(chunk_batch, collection_name)  # Push multiple chunks to the collection at once
            progress = (i + len(chunk_batch)) / total_chunks * 100
            print(f"Progress: {progress:.2f}%")

        return redirect(url_for('search_page', collection_name=collection_name))

    # Add a default return statement for GET requests
    return render_template('upload.html')

# Route to get progress updates
@app.route('/progress', methods=['GET'])
def get_progress():
    global progress
    return jsonify({"progress": progress})

# Function to get all collections from Qdrant
def get_collections():
    try:
        collections = client.get_collections()
        return [collection.name for collection in collections]
    except Exception as e:
        print(f"Error occurred while fetching collections: {str(e)}")
        return []

# Route for displaying all collections
@app.route('/collections', methods=['GET'])
def display_collections():
    try:
        collections = get_collections()
        return render_template('collections.html', collections=collections)
    except Exception as e:
        return str(e), 500




# Search function
def search_ocr_sample(query, collection_name, top_k=5):
    query_embeddings = bge_m3_embed(query)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embeddings,
        limit=top_k
    )

    results = []
    for item in search_result:
        payload = item.payload
        text = payload.get('text', '')
        date = payload.get('date', '')
        score = item.score
        results.append((text, date, score))

    return results

# Route for the search page
@app.route('/search/<collection_name>', methods=['GET', 'POST'])
def search_page(collection_name):
    if request.method == 'POST':
        query = request.form['query']
        try:
            results = search_ocr_sample(query, collection_name)
            resultsSum = ', '.join([r[0] for r in results])  # Joining all result texts

            import os
            import openai

            from openai import OpenAI
            client = OpenAI(api_key = "sk-proj-HbbqVhqnECiDEQ85qN8AT3BlbkFJsQXbB4MOzwmVhrF71N4L")

            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "user",
                "content": "user Question is {}, You must use this context and provide me the relevant output, give me the content from this context only do not miss any information  {}".format(query, resultsSum)
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
            print(response)
            try:
                rep = response.choices[0].message.content

            except:
                rep = "open ai API failed"

            return render_template('search.html', query=query, results=results, resultsSum=str(rep), collection_name=collection_name)
        except Exception as e:
            return str(e), 500

    return render_template('search.html', collection_name=collection_name)




if __name__ == '__main__':
    app.run(debug=True)
