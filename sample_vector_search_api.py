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

# Route for the upload page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return 'No PDF file provided', 400

        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return 'No file selected', 400

        filename = secure_filename(pdf_file.filename)
        collection_name = os.path.splitext(filename)[0]  # Use the file name as the collection name
        pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Convert PDF to text
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_text = convert_pdf_to_text(pdf_path)

        # Chunk the text and push to Qdrant
        chunks = chunk_text(pdf_text)
        total_chunks = len(chunks)
        push_to_qdrant(chunks, collection_name)  # Push all chunks to the collection

        return redirect(url_for('search_page', collection_name=collection_name))

    return render_template('upload.html')


# Search function
def search_ocr_sample(query, collection_name, top_k=3):
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

# Function to search and summarize top 3 chunks
def search_and_summarize(query, collection_name, top_k=3):
    chunks_results = search_qdrant(query, collection_name, top_k=top_k)
    summarized_results = []
    for text, date, score in chunks_results:
        # Summarize the text here, for example:
        summarized_text = text[:100] + "..." if len(text) > 100 else text
        summarized_results.append((summarized_text, date, score))
    return summarized_results


# Route for the search page
@app.route('/search/<collection_name>', methods=['GET', 'POST'])
def search_page(collection_name):
    if request.method == 'POST':
        query = request.form['query']
        try:
            results = search_ocr_sample(query, collection_name)
            print(results)
            return render_template('search.html', query=query, results=results, collection_name=collection_name)
        except Exception as e:
            return str(e), 500

    return render_template('search.html', collection_name=collection_name)


if __name__ == '__main__':
    app.run(debug=True)
