import pickle
import faiss
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import openai

# Flask app setup
app = Flask(__name__)

# Set OpenAI API key
openai.api_key ="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"  # Replace with your OpenAI key

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables for FAISS index and chunks
index = None
chunks = None

# Step 1: Load Pickle File
def load_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

# Load preprocessed data
data = load_from_pickle("processed_data_1.pkl")
chunks = data['chunks']
index = data['index']

# Step 2: Search Query
def search_query(query, index, chunks, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)  # Retrieve top 3 chunks
    return [chunks[i] for i in indices[0]] 

# Step 3: Generate Response
def generate_response(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Route to the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle both JSON and form queries
@app.route('/query', methods=['POST'])
def query_pdf():
    if request.is_json:  # JSON-based API request
        query = request.json.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        relevant_chunks = search_query(query, index, chunks, model)
        response = generate_response(query, relevant_chunks)
        return jsonify({"response": response})
    else:  # Form-based request from the web interface
        query = request.form.get('query')
        if not query:
            return render_template('index.html', error="Please enter a query.")
        relevant_chunks = search_query(query, index, chunks, model)
        response = generate_response(query, relevant_chunks)
        return render_template('result.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
