from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class Query(BaseModel):
    """
    A Pydantic model for the input query.
    """
    question: str

# Load dataset
dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")
texts = dataset["text"]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Load model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

@app.get("/")
async def get_current_datetime():
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"current_datetime": current_datetime}

@app.post("/generate")
async def generate_answer(query: Query):
    """
    Generate an answer to a given question based on a relevant document.

    Parameters:
    query (Query): The input query containing the question.

    Returns:
    dict: A dictionary containing the generated answer and the relevant document context.
    """
    if not query.question.strip():
        raise HTTPException(status_code=422, detail="Empty question. Please provide a valid question.")

    # Retrieve relevant documents
    query_vector = vectorizer.transform([query.question])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    relevant_doc_index = similarities.argsort()[-1]
    relevant_doc = texts[relevant_doc_index]

    # Generate answer
    input_text = f"Context: {relevant_doc}\n\nQuestion: {query.question}\n\nAnswer:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"answer": answer, "context": relevant_doc}

