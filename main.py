import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
import json
import time

# --- 1. Load Model and Tokenizer ---
model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
model_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

print(f"Downloading model: {model_file} from {model_name}...")
model_path = hf_hub_download(repo_id=model_name, filename=model_file)

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_ctx=2048
)

try:
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)
except Exception as e:
    print(f"Could not load the specific tokenizer: {e}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

# --- 2. Process Documents ---
input_dir = "/app/input"
output_dir = "/app/output"
pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

for pdf_file in pdf_files:
    pdf_path = os.path.join(input_dir, pdf_file)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    # --- 3. Define Persona and Task ---
    persona = "Travel Planner"
    task = "Plan a trip of 4 days for a group of 10 college friends."

    # --- 4. Create Prompt and Run Chain ---
    template = """
    You are a helpful assistant with the following persona: {persona}

    Your task is: {task}

    Use the following context to answer:
    {context}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["persona", "task", "context"],
        template=template,
    )

    qa_llm_chain = LLMChain(llm=llm, prompt=prompt)

    retrieved_docs = db.similarity_search(task)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    start_time = time.time()
    response = qa_llm_chain.run({
        "persona": persona,
        "task": task,
        "context": context,
    })
    end_time = time.time()

    # --- 5. Format and Save Output ---
    output_data = {
        "metadata": {
            "input_documents": [pdf_file],
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "extracted_section": [],
        "sub-section_analysis": [
            {
                "document": pdf_file,
                "page_number": 1, 
                "refined_text": response
            }
        ]
    }

    output_filename = os.path.splitext(pdf_file)[0] + ".json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Processing complete. Output saved to {output_path}")