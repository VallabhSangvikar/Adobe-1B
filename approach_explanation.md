This file provides a more detailed explanation of the methodology used in your solution.

```markdown
# Approach Explanation

Our solution for Problem Statement 2 of the Adobe India Hackathon 2025 is designed to function as an intelligent document analyst. It takes a PDF document, a persona, and a task as input, and produces a structured JSON output with relevant extracted information.

## Methodology

The core of our approach is built around the **Retrieval-Augmented Generation (RAG)** pipeline, which allows the language model to generate responses based on information retrieved from the provided document.

1.  **Document Loading and Chunking**: We start by loading the input PDF using `PyPDFLoader`. To handle large documents and ensure that the context provided to the language model is relevant, we split the document into smaller chunks of 500 characters with a 50-character overlap using `RecursiveCharacterTextSplitter`.

2.  **Vector Embeddings and Storage**: Each chunk of text is then converted into a numerical representation (embedding) using the `all-MiniLM-L6-v2` model from `sentence-transformers`. These embeddings are stored in a `FAISS` vector store, which allows for efficient similarity searches.

3.  **Information Retrieval**: When a task is provided, we use the `FAISS` vector store to retrieve the most relevant chunks from the document based on semantic similarity to the task description. This ensures that the language model receives only the most pertinent information.

4.  **Prompt Engineering**: We have designed a custom prompt template that provides the language model with the necessary context, including the persona, the task, and the retrieved document chunks. This structured prompt guides the model to generate a response that is tailored to the user's needs.

5.  **Language Model and Generation**: The `TinyLlama-1.1B-Chat-v1.0-GGUF` model, loaded via `LlamaCpp`, is used to generate the final response. The `LLMChain` from LangChain orchestrates the process of passing the prompt and context to the model.

6.  **Output Formatting**: The generated response is then formatted into a structured JSON output that includes metadata about the request, as well as the extracted information. This ensures that the output is machine-readable and easy to parse for downstream applications.

By combining these techniques, our solution is able to effectively "connect the dots" between the user's request and the content of the document, providing a powerful and intelligent document analysis tool.
