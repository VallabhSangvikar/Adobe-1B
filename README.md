# Adobe India Hackathon 2025 - Problem Statement 2

This project is a solution for the second problem statement of the Adobe India Hackathon 2025. It leverages a language model to analyze a PDF document based on a given persona and task, extracting relevant information and providing a structured output.

## Approach

The solution follows a multi-step process:

1.  **Model and Tokenizer Loading**: It starts by downloading and loading the TinyLlama-1.1B-Chat-v1.0-GGUF model and its tokenizer.
2.  **Document Processing**: The script loads a PDF from the `input` directory, splits it into manageable chunks, and creates embeddings using the `all-MiniLM-L6-v2` model. These embeddings are then stored in a FAISS vector store for efficient similarity searches.
3.  **Persona and Task Definition**: A predefined persona ("Travel Planner") and a specific task ("Plan a trip of 4 days for a group of 10 college friends.") are used to guide the information extraction process.
4.  **Prompt Engineering and Chain Execution**: A custom prompt template is created to frame the query for the language model. The `LLMChain` from LangChain is used to run the model with the persona, task, and retrieved context as inputs.
5.  **Output Generation**: The response from the language model is formatted into a JSON structure that includes metadata, extracted sections, and a sub-section analysis. The final output is saved in the `output` directory.

## Models and Libraries

* **Language Model**: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
* **Embeddings Model**: `all-MiniLM-L6-v2`
* **Libraries**:
    * `langchain_community`, `langchain`
    * `pypdf`
    * `llama-cpp-python`
    * `transformers`, `accelerate`, `bitsandbytes`
    * `faiss-cpu`, `sentence-transformers`

## How to Build and Run

### Prerequisites

* Docker

### Build the Docker Image

To build the Docker image, run the following command from the root of the project directory:

```sh
docker build -t adobe-hackathon-ps2 .