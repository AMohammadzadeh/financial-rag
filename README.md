# FinRAG

FinRAG is a simple question-answering telegram bot that uses retrieved context to generate concise answers. It leverages language models and embeddings to provide accurate responses based on the given context.
The learning material has been extracted from the [Mofid Learning](https://learning.emofid.com/).

## Features

- Uses OpenAI and HuggingFace models for embeddings
- Retrieves relevant context for questions
- Generates concise answers using a language model

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AMohammadzadeh/finrag.git
    cd finrag
    ```

2. Install the required dependencies:
   This project uses `uv` instead of `pip`. After installing uv, you can install the dependencies using the following command:
    ```sh
    uv sync --frozen --no-dev
    ```

## Usage

1. Initialize the embedder and load documents:
    ```python
    from finrag.embedder import Embedder

    emb = Embedder(model_type='openai', persist_directory='db-openai')
    emb.load_documents("knowledge/learning.jsonl", jq_schema=".content | join(' ')")
    emb.embed_documents()
    ```

2. Build the state graph and generate answers:
    ```python
    from finrag.chat import build_graph

    graph = build_graph()
    response = graph.invoke({"question": "What is a supervising broker?"})
    print(response["answer"])
    ```

## Configuration

Set the required variables in a `.env` file to configure your OpenAI API key and Telegram keys.

## License

This project is licensed under the MIT License.