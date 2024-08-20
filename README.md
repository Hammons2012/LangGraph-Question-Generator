# Document Retrieval and Analysis Workflow

This project implements a workflow for document retrieval, grading, analysis, and question generation using various LangChain components. The workflow is designed to retrieve documents from specified URLs, grade their relevance to a given topic, analyze the relevant documents, and generate questions based on the analysis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow Overview](#workflow-overview)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project automates the process of document retrieval, relevance checking, analysis, and question generation. It leverages LangChain components such as document loaders, text splitters, vector stores, and language models to achieve this.

## Features

- Document retrieval from specified URLs.
- Document grading for relevance to a given topic.
- Document analysis to extract insights.
- Question generation based on the analyzed document.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher.
- Required Python packages (see [Installation](#installation)).

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Hammons2012/LangGraph-Question-Generator.git
    cd your-repo
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

    Make sure your `requirements.txt` includes the following packages:

    ```txt
    langchain_community
    langchain_experimental
    langchain_core
    langgraph
    ```

## Usage

1. Set the global variables in the script:

    ```python
    topic = "your_topic"
    question_count = 5
    urls = [
        "https://url1.com",
        "https://url2.com"
    ]
    ```

2. Run the script:

    ```sh
    python your_script.py
    ```

## Workflow Overview

1. **Document Retrieval**:
    - Load documents from specified URLs using `WebBaseLoader`.
    - Split the documents into chunks using `RecursiveCharacterTextSplitter`.
    - Store the document chunks in a `Chroma` vector store with `GPT4AllEmbeddings`.

2. **Document Grading**:
    - Use a prompt template to create a grading prompt.
    - Use `ChatOllama` with the `mistral` model to grade the documents.
    - Parse the output to determine if the document is relevant.

3. **Document Analysis**:
    - Use a prompt template to create an analysis prompt.
    - Use `ChatOllama` with the `mistral` model to analyze the documents.
    - Parse the output to get the analyzed document.

4. **Question Generation**:
    - Use a prompt template to create a question generation prompt.
    - Use `ChatOllama` with the `codestral` model to generate questions.
    - Parse the output to get the generated questions.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
