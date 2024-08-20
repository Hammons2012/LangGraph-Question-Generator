from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from typing import Annotated, List, TypedDict
import os
from pprint import pprint
import json


# Global variables
topic = "topic"
question_count = 5
urls = [
    "https://url.com"
]
    
# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
)
retriever = vectorstore.as_retriever()

# LLM(s) used for agents
llm_mistral = ChatOllama(model='mistral', format="json", temperature=0)
llm_codestral = ChatOllama(model='codestral', format='json', temperature=0.5)

# Define agents and LLM - Since local LLMs are being used, function calling is not supported
# This agent/LLM handles grading the retrieved URL to ensure that it actually adheres to the topic.
prompt_grader = PromptTemplate(
    template="""You are a grader tasked with assessing the relevance of a retrieved document to a topic. \n
    The retrieved document is as follows: \n ------- \n {documents} \n ------- \n
    The topic is: {topic} \n
    If the document contains keywords or phrases that are directly related to the topic, grade it as 'relevant'. \n
    The aim is to filter out any retrievals that are clearly unrelated to the topic. \n
    Provide a binary score of 'yes' or 'no' to indicate whether the document is relevant to the question. \n
    The score should be formatted as a JSON object with a single key 'score', and no additional explanation or preamble is needed.""",
    input_variables=["topic", "documents"],
)
chain_retrieval = prompt_grader | llm_mistral | StrOutputParser()

# This agent/LLM handles the analysis of the document
prompt_analyzer = PromptTemplate(
    template="""You are an expert in document analysis, with a specialization in interpreting and extracting insights from provided documents. \n
    A document is provided below: \n ------- \n  {documents} \n ------- \n
    Utilize the information within the document to provide a detailed outline of the contents of the document. \n""",
    input_variables=["documents"]
)
chain_analyzer = prompt_analyzer | llm_mistral | StrOutputParser()

# This agent/LLM handles the code generation, and the related functions to output the code
prompt_question_gen = PromptTemplate(
    template="""You are an expert in generating questions and answers based on an output. \n 
    An output of a document is outlined below:  \n ------- \n  {analyzer_output} \n ------- \n
    Use the provided outline to develop a JSON object of {question_count} questions and answers based on the output. \n
    Ensure that the questions are well-structured, clear, and easy to understand. \n
    Provide the questions as a JSON object, with no additional explanation or preamble.""",
    input_variables=["analyzer_output", "question_count"]
)
chain_question_gen = prompt_question_gen | llm_codestral | StrOutputParser()


# Defining the agent state
class GraphState(TypedDict):
    """
    Represents the state of a graph, including the topic, generation, documents, error messages, and question count.

    Attributes:
        topic (str): Topic of interest that will be passed into the LLMs
        generation (str): The LLM (Language Learning Model) generation the graph is using.
        documents (List[str]): The documents the graph is using as data.
        error (str): Any error message that occurs during the graph's operation.
        question_count (int): The number of question count the graph has gone through.
        analyzer_output (str): The output from the analyzer, which could be insights, summaries, or other forms of processed information.
    """

    topic: str
    generation: str
    documents: List[str]
    error: str
    question_count: int
    analyzer_output: str


# Defining node and edge functions
# Nodes
def retrieval(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    documents = state["documents"]

    # Retrieval
    documents = retriever.invoke(topic)
    return {"documents": documents}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the topic.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO TOPIC---")
    topic = state["topic"]
    documents = state["documents"]
    error = state["error"]

    # Score each doc
    error = "No"
    for d in documents:
        score = chain_retrieval.invoke(
            {"topic": topic, "documents": d}
        )
        grade = json.loads(score)["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            error = "Yes"
            continue
    return {"error": error}


def analyzer(state):
    """
    Analyze the crypto strategy document

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---ANALYZE---")
    documents = state["documents"]

    # RAG generation
    analyzer_output = chain_analyzer.invoke({"documents": documents, "topic": topic})
    return {"analyzer_output": analyzer_output}


def question_creator(state):
    """
    Create questiobns based on the topic
    
    Args:
        state (dict): The current graph state
        
    Returns:
        questions (dict): Dictionary of questions per the topic
    """
    print("---QUESTION GEN---")
    analyzer_output = state["analyzer_output"]
    question_count = state["question_count"]
    
    #RAG Generation
    question_output = chain_question_gen.invoke({"analyzer_output": analyzer_output, "question_count": question_count})
    return {"generation": question_output}


# Creating the graphs and edges
workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("retrieval", retrieval)
workflow.add_node("grader", grade_documents)
workflow.add_node("question_creator", question_creator)

# Edges
workflow.set_entry_point("retrieval")
workflow.add_edge("retrieval", "grader")
workflow.add_edge("grader", "question_creator")
workflow.set_finish_point("question_creator")

# Compile the app
app = workflow.compile()

# Launch the app
inputs = {"topic": topic}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node: '{key}': \nValue: {value}\n")
    pprint("\n---\n")

# Final generation
pprint(value["generation"])