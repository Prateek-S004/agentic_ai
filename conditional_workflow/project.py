from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def build_retriever(pdf_path : str):
    loader = PyMuPDFLoader(pdf_path)
    document = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100)
    
    chunks = splitter.split_documents(document)
    
    vector_store = FAISS.from_documents(
        documents = chunks,
        embedding = embeddings
    )
    
    return vector_store.as_retriever(search_kwargs = {"k" : 4})

academic_retriever = build_retriever("")
fee_retriver = build_retriever("")

model = ChatGroq(model = "openai/gpt-oss-120b", temperature = 0.5)

class State(TypedDict):
    stream : str
    messages : Annotated[list, add_messages]
    category : str
    answer : str

def classifier_node(state : State) -> dict:
    """Look at the user message and decide if the query is regarding academic, fee or general"""
    
    last_message = state["messages"][-1].content
    
    prompt = f"""You are an intent classification system for a university helpdesk. 

    Analyze the following user query and classify it into EXACTLY ONE of these three categories:
    - academic
    - fee
    - general

    Rules:
    1. Output ONLY the single category word ("academic", "fee", or "general").
    2. Do NOT output punctuation, quotes, introduction, or explanation.
    3. If uncertain, default to "general".

    User Query:
    {last_message}"""

    response = model.invoke(prompt)
    category = response.content.strip().lower()
    
    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    
    return {"category" : category}

def academic_node(state : State) -> dict:
    """Handles academic-related queries such as courses, exams, grades, and syllabus."""
    
    query = state["messages"][-1].content  
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return {"answer" : context}

def fee_node(state : State) -> dict:
    """Handles fee-related inquiries such as tuition payment schedules, financial aid, installments, and payment methods."""
    
    query = state["messages"][-1].content
    docs = fee_retriver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return {"answer" : context}

def general_node(state : State) -> dict:
    """Handles general campus inquiries such as library hours, campus facilities, events, and general administrative services."""
    
    return {"answer" : "No_Retrieval_Needed"}


def response_node(state : State) -> dict:
    """Generates a response using retrieved context when retrieval is required, or falls back to direct assistance."""
    
    query = state["messages"][-1].content 
    stream = state.get("stream", "Unknown")
    context = state["answer"]

    if context == "No_Retrieval_Needed":
        prompt = f"""You are a helpful campus information assistant at a university helpdesk.

        Your goal is to assist students with general campus queries including campus navigation, library hours, student club activities, housing services, parking permits, and general administrative contact details.

        Provide a friendly, helpful, and clear response to the student's question. If specialized assistance is required, direct them to the appropriate department or office on campus.

        Student Query:
        {query}"""
        
    else:
        prompt = f"""You are an accurate, helpful university helpdesk assistant.

        Answer the student's query based strictly on the provided retrieved context below. 

        Academic Stream: {stream}

        Context Information:
        {context}

        Student Query:
        {query}

        Instructions:
        1. Base your answer primarily on the Context Information provided.
        2. If the Context Information does not contain enough information to fully answer the query, clearly state what is missing and advise the student on where to direct their request.
        3. Keep the tone professional, polite, and directly relevant to the student's academic stream ({stream})."""

    response = model.invoke(prompt) 
    
    return {"messages" : [("ai" , response.content.strip())]}

def router(state : State):
    if state["category"] == "academic":
        return "academic"
    elif state["category"] == "fee":
        return "fee"
    else:
        return "general"

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic", academic_node)
graph.add_node("fee", fee_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

graph.add_edge(START, "classifier")
graph.add_conditional_edges("classifier", router)
graph.add_edge("academic", "response")
graph.add_edge("fee", "response")
graph.add_edge("general", "response")
graph.add_edge("response", END)

app = graph.compile()

if __name__ == "__main__":
    print("=" * 75)
    print("Welcome to the College Assistant")
    print("=" * 75)

    stream_input = input("Which programme are you in? (e.g., Computer Science, Mechanical): ").strip()
    
    print(f"\nWelcome, {stream_input} student! How can I help you today? (Type 'exit' or 'quit' to stop)\n")

    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_query:
            continue

        initial_state: State = {
            "stream": stream_input,
            "messages": [HumanMessage(content=user_query)],
            "category": "",
            "answer": ""
        }

        output = app.invoke(initial_state)
        ai_response = output["messages"][-1].content
        print(f"\nAssistant: {ai_response}")