from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
class pipeline(TypedDict):
    raw_input : str
    editor : str
    scriptwriter : str
    output : str
    
model = ChatGroq(model = "openai/gpt-oss-120b", temperature = 0.6)

def editor_node(state : pipeline) -> dict:
    """Cleans up grammar, removes typos and refines the tone of the input text"""
    
    prompt = f"""You are an expert content script editor. Polish the provided script by fixing grammar, spelling, and typos, improving tone and flow, and ensuring the pacing and dialogue sound natural for audio/video delivery without changing the core message.
    Raw_input: {state['raw_input']}"""
    
    response = model.invoke(prompt)
    
    return {"editor" : response.content}

def scriptwriter_node(state : pipeline) -> dict:
    """Transforms raw ideas, outlines, or briefs into an engaging, production-ready script."""

    prompt = f"""You are an expert scriptwriter. Convert the provided topic, outline, or raw notes into a captivating, well-structured script optimized for spoken delivery, with clear pacing, strong visual cues, and an engaging narrative hook.
    Edited text: {state['editor']}"""
    
    response = model.invoke(prompt)
    
    return {"scriptwriter" : response.content}

def translator_node(state: pipeline) -> dict:
    """Translates and adapts the input text into natural, modern Hinglish suitable for engaging casual content."""

    prompt = f"""You are an expert Hinglish content translator. Translate the provided text into modern, conversational Hinglish (a natural mix of Hindi and English written in the Roman script). Keep the tone engaging, relatable, and easy to read aloud, maintaining the original meaning while making it sound authentic to Indian digital audiences.
    Scripted text: {state['scriptwriter']}"""
    
    response = model.invoke(prompt)
    
    return {"output" : response.content}

graph = StateGraph(pipeline)

graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

app = graph.compile()

result = app.invoke({
    "raw_input" : {"topic": "5 simple habits to improve daily focus",
    "target_audience": "Tech professionals and students",
    "tone": "Casual, energetic, and practical",
    "raw_notes": "1. Put phone in another room. 2. Use 25-min Pomodoro. 3. Drink water first thing in morning. 4. Block calendar for deep work. 5. Take 5-min walk without phone.",
    "format": "YouTube Reel / Instagram Short (60 seconds)"}
}) 

print(result['output'])