from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

load_dotenv()
model = ChatGroq(model = "openai/gpt-oss-120b", temperature = 0.1)

def merge_scores_dict(existing : dict, newupdate : dict) -> dict:
    if existing is None:
        return newupdate
    return {**existing, **newupdate}
     
class State(TypedDict):
    raw_input : str
    safety_score : Annotated[dict [str, int], merge_scores_dict]
    
def toxicity_node(state: State) -> dict:
    """Evaluates whether the news content contains toxic, offensive, or harmful language.
    
    Args:
        state (State): The current state containing the news content to evaluate.
        
    Returns:
        dict: A dictionary updating 'safety_score' with the toxicity rating (0-100).
    """
    prompt = f"""You are an expert content safety reviewer. Analyze the following news article for toxicity, hate speech, explicit content, or harmful language.

Assign a toxicity score from 0 to 100:
- 0: Completely safe and objective
- 100: Highly toxic, abusive, or harmful

Respond ONLY with a single integer representing the score. Do not include any explanations or extra characters.

News Content:
{state['raw_input']}"""

    response = model.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score": {"toxicity_level": score}}


def cultural_node(state: State) -> dict:
    """Assesses if the news content misinterprets, disrespects, or threatens cultural or religious values.
    
    Args:
        state (State): The current state containing the news content to evaluate.
        
    Returns:
        dict: A dictionary updating 'safety_score' with the cultural risk rating (0-100).
    """
    prompt = f"""You are a media cultural analyst. Review the following news article to check if it misinterprets, disrespects, or poses a social/cultural threat to any group, heritage, or belief system.

Assign a cultural sensitivity risk score from 0 to 100:
- 0: Culturally respectful, objective, or neutral
- 100: Highly offensive, misleading, or threatening to a specific culture/religion

Respond ONLY with a single integer representing the score. Do not include any explanations or extra characters.

News Content:
{state['raw_input']}"""

    response = model.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score": {"cultural_level": score}}


def copyright_node(state: State) -> dict:
    """Checks the news content for potential copyright infringements or uncredited proprietary text.
    
    Args:
        state (State): The current state containing the news content to evaluate.
        
    Returns:
        dict: A dictionary updating 'safety_score' with the copyright risk rating (0-100).
    """
    prompt = f"""You are a legal and media copyright reviewer. Analyze the following news text for potential copyright risks, direct copy-pasting of proprietary material without attribution, or verbatim plagiarism.

Assign a copyright risk score from 0 to 100:
- 0: Original writing or properly attributed content with no clear infringement
- 100: Direct verbatim copyright infringement or plagiarized proprietary text

Respond ONLY with a single integer representing the score. Do not include any explanations or extra characters.

News Content:
{state['raw_input']}"""

    response = model.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score": {"copyright_score": score}}

    
graph = StateGraph(State)

graph.add_node("toxicity", toxicity_node)
graph.add_node("cultural", cultural_node)
graph.add_node("copyright", copyright_node)

graph.add_edge(START, "toxicity")
graph.add_edge(START, "cultural")
graph.add_edge(START, "copyright")

graph.add_edge("toxicity", END)
graph.add_edge("cultural", END)
graph.add_edge("copyright", END)

app = graph.compile()
input = """
    Leaked documents reveal that the national football club secretly adopted 
    a controversial training doctrine derived from ancient sacred rituals of the 
    indigenous High Plateau tribes. Critics claim the ritualistic practices are 
    being mocked and bastardized for corporate gain, while the team's media team 
    copied entire paragraphs word-for-word from an encrypted draft chapter of 
    Dr. Aris Thorne's copyrighted 2021 book, 'Sacred Rhythms of the Plateau', 
    without attribution. Fans online have called the head coach 'a brainless, 
    parasitic fraud who deserves to be driven out of the league.'
    """

sample_state = {
    "raw_input": input,
    "safety_score" : {}
}

result = app.invoke(sample_state)

print(result['safety_score'])