from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

load_dotenv()

search_tool = TavilySearch(max_results=3)

tools = [search_tool]

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.8)
model_with_tools = writer_llm.bind_tools(tools)
reviewer_model = ChatGroq(model="openai/gpt-oss-120b",temperature=0.1)

class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    feedback: str
    approval: bool
    attempt: int

def researcher_node(state: State) -> dict:
    topic = state["topic"]

    prompt = f"""
Research the following topic for a LinkedIn post:

{topic}

Find:
- Recent and relevant information
- Important facts or statistics
- Credible sources
- Recent developments
- Useful examples

Return the research results that the writer can use.
"""

    response = model_with_tools.invoke(prompt)

    return {
        "messages": [response]
    }

def writer_node(state: State) -> dict:
    feedback = state.get("feedback", "")
    topic = state["topic"]
    research = state["messages"][-1].content
    previous_draft = state.get("draft", "")

    system_prompt = """
You are an expert LinkedIn Content Strategist and Copywriter.

Your task is to write a complete, high-quality LinkedIn post.

Rules:
- Start with a strong hook.
- Use short paragraphs and good spacing.
- Provide useful insights.
- End with an engaging CTA.
- Be professional and conversational.
- Avoid generic AI buzzwords.

Do not perform additional searches after receiving research results.

Return ONLY the completed LinkedIn post.
"""

    if not feedback:
        user_prompt = f"""
Write a complete LinkedIn post about:

{topic}

Research Results : {research}
Use the research to create the post.
Return only the completed LinkedIn post.
"""
    else:
        user_prompt = f"""
Revise the LinkedIn post about:
{topic}

Previous Draft:
{previous_draft}

Reviewer feedback:
{feedback}

Create a new complete LinkedIn post that addresses
all of the reviewer's feedback which was presen in the previous draft.
The previous draft is given for reference.

Do not explain the changes.
Return ONLY the revised LinkedIn post.
"""

    messages = [
        ("system", system_prompt),
        ("human", user_prompt)
    ]

    response = writer_llm.invoke(messages)

    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

def extract_draft(state: State) -> dict:
    """
    Extracts the final writer response as the LinkedIn draft.
    This node is reached only after the writer has finished
    any required tool calls.
    """

    draft = state["messages"][-1].content

    attempt = state.get("attempt", 0) + 1

    print(f"\n--- Generated Draft Attempt {attempt} ---")
    print(draft)
    print("-------------------------------------------\n")

    return {
        "draft": draft,
        "attempt": attempt
    }

def reviewer_node(state: State) -> dict:
    """
    Reviews the generated LinkedIn post and either approves it
    or provides feedback for revision.
    """

    topic = state["topic"]
    draft = state["draft"]

    prompt = f"""
You are a strict LinkedIn Content Director reviewing a
LinkedIn post.

Target Topic:
{topic}

Post Draft:
-------------------------
{draft}
-------------------------

Evaluate the post using these criteria:

1. Hook Quality
   Is the first line compelling enough to make readers
   want to continue reading?

2. Formatting and Readability
   Does it use short paragraphs, line breaks, and
   appropriate formatting?

3. Substance and Accuracy
   Does it provide meaningful insights and avoid
   unsupported or generic claims?

4. Engagement
   Does it end with a clear and thought-provoking CTA?

5. Tone and Style
   Is it professional, authentic, conversational, and
   free from unnecessary buzzwords?

Output EXACTLY in one of these formats:

STATUS: APPROVED
FEEDBACK: Ready to publish.

OR:

STATUS: REVISED
FEEDBACK:
[Give specific feedback only if the post has a meaningful problem
that must be fixed before publishing.]

IMPORTANT:
- Approve the post if it is already good enough to publish.
- Do NOT request improvements just because the post could be made slightly better.
- Do NOT require perfection.
- Do NOT keep suggesting optional improvements.
- Use REVISED only when there is a significant problem with the
  hook, readability, accuracy, substance, tone, or CTA.
- If all major criteria are satisfied, return STATUS: APPROVED.
- The goal is to decide whether the post is publishable, not to
  continuously improve it.
"""

    response = reviewer_model.invoke(prompt)

    result = response.content.strip()

    is_approved = "STATUS: APPROVED" in result.upper()

    if "FEEDBACK:" in result:
        feedback = result.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = result

    print(f"Reviewer Approval: {is_approved}")
    print(f"Reviewer Feedback:\n{feedback}\n")

    return {
        "feedback": feedback,
        "approval": is_approved
    }

def call_tools(state: State):

    message = state["messages"][-1]

    if getattr(message, "tool_calls", None):
        return "tools"

    return "writer"

def stop_loop(state: State):

    approval = state["approval"]
    attempt = state["attempt"]

    if approval:
        return END

    if attempt >= 3:
        return END

    return "writer"


graph = StateGraph(State)

graph.add_node("research", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract", extract_draft)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "research")
graph.add_conditional_edges( "research", call_tools)
graph.add_edge("tools", "writer")
graph.add_edge("writer", "extract")
graph.add_edge("extract", "reviewer")
graph.add_conditional_edges("reviewer", stop_loop)

app = graph.compile()

def main():

    topic = input("Enter the LinkedIn post topic: ")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "feedback": "",
        "approval": False,
        "attempt": 0,
    }

    print("\nStarting LinkedIn Content Agent...")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL LINKEDIN POST")
    print("=" * 60)

    print(final_state["draft"])

    print("\n" + "=" * 60)
    print(f"Attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['approval']}")
    print("=" * 60)


if __name__ == "__main__":
    main()