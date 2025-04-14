from langgraph.graph import StateGraph, END
from typing import Optional, List, TypedDict
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from PIL import Image
import io
import json
import google.generativeai as genai
from pydantic import BaseModel  # pydantic v2 compatible
# import pprint
import os

import streamlit as st


# Load environment variables from .env file
load_dotenv()

# Retrieve the Gemini API key
# gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_api_key = st.secrets['GEMINI_API_KEY']


# Configure Google Gemini API
genai.configure(api_key=gemini_api_key)

# 1. Define the State for the Chat Workflow
class ChatState(TypedDict):
    """
    Represents the state of the conversation at any point in the workflow.
    """
    image: Optional[bytes]  # Optional image data (for issue detection)
    message: str  # The current user message
    location: Optional[str]  # Optional user location for context
    stage: Optional[str]  # Current stage of the conversation (not actively used in the provided code)
    category: Optional[str]  # Predicted category of the user's query ("issue_detection" or "faq")
    response: Optional[str]  # The AI's response to the user
    conversation_history: List[dict]  # History of the conversation (not actively used in the provided code)
    intermediate_responses: List[str]  # List of responses from intermediate steps
    requires_human_input: Optional[bool]  # Flag indicating if human input is needed
    clarification_question: Optional[str]  # The question to ask the user for clarification
    clarification_attempts: int  # Counter for the number of clarification attempts
    exit_requested: Optional[bool] 

# 2. Node 1: Classifier Agent
def classifier(state: ChatState) -> ChatState:
    """
    Classifies the user's query into "issue_detection" or "faq".
    """
    prompt = f"""

You are a **Classifier Agent** responsible for routing user queries to the correct downstream agent based on the nature of the query.

There are **two agents** you can classify into:

---

### 🛠️ **Agent1: Issue Detection & Troubleshooting Agent (Image + Text)**
**Responsibilities:**
- Accepts **user-uploaded images** of properties, along with **optional textual descriptions**.
- Detects **visible property issues**, such as:
    - Water damage
    - Mold
    - Cracks
    - Poor lighting
    - Broken fixtures
- Provides **troubleshooting suggestions**, such as:
    - “You might need to contact a plumber.”
    - “This looks like paint peeling due to moisture—consider using the anti-damp coating.”
- Can ask **clarifying follow-up questions** to improve diagnostics.

---

### 📜 **Agent2: Tenancy FAQ Agent (Text-only)**
**Responsibilities:**
- Handles **text-based** questions about **tenancy laws**, agreements, **rent**, deposits, and **landlord/tenant responsibilities**.
- **Capable of giving location-specific guidance** if the user's **city or country** is provided.
- Can answer common tenancy-related questions such as:
    - “How much notice do I need to give before vacating?”
    - “Can my landlord increase rent midway through the contract?”
    - “What should I do if the landlord is not returning the deposit?”
- If the user's location or context is **missing**, and it's required for accurate advice, sets `requires_human_input: true`.
- If the query is **ambiguous**, **asks a clarifying question** to determine the correct agent.

---

### 🧠 **Chain-of-Thought (CoT) Reasoning:**
1. **Visual Issue Detection**:
    - Does the query describe or imply a **visible property issue** (e.g., mold, cracks, water leak)?
        - **If yes**, and **an image is provided**, classify as `"issue_detection"` with `missing_image: false`.
        - **If yes**, and **no image is provided**, classify as `"issue_detection"` with `missing_image: true`.
2. **Text-based Tenancy Query**:
    - If the query is **text-based** and relates to **tenancy policies, agreements, rights, or rent**, classify as `"faq"`.
3. **Requires Clarification**:
    - If the query **lacks sufficient context** (e.g., missing location or unclear question), set `requires_human_input: true`.

---

### 📦 **Few-shot Examples:**

**Example 1:**
User Query: "My wall has black patches and water is dripping from the ceiling."
→ `predicted_category`: `"issue_detection"`, `missing_image`: `true`, `requires_human_input`: false, `clarification_question`: false

**Example 2:**
User Query: "Is my landlord allowed to increase rent after 6 months in Delhi?"
→ `predicted_category`: `"faq"`, `requires_human_input`: false, `clarification_question`: false

**Example 3:**
User Query: "See this crack. Is it dangerous?" (with image)
→ `predicted_category`: `"issue_detection"`, `missing_image`: `false`, `requires_human_input`: false, `clarification_question`: false

**Example 4:**
User Query: "What can I do if my landlord won’t return the deposit?"
→ `predicted_category`: `"faq"`, `requires_human_input`: false, `clarification_question`: false

**Example 5:**
User Query: "The place I’m renting has mold and the bathroom lights don’t work." (No image)
→ `predicted_category`: `"issue_detection"`, `missing_image`: `true`, `requires_human_input`: false, `clarification_question`: false

**Example 6:**
User Query: "I need advice on rent rules."
→ `predicted_category`: `"faq"`, `requires_human_input`: true, `clarification_question`: true

**Example 7:**
User Query: "How much notice do I need to give before vacating?"
→ `predicted_category`: `"faq"`, `requires_human_input`: false, `clarification_question`: false

---

```python
### ✅ **Output Format**:
Respond in **raw JSON only**, no extra explanation.

Example:
{{
    "title": "Analyzing",
    "content": "Detecting the nature of the query...",
    "next_action": "continue",
    "predicted_category": "issue_detection",
    "requires_human_input": false,
    "missing_image": false,
    "User_Query": "{state['message']}",
    "clarification_question": false
}}
User Query:
"{state['message']}"

"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip("json").strip()

    print("🔍 Cleaned Classifier Response:\n", raw)

    try:
        parsed_result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"❌ Gemini Classifier still didn't return valid JSON. Got:\n{raw}")

    state["category"] = parsed_result["predicted_category"]
    state["intermediate_responses"].append(parsed_result)
    return state

# 3. Node 2: Critic Agent
def critic(state: ChatState) -> ChatState:
    """
    Reviews and validates the classification made by the Classifier Agent.
    """
    prompt = f"""
You are Agent2 (Critic).

Your role is to critically review and validate the classification made by Agent1.
You must check if Agent1 correctly categorized the user's query based on the rules below.

---

### Guidelines for Decision-Making:

1. If the query involves a **visible property issue** (e.g., mold, cracks, poor lighting), it should be classified as `"issue_detection"`.
    - If **no image** is provided, set `"missing_image": true`.
    - If an image **is present**, set `"missing_image": false`.

2. If the query is a **text-based legal question** (e.g., tenancy laws, rent disputes, landlord duties), classify as `"faq"`.
    - IMPORTANT: For FAQ queries, you must strictly check if it includes the necessary location information.
    - If the query is about legal rights, tenant obligations, or landlord responsibilities AND doesn't specify a location, you MUST set `"requires_human_input": true` and `"clarification_question": true`.
    - Only mark FAQ queries as not requiring clarification if they either:
        a) Already contain specific location information, or
        b) Are general questions that can be answered without location-specific context

3. If the query **does not provide enough details** to decide the category or give accurate advice (e.g., missing location, vague question), set `"requires_human_input": true` and set `"clarification_question": true`.

You must **correct** any misclassification by Agent1 if needed.

---

Agent1 prediction: "{state['category']}"
User Query: "{state['message']}"

Respond only in **raw JSON**, no extra explanation.

### Output Format:

```json
{{
    "title": "Final Review",
    "content": "Verified category.",
    "next_action": "final_answer",
    "predicted_category": "issue_detection",  # Choose one category based on your validation
    "requires_human_input": false,  # MUST set to true for FAQ queries without location info
    "missing_image": false,
    "clarification_question": false,  # MUST set to true for FAQ queries without location info
    "User_Query": "{state['message']}"   # User input
}}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip("json").strip()

    print("🔍 Cleaned Critic Response:\n", raw)

    try:
        parsed_result = json.loads(raw)
        final_category = parsed_result["predicted_category"]
        state["category"] = final_category
        state["intermediate_responses"].append(parsed_result)
    except json.JSONDecodeError:
        print(f"⚠️ Critic returned invalid JSON: {raw}. Using classifier's prediction.")
        state["category"] = state.get('intermediate_responses', [{}])[-2].get('predicted_category') # Use classifier's previous prediction
        state["intermediate_responses"].append({"error": "Invalid Critic JSON", "raw_output": raw})

    return state

# 4. Node 3: Image Analyzer Agent
def ask_gemin_ImageAnalyzer(state: ChatState) -> ChatState:
    """
    Analyzes property images for issues and provides troubleshooting.
    """
    image_bytes = state.get("image")
    user_text = state.get("message", "")

    if not image_bytes:
        state["response"] = "Please upload an image of the issue so I can help you diagnose it better."
        return state

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f'''
    You are Agent1 (Issue Detection & Troubleshooting Agent).

    Your role is to analyze user-submitted **images of properties**, optionally accompanied by **textual context**, to detect **visible property issues** and offer basic troubleshooting suggestions.

    ---

    💡 **Key Rules:**
    1. An **image is required** to process the query. If no image is provided, respond with a message asking the user to upload one.
    2. **Textual context** (e.g., “What’s wrong with this wall?” or “This is near the bathroom.”) can help improve diagnosis but is optional.
    3. You are allowed to ask **clarifying follow-up questions** if the image is unclear or context is insufficient.
    4. Focus on **common property defects** like:
        - Water damage
        - Cracks in walls or ceilings
        - Mold or dampness
        - Broken lighting or fixtures
        - Paint peeling
        - Poor insulation or ventilation

    ---

    🧠 **Responsibilities:**
    - Detect any visible issues in the uploaded property image.
    - Give a **brief explanation** of the problem and suggest **basic next steps** or remedies.
    - If unsure, ask for **additional context or a clearer image**.

    ---

    📌 **Example Interactions (Few-Shot):**

    **Example 1:**
    User: “What’s wrong with this wall?” *(Image uploaded)*
    Agent1: “It appears there is mold growth near the ceiling. This might be due to high humidity or a leak. I recommend checking for water seepage and using a dehumidifier.”

    **Example 2:**
    User: *(Image of cracked ceiling, no text)*
    Agent1: “There seems to be a structural crack in the ceiling. This could indicate settling or moisture damage. I recommend having a structural engineer or contractor assess it.”

    **Example 3:**
    User: “My light doesn’t work.” *(No image)*
    Agent1: “Please upload a photo of the light or the affected area so I can assist you better.”

    ---

    ✅ **Action Rules**:
    - If image is missing → Ask user to upload an image.
    - If image is unclear → Ask clarifying questions.
    - If issue is visible → Describe the issue and suggest basic troubleshooting.

    Now respond to:
        User: {state['image']} Optional:{state['message']}
    '''


    if user_text:
        response = model.generate_content([image, user_text], stream=False)
    else:
        response = model.generate_content([image], stream=False)

    state["response"] = response.text.strip()
    return state

# 5. Node 4: Tenancy FAQ Agent
def tenancy_faq(state: ChatState) -> ChatState:
    """
    Handles text-based queries related to tenancy and rental concerns.
    """
    prompt = f'''
    You are Agent 2 (Tenancy FAQ Agent), responsible for handling text-based queries related to tenancy and rental concerns.

Responsibilities:
● Answer frequently asked questions about:
    - Tenancy laws and policies
    - Landlord/tenant responsibilities
    - Rent agreements, eviction, deposit issues, and rental rights
● Provide location-specific legal guidance when the user's region or city is mentioned.
● If the query lacks enough context (e.g., no location mentioned for legal questions), ask a clarifying follow-up.
● Ensure that your responses are concise, clear, and legally sound.

Expected Behavior:
1. Identify the legal or procedural topic in the question.
2. If **location is missing** and required to give a proper response, politely ask the user to share it.
3. Provide actionable, easy-to-understand advice.
4. Never hallucinate legal outcomes—only respond based on the user's input.

---

**Few-shot Examples:**

**User:** "Can my landlord evict me without notice?"
**Agent 2:** "In most regions, landlords are required to give written notice before eviction unless there's an urgent reason like non-payment or illegal activity. Could you tell me your city or country so I can give a more specific answer?"

**User:** "How long does a landlord have to return the deposit?"
**Agent 2:** "In many areas, landlords must return the security deposit within 30 days of the tenant moving out. Let me know your region for more precise info."

**User:** "Do I have to pay for repairs if something breaks?"
**Agent 2:** "It depends on the cause of the damage. Typically, landlords are responsible for structural and major appliance repairs. If you caused the damage, you may be liable. Would you like to share your location so I can check your local laws?"

---

    User question: {state["message"]}
If the question lacks legal or regional clarity, politely ask for the user's location. Use factual, concise responses.
    '''
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)
    state["response"] = response.text.strip()
    return state

# 6. Node 5: Human Intervention Handler
def handle_human_input(state: ChatState, ask_func=input) -> ChatState:
    """
    Handles cases where human input is required. Prompts the user and updates the state.
    """
    if state.get("requires_human_input", False):
        clarification = state.get("clarification_question")
        if not clarification:
            clarification = "Can you provide more details to proceed further?"

        print(f"\n🔍 System: Clarification required (Attempt {state.get('clarification_attempts', 0) + 1}).")
        print(f"🧠 Asking user: {clarification}")

        user_input = ask_func(f"{clarification}\n→ ")

        if user_input.lower() == "exit":
            print("Exiting clarification loop.")
            state["response"] = "User chose to exit the conversation."
            state["requires_human_input"] = False
            state["exit_requested"] = True  # Flag for explicit exit
            state["clarification_attempts"] = state.get("clarification_attempts", 0) + 1
            print(f"DEBUG (handle_human_input - exit): exit_requested={state.get('exit_requested')}, requires_human_input={state.get('requires_human_input')}")
            return state

        state["message"] = f"{state['message']} {user_input}"
        state["requires_human_input"] = False  # Reset the flag
        state["clarification_question"] = None
        state["clarification_attempts"] = state.get("clarification_attempts", 0) + 1
        print(f"DEBUG (handle_human_input - input): exit_requested={state.get('exit_requested')}, requires_human_input={state.get('requires_human_input')}")
        return state
    return state

# 7. Node 6: Answer Verification Agent
def answer_verification(state: ChatState) -> ChatState:
    """
    Reviews the agent's response to check if it adequately addresses the user's query.
    """
    print(f"DEBUG (answer_verification - entry): exit_requested={state.get('exit_requested')}, requires_human_input={state.get('requires_human_input')}")
    if state.get("exit_requested", False):
        print("DEBUG (answer_verification): exit_requested is True, returning state to end flow.")
        return state

    response_text = state.get("response", "")
    prompt = f'''
    You are a helpful assistant reviewing an AI agent's response.
    Your task is to determine if the user's original query has been adequately addressed.

    Answer 'yes' if the response seems complete and doesn't require further user input, otherwise answer 'no'.

    ---
    Response:
    "{response_text}"
    ---
    '''
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)
    model_output = response.text.strip().lower()

    if "no" in model_output and "yes" not in model_output:
        state["requires_human_input"] = True
        state["clarification_question"] = "The previous response might need more detail. How can I further assist you?"
    else:
        state["requires_human_input"] = False
        state["clarification_question"] = None

    print(f"DEBUG (answer_verification - exit): requires_human_input={state.get('requires_human_input')}, clarification_question={state.get('clarification_question')}")
    return state

# 8. Workflow Setup
workflow = StateGraph(ChatState)

# 9. Add Nodes to the Workflow
workflow.add_node("classifier", classifier)
workflow.add_node("critic", critic)
workflow.add_node("agent_1", ask_gemin_ImageAnalyzer)
workflow.add_node("agent_2", tenancy_faq)
workflow.add_node("answer_verification", answer_verification)
workflow.add_node("human_intervention", handle_human_input)

# 10. Define Basic Edges (Linear Flow)
workflow.add_edge("classifier", "critic")
workflow.add_edge("agent_1", "answer_verification")
workflow.add_edge("agent_2", "answer_verification")

# 11. Define Conditional Edges for Human Intervention
def route_from_human_intervention(state):
    """Routes based on whether the user requested to exit or needs further clarification."""
    if state.get("exit_requested", False):
        return state.get("category", "__end__")  # Route to predicted category or end
    return "classifier"  # Go back to classifier for new input

workflow.add_conditional_edges(
    "human_intervention",
    route_from_human_intervention,
    {
        "classifier": "classifier",
        "issue_detection": "agent_1",
        "faq": "agent_2",
        "__end__": END  # Allow direct exit from human intervention
    }
)

# 12. Define Conditional Edges for Critic Output
def get_routing_decision(state: ChatState) -> str:
    """Routes based on the critic's assessment and predicted category."""
    if state.get("requires_human_input", False):
        return "human_intervention"
    return state["category"]

workflow.add_conditional_edges("critic", get_routing_decision, {
    "issue_detection": "agent_1",
    "faq": "agent_2",
    "human_intervention": "human_intervention"
})

# 13. Define Conditional Edges for Answer Verification
def verify_answer_and_route(state):
    """Routes based on whether the answer is sufficient or requires human intervention, or if the user exited."""
    if state.get("exit_requested", False):
        return "__end__"
    elif state.get("requires_human_input", False):
        return "human_intervention"
    else:
        return "__end__"

workflow.add_conditional_edges("answer_verification", verify_answer_and_route, {
    "human_intervention": "human_intervention",
    "__end__": END
})

# 14. Set the Entry Point of the Workflow
workflow.set_entry_point("classifier")

# 15. Compile the Workflow
chain = workflow.compile()

# 16. Example Usage
def load_image(filename):
    with open(filename, "rb") as f:
        return f.read()

if __name__ == "__main__":
    img_bytes = load_image(r"D:\Multi_Agentic_Real_Estate_Chatbot\defect1.jpg")
    # Test Case 2: FAQ with location (Simulating a query that might lead to clarification)
    result_2 = chain.invoke(ChatState(
        image=img_bytes,
        message="“diagnose better.?",
        location=None,
        stage=None,
        category=None,
        response=None,
        conversation_history=[],
        intermediate_responses=[],
        requires_human_input=None,
        clarification_question=None,
        clarification_attempts=0,
        exit_requested=False
    ))
    print("\n--- Test Case 2: FAQ with potential for Clarification ---")
    response_text_2 = result_2.get('response', '').strip()
    print(response_text_2)

    # You can add more test cases here to evaluate different scenarios,
    # including the "exit" command within the human intervention loop.
    # To test the "exit" flow, you would need to simulate the interaction
    # where 'requires_human_input' becomes True, leading to the
    # 'human_intervention' node, and then simulate the user typing 'exit'.
    # This often requires a more interactive testing setup or mocking
    # the input function.
