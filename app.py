import os
import requests
from langchain.tools import tool
from fuzzywuzzy import process
import dotenv 
import certifi
from flask import Flask, jsonify, render_template, request
import json

# Optional: BeautifulSoup was imported but unused; remove to keep things lean

# Load environment variables from .env if present
dotenv.load_dotenv()

# API keys
DATA_GOV_API_KEY = os.getenv("api_mandi")  # Agmarknet API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Gemini key for the agent

# Fail fast for Agmarknet key
if not DATA_GOV_API_KEY:
    raise RuntimeError("Missing api_mandi (DATA_GOV_API_KEY). Add it to your .env or environment.")

# Agent is optional: if GOOGLE_API_KEY is missing, we will run in tool-only mode
USE_AGENT = True
if not GOOGLE_API_KEY:
    USE_AGENT = False

BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

@tool
def market_price(query: str) -> str:
    """
    Fetches agricultural market prices from the data.gov.in API.
    The query should be in the format "price of [commodity] in [state]".
    Returns a JSON string with the query and the API response.
    """
    # Basic query parsing
    query = query.lower()
    if " in " not in query:
        return json.dumps({"error": "Invalid query format. Use 'price of [commodity] in [state]'"})

    parts = query.split(" in ")
    commodity = parts[0].replace("price of ", "").strip()
    state = parts[1].strip()

    params = {
        "api-key": DATA_GOV_API_KEY,
        "format": "json",
        "filters[state]": state.title(),
        "filters[commodity]": commodity.title(),
        "limit": 100  # Increased limit to get more results for filtering
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        return json.dumps({
            "crop": commodity,
            "state": state,
            "records": data.get("records", [])
        })

    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {e}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred: {e}"})


# --- Agent setup (LangChain + Gemini) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# Initialize LLM and Agent only if GOOGLE_API_KEY is provided
agent_executor = None
if USE_AGENT:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY, temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=[market_price], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[market_price], verbose=False)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    commodity = data.get("commodity")
    state = data.get("state")
    market = data.get("market")

    if not commodity or not state:
        return jsonify({"error": "Commodity and state are required"}), 400

    # We will call the market_price tool directly.
    # The agent is not strictly necessary for this simple query.
    user_input = f"price of {commodity} in {state}"
    
    try:
        output_text = market_price.invoke(user_input)
        data = json.loads(output_text)

        if "error" in data:
            return jsonify({"error": data["error"]})

        records = data.get("records", [])

        # If a market is specified, filter the results.
        if market:
            # Use fuzzy matching to find the best market match.
            market_names = [rec.get("market", "") for rec in records]
            best_match, _ = process.extractOne(market, market_names)
            
            if best_match:
                records = [rec for rec in records if rec.get("market") == best_match]

        if not records:
            return jsonify({"error": "No records found for this query."})

        # Format records for table display
        formatted_records = []
        for rec in records:
            formatted_records.append({
                "Market": rec.get("market", ""),
                "Min Price (₹)": rec.get("min_price", ""),
                "Max Price (₹)": rec.get("max_price", ""),
                "Modal Price (₹)": rec.get("modal_price", ""),
                "Date": rec.get("arrival_date", ""),
                "Commodity": rec.get("commodity", ""),
                "State": rec.get("state", "")
            })
            
        return jsonify({
            "type": "table",
            "data": {
                "crop": data.get("crop", ""),
                "state": data.get("state", ""),
                "records": formatted_records
            }
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"})

if __name__ == '__main__':
    app.run(debug=True)