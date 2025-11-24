import os
import requests
import json
from fuzzywuzzy import process
from flask import Flask, jsonify, render_template, request
import dotenv

# Load env variables
dotenv.load_dotenv()

# API Keys
DATA_GOV_API_KEY = os.getenv("api_mandi")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not DATA_GOV_API_KEY:
    raise RuntimeError("❌ Missing api_mandi (DATA_GOV_API_KEY). Please add it to environment.")

USE_AGENT = bool(GOOGLE_API_KEY)

BASE_URL = "https://api.data.gov.in/resource/32e934d1-3c2b-4a24-9b3a-7fc4f1fc9de8"


# ------------------------- TOOL FUNCTION ----------------------------------------
from langchain.tools import tool

@tool
def market_price(query: str) -> str:
    """
    Fetches agriculture mandi prices via data.gov.in API
    Query format: "price of tomato in maharashtra"
    """
    query = query.lower()
    if " in " not in query:
        return json.dumps({"error": "Format error. Use 'price of [commodity] in [state]'"})

    commodity, state = query.replace("price of ", "").split(" in ")
    commodity, state = commodity.strip(), state.strip()

    params = {
        "api-key": DATA_GOV_API_KEY,
        "format": "json",
        "filters[state]": state.title(),
        "filters[commodity]": commodity.title(),
        "limit": 100
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
        return json.dumps({"error": f"Unexpected error: {e}"})


# ------------------------- AGENT SETUP -----------------------------------------
agent_executor = None

if USE_AGENT:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.agents import AgentExecutor
        from langchain.agents.tool_calling import create_tool_calling_agent
        from langchain.prompts import PromptTemplate

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0
        )

        react_prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            template="""
You are an agriculture mandi price assistant helping farmers.
Use tool 'market_price' when needed to get real mandi data from the Govt website.
If no data found, say: "No data found for this crop or location."

{agent_scratchpad}
User Query: {input}
"""
        )

        agent = create_tool_calling_agent(
            llm=llm,
            tools=[market_price],
            prompt=react_prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=[market_price],
            verbose=True,
            handle_parsing_errors=True
        )

        print("🌟 Gemini Agent Enabled - Hybrid Mode Active")

    except Exception as e:
        print(f"⚠ Agent initialization failed. Running tool-only mode: {e}")
        agent_executor = None


# ------------------------- FLASK SERVER -----------------------------------------
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
    question = data.get("question")

    try:
        if agent_executor and question:
            print("🤖 Using AI agent for natural question")
            response = agent_executor.invoke({"input": question})
            return jsonify({"type": "agent", "response": response})

        if not commodity or not state:
            return jsonify({"error": "Commodity & State are required"}), 400

        user_input = f"price of {commodity} in {state}"
        result = market_price.invoke(user_input)
        data = json.loads(result)

        if "error" in data:
            return jsonify({"error": data["error"]})

        records = data.get("records", [])

        if market:
            names = [rec.get("market", "") for rec in records]
            best_match, _ = process.extractOne(market, names)
            records = [rec for rec in records if rec.get("market") == best_match]

        if not records:
            return jsonify({"error": "No records found"}), 404

        formatted_records = [{
            "Market": rec.get("market", ""),
            "Min Price (₹)": rec.get("min_price", ""),
            "Max Price (₹)": rec.get("max_price", ""),
            "Modal Price (₹)": rec.get("modal_price", ""),
            "Date": rec.get("arrival_date", "")
        } for rec in records]

        return jsonify({
            "type": "table",
            "data": {
                "crop": data.get("crop", ""),
                "state": data.get("state", ""),
                "records": formatted_records
            }
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
