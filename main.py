import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 

load_dotenv()

def verify_setup():
    print("Verifying environment setup...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found. Please check your .env file.")
        return
    if not langsmith_api_key:
        print("Warning: LANGSMITH_API_KEY not found. LangSmith tracing will be disabled.")
    print("Required API keys found.")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("LLM initialized successfully.")
        response = llm.invoke("Hello, world!")
        print("First LLM call successful.")
        print(f"LLM Response: {response.content}")
        print("\nYour environment is set up correctly!")
        if langsmith_api_key:
            print("Check your LangSmith project to see the trace of this run.")

    except Exception as e:
        print(f"An error occurred during LLM call: {e}")
        print("Please check your API key and network connection.")

if __name__ == "__main__":
    verify_setup()