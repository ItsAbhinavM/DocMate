from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)

response = llm.invoke("What's the capital of France?")
print(response.content)
