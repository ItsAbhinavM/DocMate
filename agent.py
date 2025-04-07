import builtins

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.tools.render import render_text_description
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

# Define the tools to be used by the agent
tools = []


# Pull the prompt from the hub
prompt = hub.pull("hwchase17/react-chat")

# Partially apply the prompt with the tools description and tool names
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Bind the LLM with a stop condition
llm_with_stop = llm.bind(stop=["\nObservation"])

# Define the template for tool response
# Define the agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

# Initialize the conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="output"
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    return_intermediate_steps=True,
)
