from Functions.code_gen_execution.arbitrary_code import arbitrary_code
from Functions.serial_io.serial_in import serial_in
from Functions.serial_io.serial_out import serial_out
from Functions.serial_io.lights import lights_on, lights_off
from Functions.web_interface.search import custom_search_tool
from Functions.discord_module.discord_message import discord_messaging
from dependencies import *
from models.llm import gemini_pro
import builtins

llm = gemini_pro

llm.temperature = 0.3


# Define the tools to be used by the agent
tools = [custom_search_tool, discord_messaging, arbitrary_code, serial_in, serial_out, lights_on, lights_off]


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
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='output')

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, return_intermediate_steps=True)
