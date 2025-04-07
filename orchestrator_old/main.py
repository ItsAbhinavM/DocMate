import streamlit as st
from agent import agent_executor
import builtins


def main():
    st.set_page_config(page_title="Orchestrator", page_icon=":robot:", layout="centered")

    builtins.global_prompt = ''


    st.title("Orchestrator")

    # Chat messages
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Function to send a message
    def send_message():
        if st.session_state.new_message != "":
            builtins.global_prompt = st.session_state.new_message
            st.session_state.messages.append({"user": "User", "message": st.session_state.new_message})

            try:
                st.session_state.new_message = "Generating output..."
                output = agent_executor.invoke({"input": builtins.global_prompt})["output"]
                st.session_state.new_message = ""
                st.session_state.messages.append({"user": "Agent", "message": output})

            except Exception as e:
                st.session_state.messages.append({"user": "Agent", "message": f"An error has occurred:\n{e}"})
                st.session_state.new_message = ""

    # Display chat messages
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for message in st.session_state['messages']:
            st.write(f"**{message['user']}**: {message['message']}")

    # New message input
    st.text_area("Enter prompt...", key="new_message", on_change=send_message)
    if st.button("Send"):
        send_message()


main()

try:
    del builtins.global_prompt
except Exception as e:
    print(e)
