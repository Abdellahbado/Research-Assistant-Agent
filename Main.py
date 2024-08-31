from create_agent import create_research_assistant
import streamlit as st

def main():
    st.title("Research Assistant AI")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf', 'txt'])

    if "agent" not in st.session_state or uploaded_files:
        st.session_state.agent = create_research_assistant(uploaded_files)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a research question:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                response = st.session_state.agent.run(input=user_input)
            st.write(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

