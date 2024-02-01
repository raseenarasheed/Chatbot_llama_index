#Build a chatbot with custom data sources, powered by LlamaIndex
1. This app uses LlamaIndex to load and index data
2. Create a chat UI with Streamlit's st.chat_input and st.chat_message methods
3. Store and update the chatbot's message history using the session state
4. Augment GPT-3.5 with the loaded, indexed data through LlamaIndex's chat engine interface so that the model provides relevant responses based on the document