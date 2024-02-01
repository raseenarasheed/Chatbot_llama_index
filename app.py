import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from pathlib import Path
from llama_index import download_loader
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms import OpenAI

#Reference: https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
# https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot.html

OPENAI_API_KEY = st.secrets.OPENAI_API_KEY
st.header("Chat with a doc 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about the restaurent menu!"}
    ]
#load_data function is wrapped in Streamlit’s caching decorator st.cache_resource to minimize the number of times the data is loaded and indexed.
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        # load json document
        JSONReader = download_loader("JSONReader")

        loader = JSONReader()
        documents = loader.load_data(Path('data.txt'))
        llm = OpenAI(model="gpt-4",openai_api_key=OPENAI_API_KEY)

        # create a service_context instance- specify chunk size & overlap
        service_context = ServiceContext.from_defaults(
            chunk_size=512, chunk_overlap=50,llm=llm
        )

        # creating index from the document
        #creates queryable indices from these chunks into the Knowledge Base
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        return index
    
index = load_data()

# define the above index as chat_engine
# condense question mode is always queries the knowledge base (doc) when generating a response.
chat_engine = index.as_chat_engine(chat_mode="condense_question",similarity_top_k=4)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant": # checking the last message is from assistant or not
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.markdown(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history



