import streamlit as st
from langchain_openai import ChatOpenAI
from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.docarray import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from dotenv import dotenv_values
from langchain.vectorstores.chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
import pdfplumber
from streamlit_js_eval import streamlit_js_eval

config = dotenv_values(".env")
api_key = config["OPENAI_API_KEY"]

class App:
    def show_sidebar(self):
        st.sidebar.title("PDF Upload")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
        reset_button = st.sidebar.button("Reset Chat")
        if uploaded_file is not None:
            with st.sidebar.expander("Uploaded PDF", expanded=False):
                try:
                    full_text = ""
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            st.write(page_text)
                            full_text += page_text + "\n"
                    # Storing the full text in the session state
                    st.session_state["memory"] = full_text
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
        return reset_button

    def display_chat_history(self, current_user_input=None, current_bot_response=None):
        """Displays the chat history along with the current user input and bot response.""" 
    
        if "messages" in st.session_state:
            for message in st.session_state['messages']:
            # Check the type of the message and extract the content
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(message.content)
                else:
                    st.markdown("Unknown message type")
        else:
            st.info("No chat history yet. Start a conversation!")
        
        if current_user_input:
            st.session_state["messages"].append(HumanMessage(content=current_user_input))
            with st.chat_message("user"):
                st.markdown(current_user_input) 

        # Append the current bot response to the chat history
        if current_bot_response:
            st.session_state["messages"].append(AIMessage(content=current_bot_response.content))
            with st.chat_message("assistant"):
                st.markdown(current_bot_response.content)
    
    def __init__(self) -> None:
        st.session_state["memory"] = (
            list() if "memory" not in st.session_state else st.session_state["memory"]
        )
        if 'reset' not in st.session_state:
            st.session_state.setdefault("reset", False)
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        self.embeddings = OpenAIEmbeddings(openai_api_key= api_key)
        self.model = ChatOpenAI(model="gpt-3.5-turbo-16k", openai_api_key= api_key)

        self.contextualize_q_system_prompt = """Act as an curious conversationalist chatbot. Given a chat history, reformulate the latest user question \
            which might reference context in the chat history, formulate a standalone question with necessary background information \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        self.qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        The context is given {context}. \
        Example:
        User: Can you tell me about the steps for developing an LLM project?
        Answer: I am not familiar with the concept of LLM. Can you tell me little bit about LLM?\

        User: LLM means large language model.
        Answer: Got it. The steps for building an LLM project are Preparation, Building, Deployment, Evaluation.\
        (End of examples)
        After giving a response to the user, you must ask the user a confirmatory question about the response that you provided.
        Remember to always ask question with every response to the user to make sure your answers satisfy their question."""
        
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.output_parser = StrOutputParser()

    def __call__(self):
        st.title("Answering Bot")

        reset_button = self.show_sidebar()

        if reset_button:
            st.session_state.clear()
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
        
        question_txt = st.text_input("Ask anything from memory")
        
        if question_txt:
            vectorstore = Chroma.from_texts(
                st.session_state["memory"],
                embedding=self.embeddings,
            )
            retriever = vectorstore.as_retriever()
            
            def contextualized_question(input: dict):
                if input.get("chat_history"):
                    return contextualize_q_chain
                else:
                    return input["question"]
                
            contextualize_q_chain = self.contextualize_q_prompt | self.model | self.output_parser
            chain = (
                    RunnablePassthrough.assign(
                        context= contextualized_question | retriever
                    )
                    | self.qa_prompt
                    | self.model
                )
            chain_input = {"question": question_txt, "chat_history": st.session_state["messages"]}
            res = chain.invoke(chain_input)
            # logger.debug(retriever.get_relevant_documents(question_txt))
            self.display_chat_history(current_user_input=question_txt, current_bot_response=res)

            # print(st.session_state["messages"])

        
if __name__ == "__main__":
    app = App()
    app()