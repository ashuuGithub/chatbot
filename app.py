import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os

# Set page configuration at the very start
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

# Load environment variables
load_dotenv()

def init_database() -> SQLDatabase:
    # Get credentials from environment variables
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")

    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Automatically initialize the database when the app starts
if "db" not in st.session_state:
    db = init_database()
    st.session_state.db = db
    st.success("Connected to database!")  # Provide feedback to the user

# Initialize session state for multiple chat sessions
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = [{"chat_history": [], "user_questions": []}]

if "current_session_index" not in st.session_state:
    st.session_state.current_session_index = 0  # Index of the current chat session

# Sidebar with "🆕" for new chat option
with st.sidebar:
    st.subheader("Chat Questions")
    # New chat button with "🆕" symbol
    if st.button("🆕 Start New Chat"):
        # Add a new chat session
        st.session_state.chat_sessions.append({"chat_history": [], "user_questions": []})
        st.session_state.current_session_index = len(st.session_state.chat_sessions) - 1

    # Display the list of previous chat sessions
    for i, session in enumerate(st.session_state.chat_sessions):
        if st.button(f"Chat {i + 1}: {session['user_questions'][0] if session['user_questions'] else 'New Chat'}"):
            st.session_state.current_session_index = i  # Switch to the selected session

# Get the current chat session
current_session = st.session_state.chat_sessions[st.session_state.current_session_index]

# Display the ongoing chat history
for message in current_session["chat_history"]:
    if isinstance(message, AIMessage):
        with st.chat_message("ChatBot"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("User"):
            st.markdown(message.content)

# Handle user input
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    # Add the user query to the questions list
    current_session["user_questions"].append(user_query)

    # Display the user query in the main chat window
    current_session["chat_history"].append(HumanMessage(content=user_query))
    with st.chat_message("User"):
        st.markdown(user_query)

    # Generate and display chatbot response
    with st.chat_message("ChatBot"):
        response = get_response(user_query, st.session_state.db, current_session["chat_history"])
        st.markdown(response)

    # Append the chatbot's response to the main chat history
    current_session["chat_history"].append(AIMessage(content=response))
