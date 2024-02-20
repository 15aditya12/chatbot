import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import streamlit as st

openai_api_key = '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'
# Streamlit app layout
st.title('Chat with Data')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Read the file based on its type
    if uploaded_file.type == "text/csv":
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        data = pd.read_excel(uploaded_file)
    # Display the uploaded data if requested
    if st.checkbox('Show uploaded data'):
        st.write(data)

    # Creating Langchain Pandas agent
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo-0613"),
        data,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    # Text input for the query
    query = st.text_input("Enter your query")

    if query:
        # Run the agent with the user's query
        response = agent.run(query)

        # Generate a conversational response
        st.markdown(f"**User:** {query}")
        st.markdown("**Assistant:**")
        st.write(f"{response}")
