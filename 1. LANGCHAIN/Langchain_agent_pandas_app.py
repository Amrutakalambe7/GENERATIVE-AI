# streamlit_app.py

import os
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Set OpenAI API Key securely
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

st.title("LangChain Agent with Pandas")
st.write("Interact with your CSV data using natural language queries!")

uploaded_files = st.file_uploader("Upload one or more CSV files", accept_multiple_files=True, type=["csv"])

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.warning("Please enter your OpenAI API key to continue.")

dfs = []
file_names = []

# Load dataframes
if uploaded_files and openai_api_key:
    for file in uploaded_files:
        df = pd.read_csv(file)
        dfs.append(df)
        file_names.append(file.name)

    st.success(f"{len(dfs)} file(s) uploaded successfully.")

    # Display the uploaded DataFrames
    for i, df in enumerate(dfs):
        st.subheader(f"📄 Preview: {file_names[i]}")
        st.dataframe(df.head())

    # Initialize the LLM and Agent
    llm = OpenAI(temperature=0)
    agent = create_pandas_dataframe_agent(llm, dfs if len(dfs) > 1 else dfs[0], verbose=True, allow_dangerous_code=True)

    st.markdown("---")
    query = st.text_input("Ask a question about the data:")

    if st.button("Submit") and query:
        with st.spinner("Thinking..."):
            try:
                result = agent.run(query)
                st.success("✅ Answer:")
                st.markdown(result)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("Upload CSV files to get started.")
