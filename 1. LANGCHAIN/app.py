import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os

# Set your OpenAI API Key here or from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = "your openai API KEY"

st.title("DataFrame Q&A Chatbot with LangChain")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load original df
    df = pd.read_csv(uploaded_file)
    st.subheader("Original DataFrame (df)")
    st.dataframe(df)

    # Create df1: Fill missing ages
    df1 = df.copy()
    if 'Age' in df1.columns:
        df1['Age'] = df1['Age'].fillna(df1['Age'].mean())
    st.subheader("df1: Age Filled with Mean")
    st.dataframe(df1)

    # Create df2: Age multiplied
    df2 = df1.copy()
    if 'Age' in df2.columns:
        df2['Age_Multiplied'] = df2['Age'] * 2
    st.subheader("df2: Age_Multiplied Added")
    st.dataframe(df2)

    # LangChain Chat Agent Setup
    llm = ChatOpenAI(temperature=0, model="gpt-4")  # or "gpt-4"
    agent = create_pandas_dataframe_agent(llm, [df, df1, df2], verbose=True, allow_dangerous_code=True)

    # User Query
    st.markdown("### Ask a question about the DataFrames (df, df1, df2)")
    query = st.text_input("For example: Compare the Age column in df and df1.")

    if st.button("Submit") and query:
        with st.spinner("Thinking..."):
            try:
                answer = agent.run(query)
                st.success(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.info("💡 You can reference the DataFrames as `df`, `df1`, and `df2` in your question.")
