import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import time
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Recruit Ease")
st.title("Recruit Ease")

DATA_PATH = "./resumes1.csv"

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(DATA_PATH, dtype={'ID': 'Int64'})

class ChatBot:
    def __init__(self, api_key: str, model: str):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.2
        )

    def get_ai_analysis(self, resume_text):
        system_message = SystemMessage(content=f"""
                                       
            You are an expert in talent acquisition. Analyze the following resume based on these requirements:

            1. Phone Number:
                - The candidate's phone number should be a UK number.
            
            2. Location:
                - The candidate must be a resident of England or have a degree from England.
            
            3. Qualification:
                - The candidate must have passed the GCSE exam in one of the following countries: UK, IRE, AUS, NZ, CAN, SA.
                - The candidate must have a tertiary qualification.
            
            4. Experience:
                - The candidate should have classroom experience or formal teacher training in one of the following countries within the last 2 years: UK, IRE, AUS, NZ, CAN, SA.
                - The candidate should be a Primary Teacher, Secondary Teacher, Teaching Assistant, SEN Teacher, SEN Teaching Assistant, LSA, or HLTA.

            Resume:
            {resume_text}

            Provide the analysis in the following format:

            Resume Name: [Candidate Name]
            Overall: pass if and only if all requirements are met, fail otherwise.
            Fail (if any): [One-line reason for each failing requirement, separated by commas. Include the phone number in the reason if the phone number requirement fails. If all requirements are met, if fail donot show this ."]
        """)

        try:
            response = self.llm.invoke([system_message, HumanMessage(content="Analyze the resume based on the provided requirements.")])
            total_tokens = response.response_metadata['token_usage']['total_tokens']
            response = response.content.strip()
            return response , total_tokens
        except Exception as e:
            logging.error(f"Error during API call: {e}")
            return "Error in analysis."

    def analyze_resumes(self, df):

        results = []
        total_tokens_consumed = 0
        for index, row in df.iterrows():
            if index > 20:
                break
            resume_text = row["Resume"]
            analysis_result = {
                "ID": row["ID"],
                "details": []
            }

            ai_analysis, tokens_used = self.get_ai_analysis(resume_text)
            total_tokens_consumed += tokens_used
            lines = ai_analysis.split('\n')
            details = [line for line in lines]

            analysis_result["details"] = details
            results.append(analysis_result)

        return results, total_tokens_consumed

chatbot = ChatBot(api_key=API_KEY, model="gpt-4o-mini")

if st.button("Analyze Resumes"):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"Analysis started at: {current_time}")
    
    with st.spinner('Analyzing resumes...'):
        start_time = time.time()

        analysis_results , tokens = chatbot.analyze_resumes(st.session_state.df)
        st.session_state.analysis_results = analysis_results

        end_time = time.time()
        execution_time = end_time - start_time

    st.write(f"Total tokens consumed: {tokens}")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"Analysis completed at: {current_time}")
    st.write(f"Execution time: {execution_time} seconds")

if "analysis_results" in st.session_state:
    st.header("Analysis Results")
    for result in st.session_state.analysis_results:
        st.subheader(f"-----Resume ID: {result['ID']}-----")
        for detail in result["details"]:
            st.write(detail)
        st.write("\n")