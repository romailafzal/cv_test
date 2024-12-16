import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
import time
from datetime import datetime
import asyncio

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

    async def get_ai_analysis(self, resume_text):
        system_message = SystemMessage(content=f"""
                                            
            You are an expert in talent acquisition. Analyze the following resume based on these requirements:

            **Location:**
                - The candidate's phone number should be a valid UK mobile number that starts with "07", "+44", or "0044".
                - The candidate must be a resident of England.

            **Qualification:**
                - The candidate must have completed their secondary education by passing the **GCSE exam * (or equivalent) or have GCSE certificate in one of the following countries: UK, IRE, AUS, NZ, CAN, SA.
                - The candidate must also have a **tertiary (higher) qualification**, such as a degree or diploma, obtained after secondary school.

            **Experience:**
                - The candidate must have worked in a classroom or received formal teacher training within the last 2 years in one of the following countries: UK, IRE, AUS, NZ, CAN, SA.
                - The candidate should have one of the following roles: **Primary Teacher**, **Secondary Teacher**, **Teaching Assistant**, **SEN Teacher**, **SEN Teaching Assistant**, **Learning Support Assistant (LSA)**, or **Higher Level Teaching Assistant (HLTA)**.

            **Resume**: {resume_text}

            If the candidate's resume meets all the requirements, the overall result should be "pass." Otherwise, mark it as "fail." In case of a failure, provide clear reasons specifying which requirement(s) the candidate did not meet. If the candidate passes, do not display failure reasons.

            **Format:**

            **Resume Name**: Name of the candidate
            **Overall**: pass if and only if all requirements are met, fail otherwise.
            **Reason**: [One-line reason for each failing or passing requirement, separated by commas.]

            **Examples:**

            **In case of pass:**

            **Resume Name**: Andrew Elon  
            **Overall**: pass  
            **Reason**: The candidate meets all the requirements. The phone number is a valid UK number, the candidate is a resident of England, they passed the GCSE exam in the UK, they hold a tertiary qualification, and have recent classroom experience as a Secondary Teacher in the UK.

            **In case of fail:**

            **Resume Name**: Andrew Elon  
            **Overall**: fail  
            **Reason**: The candidate's phone number is +923364032403, which is not a valid UK number. The candidate is a resident of Scotland, not England. Additionally, the candidate did not pass the GCSE exam in any of the required countries, and there is no mention of recent classroom experience or formal teacher training within the last 2 years.
        """)

        try:
            response = await asyncio.to_thread(self.llm.invoke, [system_message, HumanMessage(content="Analyze the resume based on the provided requirements.")])
            total_tokens = response.response_metadata['token_usage']['total_tokens']
            response = response.content.strip()
            return response, total_tokens
        except Exception as e:
            logging.error(f"Error during API call: {e}")
            return "Error in analysis.", 0

    async def analyze_resumes(self, df):
        results = []
        total_tokens_consumed = 0
        tasks = []
        
        for index, row in df.iterrows():
            if index > 20:
                break
            resume_text = row["Resume"]
            task = asyncio.create_task(self.get_ai_analysis(resume_text))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for index, (ai_analysis, tokens_used) in enumerate(responses):
            total_tokens_consumed += tokens_used
            analysis_result = {
                "ID": df.at[index, "ID"],
                "details": ai_analysis.split('\n')
            }
            results.append(analysis_result)

        return results, total_tokens_consumed

chatbot = ChatBot(api_key=API_KEY, model="gpt-4o-mini")
# chatbot = ChatBot(api_key=API_KEY, model="gpt-4o")


if st.button("Analyze Resumes"):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"Analysis started at: {current_time}")
    
    with st.spinner('Analyzing resumes...'):
        start_time = time.time()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis_results, tokens = loop.run_until_complete(chatbot.analyze_resumes(st.session_state.df))
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
