import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sqlite3
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("models/gemini-3.5-flash")

# Load columns from dataset
df = pd.read_csv("data/application_train.csv")
columns = ", ".join(df.columns.tolist())

# Connect Database
conn = sqlite3.connect("credit_risk.db")

# User Question
question = input("Ask a question: ")

prompt = f"""
You are a SQL expert.

Table Name: applications

Available Columns:
{columns}

Important:
- TARGET = 1 means default
- TARGET = 0 means non-default
- AMT_INCOME_TOTAL = applicant income
- AMT_CREDIT = credit amount
- AMT_ANNUITY = loan annuity
- CODE_GENDER = gender
- OCCUPATION_TYPE = occupation

Generate ONLY valid SQLite SQL.
Do not use columns that are not listed above.

Question:
{question}
"""

response = model.generate_content(prompt)

sql_query = response.text.strip()
sql_query = sql_query.replace("```sql", "").replace("```", "")

print("\nGenerated SQL:")
print(sql_query)

try:
    result = pd.read_sql(sql_query, conn)

    print("\nResult:")
    print(result)

except Exception as e:
    print("\nSQL Error:")
    print(e)

conn.close()