import pandas as pd
import sqlite3

print("Loading dataset...")

df = pd.read_csv("data/application_train.csv")

conn = sqlite3.connect("credit_risk.db")

df.to_sql(
    "applications",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Database created successfully!")