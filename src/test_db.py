import sqlite3
import pandas as pd

conn = sqlite3.connect("credit_risk.db")

query = """
SELECT TARGET, COUNT(*) as count
FROM applications
GROUP BY TARGET
"""

result = pd.read_sql(query, conn)

print(result)

conn.close()