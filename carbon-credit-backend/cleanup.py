import psycopg2

conn = psycopg2.connect('postgresql://postgres:student@localhost:5433/carbon_credit_db')
cur = conn.cursor()

project_id = 'a3e6ccc8-b775-4ee3-8150-51b6926e9588'

cur.execute("DELETE FROM verifications WHERE project_id = %s", (project_id,))
cur.execute("DELETE FROM fraud_flags WHERE project_id = %s", (project_id,))
cur.execute("DELETE FROM credit_ledger WHERE project_id = %s", (project_id,))
cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))

conn.commit()
cur.close()
conn.close()
print('Deleted cleanly')