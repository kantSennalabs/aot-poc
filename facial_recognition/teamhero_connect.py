import psycopg2

conn = psycopg2.connect(
    host="13.229.212.217",
    database="team_hero_staging",
    user="rails",
    password="sennalabs",
    port="5432")
cur = conn.cursor()
checkin_select_query = "select * from check_ins"
cur.execute(checkin_select_query)
check_ins = cur.fetchall()

checkin_select_query = "select * from users"
cur.execute(checkin_select_query)
users = cur.fetchall()
for row in users:
    print(row[0])

