import psycopg2
import requests
import datetime

conn = psycopg2.connect(
    host="13.229.212.217",
    database="team_hero",
    user="rails",
    password="sennalabs",
    port="5432")
    
cur = conn.cursor()

def checkin_teamhero(user_name=None, file_name=None, emotion = None):
    detail = f"Camera - {user_name}"
    userid_select_query = f"select * from users where nick_name ilike '{user_name}'"
    cur.execute(userid_select_query)
    user = cur.fetchone()
    if user == None:
        return 0 

    user = transform_user(user)
    print(user_name)
    
    checkin_select_query = f"select * from check_ins where date = current_date and user_id = {user['id']}"
    cur.execute(checkin_select_query)
    user_checkin = cur.fetchall()

    if datetime.datetime.now().time() > user['check_in_time']:
        on_time = "False"
    else:
        on_time = "True"

    if len(user_checkin) == 0:
        checkin_insert_query = f"""insert into check_ins(date, created_at, updated_at, user_id, time, details, on_time, user_check_in_time,workplace_id,user_time_zone,company_id) 
                                   values(current_date, now(), now(), {user['id']},'{(datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")}', '{detail}', '{on_time}', '{user["check_in_time"]}', 2, 'Bangkok', '{user["company_id"]}')"""
        cur.execute(checkin_insert_query)
        conn.commit()
        my_file = {
            'file' : (f'{file_name}', open(f'{file_name}', 'rb'), 'JPEG')
        }

        payload={
            "filename":f"{file_name}", 
            "token":"xoxb-128417594390-1305178157574-RHLUyiLCPembD4BJCFI666cW", 
            "channels":"CQGGC3X6Z", 
            "initial_comment":f"{user_name.upper()} has checked in with {emotion} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'ON TIME' if on_time == 'True' else 'LATE'}" 
        }
        print(requests.post("https://slack.com/api/files.upload", params=payload, files=my_file).text)
        return 1
    else:
        return 0


def transform_user(user):
    data = {}
    col_name =['id','email','encrypted_password','reset_password_token','reset_password_sent_at', 'remember_created_at',
                                 'created_at','updated_at','avatar','check_in_time','first_name','last_name','personal_email',
                                 'birthday','joined_date','mobile_number','position_id','provider','uid','remember_token',
                                 'time_zone','max_sick_leave','max_personal_leave','max_vacation_leave','slack_member',
                                 'company_id','nick_name','status','github_username','user_group_id','on_time_count',
                                 'star_point','primary_id','secondary_id','tertiary_id','max_wfh']

    for index, item in enumerate(col_name):
        data[item] = user[index]

    return data



# print(checkin_teamhero("Kant"))
