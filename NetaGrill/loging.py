import psycopg2

import boto3
from botocore.exceptions import NoCredentialsError

def loging(img_path, predicted_emotion):
    
    
    ACCESS_KEY = 'AKIA6AZRH775CZCGWME7'
    SECRET_KEY = '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'

    def update_db():
        conn = psycopg2.connect(
        host="13.229.212.217",
        database="netagrill",
        user="rails",
        password="sennalabs",
        port="5432")
        
        cur = conn.cursor()
        insert_query = f"""insert into customer_record(date, emotion, filename) 
                                   values(now(), '{predicted_emotion}', '{img_path.replace("cap_img/","")}')"""


        cur.execute(insert_query)
        print("Update db: ",conn.commit())


    def upload_to_aws(local_file, bucket, s3_file):
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY)

        try:
            s3.upload_file(local_file, bucket, s3_file)
            print("Upload Successful")
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False

    print(f'save to: img/{predicted_emotion}/{img_path.replace("cap_img/","")}')
    update = update_db()
    uploaded = upload_to_aws(img_path, 'netagrill-image', f'img/{predicted_emotion}/{img_path.replace("cap_img/","")}')
    

    return 0