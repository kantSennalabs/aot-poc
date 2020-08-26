import requests
import json
import os
from datetime import datetime
# the function to send notification to slack
# def sendtoslack(person_name):
#     # channel's webhook URL
#     wekbook_url = 'https://hooks.slack.com/services/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'
#     # print('name in slack', predictions)
#     # data to send
#     now = datetime.now()
#     current_time = now.strftime("%H:%M:%S")
#     data = {
#         'text': f"{person_name} Has checked in at {current_time}",
#         'username': 'test',
#         'icon_emoji': ':robot_face:'
#     }
#     file_upload = {
#     'file':('train/pot/pot.jpg', open('train/pot/pot.jpg', 'rb'), 'jpg')
#     }
#     #get response
#     response = requests.post(wekbook_url, data=json.dumps(
#         data), headers={'Content-Type': 'application/json'}, files=file_upload)
#     # print('Response: ' + str(response.text))
#     # print('Response code: ' + str(response.status_code))

def post_file_to_slack(text, file_name, filepath, file_type=None, title=None):
    return requests.post(
      'https://slack.com/api/files.upload', 
      {
        'token': "xoxb-128417594390-1187449933264-7tdwf01MNpS0ISbSYZRH0AF3",
        'filename': file_name,
        'channels': "C018R49QYHM",
        'filetype': file_type,
        'initial_comment': text,
        'title': title
      },
      files = {
        'file': (filepath, open(filepath, 'rb'), 'image/jpg', {
            'Expires': '0'
            })
        })

text = f"Pot Has checked in at {datetime.now().strftime('%H:%M:%S')}"
print(post_file_to_slack(text, f"Pot.jpg", 'train/pot/pot.jpg'))
