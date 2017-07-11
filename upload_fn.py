import json
import requests

from collections import OrderedDict


def send_data(TOKEN, data, index):
    data = json.dumps(data)
    headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}

    # Have not checked to see if try/except actually catches problems with batches
    try:
        response = requests.put(
            "https://reinfer.io/api/voc/datasets/tanrajbir/ubuntu_support_chat/comments",
            headers=headers, data=data)
        print response.content
        print index, 'working' # index starts from 0
    except:
        print index, "problem"


def create_messages(comment, speaker_list, time_stamp_list, dialog_list):
    comment['messages'] = []

    # Creates a thread of messages: one message for each turn of the convo
    for turn_number in xrange(len(speaker_list)):
        message = {}
        message['from'] = speaker_list[turn_number]
        message['sent_at'] = time_stamp_list[turn_number]
        message['original_text'] = dialog_list[turn_number]

        # Message directed to next speaker in speaker_list
        if turn_number == len(speaker_list) - 1: # i.e if last speaker
            message['to'] = 'empty'
        else:
            message['to'] = speaker_list[turn_number + 1]

        comment['messages'].append(message)

def create_users(comment, speaker_list):
    comment['user_properties'] = {}
    speakers_no_dup = list(OrderedDict.fromkeys(speaker_list))

    for index, speaker in enumerate(speakers_no_dup):
        key = 'string:speaker_' + str(index)
        comment['user_properties'][key] = speaker

def create_comment(raw_line):
    line = json.loads(raw_line)

    # Load variables from line
    speaker_list = line.get('speaker_list', 'empty')
    time_stamp_list = line.get('time_stamp_list', 'empty')
    id = line.get('dialog_id', 'empty')
    dialog_list = line.get('dialog_list', 'empty')

    # A comment is a complete back and forth conversation.
    comment = {}
    comment['timestamp'] = time_stamp_list[0]
    comment['id'] = str(id) + "18"

    # Adds a chain of messages to a comment
    create_messages(comment, speaker_list, time_stamp_list, dialog_list)

    # Adds user_properties to a comment
    create_users(comment, speaker_list)

    return comment
