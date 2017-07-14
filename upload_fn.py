import ast
from collections import OrderedDict
import json
import requests
import time
from random import randrange
import datetime
import pytz
import random


def random_date():
    year = random.randint(2010, 2016)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    hour = random.randint(1, 23)
    minute = random.randint(1, 59)
    return datetime.datetime(year, month, day, hour, minute, 0, 0, pytz.UTC)

def random_date_list(list):
    current = random_date()
    dates = []
    for i in xrange(list):
        current = current + datetime.timedelta(minutes=randrange(10))
        dates.append(current.isoformat())
    return dates

def send_data(TOKEN, batch, index, path):
    # Structure data in specified format
    data = {}
    data['comments'] = batch

    delay = 0
    data_json = json.dumps(data)
    headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}

    # Stay in the while loop until data is uploaded or delay becomes too
    # long (10 seconds)
    while (delay < 10):
        time.sleep(delay)

        response = requests.put(path, headers=headers, data=data_json )

        try:
            response_dict = ast.literal_eval(response.content)
            if response_dict['status'] == 'ok':
                print index, response_dict
                return  # This is the 'successful' end of the function

        except:
            # If we are here, it usually means we have received a 5xx error
            pass

        delay += 2

    print "----------------- index %d failed -----------------" % index
    print response.content

    # If errors persist, return index of the failed batch of data
    return index

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
            message['to'] = ['empty']
        else:
            message['to'] = [speaker_list[turn_number + 1]]

        comment['messages'].append(message)

def create_users(comment, speaker_list):
    comment['user_properties'] = {}
    speakers_no_dup = list(OrderedDict.fromkeys(speaker_list))

    for index, speaker in enumerate(speakers_no_dup):
        key = 'string:speaker_' + str(index)
        comment['user_properties'][key] = speaker

def create_comment(raw_line, unique_id, timestamp_given):
    line = json.loads(raw_line)

    # A comment is a complete back and forth conversation.
    comment = {}

    # Load variables from line
    speaker_list = line.get('speaker_list', 'empty')
    dialog_list = line.get('dialog_list', 'empty')

    # if we don't have a time_stamp_list, we generate one
    time_stamp_list = line.get('time_stamp_list', random_date_list(len(dialog_list)))

    comment['timestamp'] = time_stamp_list[0]

    # To avoid conflicts with repeated id's in the Ubuntu Corpus
    comment['id'] = str(unique_id)

    # Adds a chain of messages to a comment
    create_messages(comment, speaker_list, time_stamp_list, dialog_list)

    # Adds user_properties to a comment
    create_users(comment, speaker_list)

    return comment
