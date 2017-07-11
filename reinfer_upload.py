import os

from upload_fn import send_data, create_comment

# Store the environment variable $TOKEN
TOKEN = os.environ['TOKEN']

comments = []  # Contains a list of every 'comment'
batches = []  # comments in batches of 1024

with open('./data/ubuntu/data_ubuntu_mixed_100000.json', 'r') as file:
    for line in file:
        comment = create_comment(line)

        # Temporary -- I need to remove this!
        comment['original_text'] = "comment you can't see!"

        # Temporary
        comments.append(comment)
        if len(comments) == 1: break

        # Can send a max of 1024 comments so structure data in batches.
        if len(comments) == 1024:
            batches.append(comments)
            comments = []

if comments: batches.append(comments) # To ensure we don't miss any data

for index, batch in enumerate(batches):
    data = {}
    data['comments'] = batch
    send_data(TOKEN, data, index)
