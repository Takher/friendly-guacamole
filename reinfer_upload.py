import os

from upload_fn import send_data, create_comment

# Store the environment variable $TOKEN
TOKEN = os.environ['TOKEN']

# Manually set constants
OWNER = 'tanrajbir'
DATASET = 'testing_ubuntu_support_chat'
FILE_PATH = "./data/movies/data_movie_2000.json"
TIMESTAMP_GIVEN = False

data_url = "https://reinfer.io/api/voc/datasets/%s/%s/comments"%(OWNER, DATASET)

comments = []  # Contains a list of every 'comment'
batches = []  # comments in batches of 1024
unique_id = 0 # To set a unique id. id's in Ubuntu Corpus contain repeats

with open(FILE_PATH, 'r') as file:
    for line in file:
        comment = create_comment(line, unique_id, TIMESTAMP_GIVEN)
        comments.append(comment)
        unique_id += 1

        # There is a limit to the amount of data that can be sent at any time,
        # so structure in batches.
        if len(comments) == 250:
            batches.append(comments)
            comments = []

if comments: batches.append(comments) # To ensure we don't miss any data

problem_indices = []
for index, batch in enumerate(batches):
    # send_data returns the indices of the batches which returned errors
    problem_index = send_data(TOKEN, batch, index, data_url)
    if problem_index is not None: problem_indices.append(problem_index)

if problem_indices:
    print "Issues with the following batch numbers:", problem_indices
