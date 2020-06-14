import time
import json
import pandas as pd
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi

# Read data and get a random permutation of video ids.
data = pd.read_csv('data/USvideos.csv').drop_duplicates('video_id', 'first')
videos_ids = np.random.permutation(data['video_id'])

# Distribute video ids into ten batches.
video_id_batches = []
n_batches = 10
n_ids = data.shape[0] // n_batches + 1
for i in range(n_batches):
    idx = i * n_ids
    video_ids = list(videos_ids[idx:idx+n_ids])
    video_id_batches.append(video_ids)

# Get captions for each batch of video ids.
for i in range(1, len(video_id_batches)):
    video_ids, transcripts, unavailable = [], [], []
    # Try to get transcripts for each video in this batch.
    for video_id in video_id_batches[i]:
        try:
            # Download and save the original captions for this video.
            captions = YouTubeTranscriptApi.get_transcript(video_id)
            with open(f'{video_id}.json', mode='w') as f:
                f.write(json.dumps(captions))
            # Concatenate the captions into a complete transcript.
            transcript = ' '.join(caption['text'] for caption in captions).replace('\n', ' ')
            # Append the video id and transcript.
            video_ids.append(video_id)
            transcripts.append(transcript)
        except:
            # Append the video id among those unavailable.
            unavailable.append(video_id)
        # Wait a few seconds before making the next request.
        time.sleep(4)
    # Write the available transcripts from this batch to file.
    with open('data/transcripts.txt', mode='a') as f:
        for data in zip(video_ids, transcripts):
            f.write('%s\t%s\n' % data)
    # Write the unavailable video ids in this batch to faile.
    with open('data/unavailable.txt', mode='a') as f:
        for video_id in unavailable:
            f.write('%s\n' % video_id)

