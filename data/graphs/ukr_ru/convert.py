import json
import pandas as pd

# Load your json file
with open('data.json', 'r') as f:
    data = json.load(f)

rows = []

# Iterate through the URL keys
for url, content in data.items():
    # Flatten the structure manually to get exactly what you need
    row = {
        "url": url,
        "user_id": content['users'][0]['user_id'] if content['users'] else None,
        "user_name": content['users'][0]['user_name'] if content['users'] else None,
        "description": content['text'].get('desc'),
        "video_id": content['video'].get('id'),
        "createTime": content['video'].get('createTime'),
        "diggCount": content['stats'].get('diggCount'),
        "playCount": content['stats'].get('playCount'),
        "shareCount": content['stats'].get('shareCount')
    }
    rows.append(row)

# Create DataFrame and export
df = pd.DataFrame(rows)
df.to_csv('tiktok_data.csv', index=False)

print("Conversion complete: tiktok_data.csv")
