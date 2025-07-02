import pandas as pd
import os
import orjson
import matplotlib.pyplot as plt

def find_all_files(folder_path):
    all_files = os.listdir(folder_path)
    files = [os.path.join(folder_path, f) for f in all_files if f.endswith('.json')]
    return files

def read_json_files(files):
    dfs = []
    cols_to_keep = ['sender_name', 'datetime', 'content', 'reactions', 'share.link', 'share.share_text', "share.original_content_owner", 'photos', 'audio_files']

    for file in files:
        with open(file, 'rb') as f:
            data = orjson.loads(f.read())
        messages = data.get('messages')
        df = pd.json_normalize(messages)

        df['datetime'] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df = df[[col for col in cols_to_keep if col in df.columns]]
        df = df[df['content'] != 'Liked a message']

        dfs.append(df)

    combined_dataframe = pd.concat(dfs, ignore_index=True)
    combined_dataframe = combined_dataframe.sort_values(by='datetime')
    print(combined_dataframe)
    return combined_dataframe

def count_messages_sent(df):
    return df.value_counts('sender_name')

def count_most_frequent(df):
    return df['content'].value_counts().head(20)

def build_freq_graph(df):
    df = df.set_index("datetime")
    messages_per_day = df.resample("D").size()
    messages_per_day.to_csv("output.csv")
    plt.figure(figsize=(12,6))
    messages_per_day.plot()
    plt.title("Messages Over Time (Daily)")
    plt.xlabel("Date")
    plt.ylabel("Number of Messages")
    plt.tight_layout()
    plt.savefig("messages_over_time.png")

files = find_all_files(r"")
data = read_json_files(files)
result = count_messages_sent(data)

build_freq_graph(data)
print(count_most_frequent(data))
print(result)