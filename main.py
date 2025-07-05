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
    return combined_dataframe

def count_messages_sent(df):
    numbers = df.value_counts('sender_name')
    proportions = df.value_counts('sender_name', True)
    print(numbers)
    print(proportions)

def count_most_frequent(df):
    return df['content'].value_counts().head(10)

def get_messages_per_day(df):
    return df.set_index("datetime").resample("D").size()

def most_active_days(df, num):
    return get_messages_per_day(df).sort_values(ascending=False).head(num)

def build_freq_graph(df):
    daily = get_messages_per_day(df)
    plt.figure(figsize=(12,6))
    daily.plot()
    plt.title("Messages Sent Per Day")
    plt.xlabel("Date")
    plt.ylabel("Message Count")
    plt.tight_layout()
    plt.savefig("messages_over_time.png")

def get_message_streaks(df):
    df = df.sort_values("datetime")
    dates = df["datetime"].dt.normalize().drop_duplicates()
    data_range = pd.date_range(start=dates.min(), end = dates.max(), freq = "D")
    sent_range = data_range.isin(dates)
    
    max_streak = current_streak = 0
    streak_start = max_streak_start = max_streak_end = None

    max_gap = current_gap = 0
    gap_start = max_gap_start = max_gap_end = None

    for i, sent in enumerate(sent_range):
        date = data_range[i]
        if sent:
            if current_streak == 0:
                streak_start = date
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                max_streak_start = streak_start
                max_streak_end = date

            current_gap = 0
        else:
            if current_gap == 0:
                gap_start = date
            current_gap += 1
            if current_gap > max_gap:
                max_gap = current_gap
                max_gap_start = gap_start
                max_gap_end = date

            current_streak = 0

    return {
    "max_message_streak": {
        "length": max_streak,
        "start": max_streak_start,
        "end": max_streak_end
    },
    "max_gap": {
        "length": max_gap,
        "start": max_gap_start,
        "end": max_gap_end
    }
    }   

files = find_all_files(r"YOUR FOLDER HERE")
data = read_json_files(files)
print(most_active_days(data, 5))