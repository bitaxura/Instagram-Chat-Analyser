import os
import re
import swifter

import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import orjson
import pandas as pd
from wordcloud import WordCloud
import grapheme

nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def find_all_files(folder_path: str):
    all_files = os.listdir(folder_path)
    files = [os.path.join(folder_path, f) for f in all_files if f.endswith('.json')]
    return files

def read_json_files(files: list[str]):
    dfs = []
    cols_to_keep = ['sender_name', 'datetime', 'content', 'reactions', 'share.link', 'share.share_text', "share.original_content_owner", 'photos', 'audio_files']
    pattern = r'^[\w\.\-_]+ reacted .+ to your message$'

    for file in files:
        with open(file, 'rb') as f:
            data = orjson.loads(f.read())
        messages = data.get('messages')
        df = pd.json_normalize(messages)

        df['datetime'] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')

        df = df[[col for col in cols_to_keep if col in df.columns]]

        df = df[~df['content'].str.lower().str.endswith('liked a message', na=False)]
        df = df[~df['content'].str.lower().str.strip().str.endswith('to your message', na=False)]
        df = df[~df['content'].str.contains('This poll is no longer available.', na=False)]

        mask = df['content'].swifter.apply(looks_double_encoded)
        df.loc[mask, 'content'] = df.loc[mask, 'content'].swifter.apply(fix_emoji_encoding)

        dfs.append(df)

    combined_dataframe = pd.concat(dfs, ignore_index=True)
    combined_dataframe = combined_dataframe.sort_values(by='datetime')
    return combined_dataframe    

def clean_text(text: str):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    tokens = [word.strip() for word in tokens]
    
    output = [word for word in tokens if word not in stop_words]
    return output

def fix_emoji_encoding(s: str):
    try:
        return s.encode('latin1').decode('utf-8')
    except Exception:
        return s

def looks_double_encoded(s):
    if not isinstance(s, str):
        return False
    return any(128 <= ord(c) <= 255 for c in s)

def preserve_attachment_phrases(s: str):
    s = s.lower()

    s = re.sub(r'\b(\w+)\s+sent an attachment\b', r'\1_sent_an_attachment', s)
    return s


def count_messages_sent(df: pd.DataFrame):
    numbers = df.value_counts('sender_name')
    proportions = df.value_counts('sender_name', True)

    return {
        "message_per_user": numbers,
        "message_proportions": proportions
    }

def count_most_frequent(df: pd.DataFrame):
    return df['content'].value_counts().head(10)

def get_messages_per_index(df: pd.DataFrame, index: str):
    return df.set_index("datetime").resample(index).size()

def most_active_days(df: pd.DataFrame, num: int):
    return get_messages_per_index(df, "D").sort_values(ascending=False).head(num)

def average_message_length(df: pd.DataFrame):
    df['content_length'] = df['content'].swifter.apply(lambda x: grapheme.length(x) if isinstance(x, str) else 0)
    avg_lengths = df.groupby('sender_name')['content_length'].mean()
    
    return avg_lengths

def get_message_streaks(df: pd.DataFrame):
    dates = df["datetime"].dt.normalize().drop_duplicates()
    data_range = pd.date_range(dates.min(), dates.max(), freq = "D")
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

def day_time_graph(df: pd.DataFrame):
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    
    pivot_table = df.pivot_table(
        index='day_of_week',
        columns='hour',
        values='datetime',
        aggfunc='count',
    )
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(days_order)
    pivot_table = pivot_table.fillna(0).astype(int)

    data = pivot_table.to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    cax = ax.imshow(data, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(np.arange(24))
    ax.set_yticks(np.arange(len(days_order)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(days_order)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = fig.colorbar(cax)
    cbar.set_label('Count')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = int(data[i, j])
            ax.text(j, i, str(value), ha='center', va='center', color='black', fontsize=8)

    ax.set_title('Activity by Day and Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')

    plt.show()


def build_freq_graph(df: pd.DataFrame, index: str):
    frequency = get_messages_per_index(df, index)
    plt.figure(figsize=(12,6))
    pl = frequency.plot()
    plt.title("Messages Sent Per Day")
    plt.xlabel("Date")
    plt.ylabel("Message Count")
    plt.tight_layout()

    plt.show()

def generate_wordcloud(df: pd.DataFrame):
    all_words = df['content'].dropna().swifter.apply(preserve_attachment_phrases).swifter.apply(clean_text).explode()
    all_words = all_words[all_words.str.len() > 0]
    all_words.to_csv("all_words.csv", index=False)
    wordcloud = WordCloud(width=2000, height=1000, background_color='white', collocations=False).generate(' '.join(all_words))
    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig("wordcloud.png", bbox_inches='tight', pad_inches=0.5)

files = find_all_files(r"YOUR_FOLDER_PATH_HERE")
data = read_json_files(files)