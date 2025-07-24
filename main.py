import os
import regex as re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spacy
import orjson
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import grapheme

url_pattern = re.compile(r'https?://\S+|www\.\S+')
punct_pattern = re.compile(r'[^\w\s\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]')
attachment_pattern = re.compile(r'\b(\w+)\s+sent an attachment\b')

stop_words = {
    'his', 'while', 'been', 'few', 'haven', "we've", 'is', 'which', "you'll", 'shan',
    "hasn't", "you've", "you're", 'against', 'no', 'if', "they're", 'again', 'these',
    'about', 'hers', "she'll", 'some', 'into', 'with', 'as', "hadn't", "haven't",
    "doesn't", 'who', 'the', 'and', 'can', 'only', 'there', 'to', "he'll", "we're",
    "couldn't", 'it', 'my', 'than', "she's", 'out', 'shouldn', 'don', 'once', 'o',
    'itself', 'during', 'theirs', 're', "he's", "they'd", "won't", 'm', 'more',
    'their', "it'll", "don't", 'very', 'where', "wouldn't", 'himself', 'here', 'him',
    'me', 'from', 'was', 'not', 'for', 'by', 'should', 'own', 'through', 'i', 'those',
    "they've", 'hadn', 'ain', 'mustn', "you'd", 'both', 'over', 'whom', 'below',
    'mightn', "we'd", 'her', "i'll", 'd', 'most', "shouldn't", "she'd", 'at', 'our',
    'further', 'll', 'because', 'above', 'on', 've', 'an', 'how', 'what', 'after',
    'too', 'in', "mightn't", 'didn', 'same', 'yourselves', 'or', "should've", 'your',
    'yourself', 'do', "that'll", "isn't", "needn't", 'down', 'won', 'aren', 'had',
    "i'd", 'then', 'this', 'its', 'did', 'when', 'all', 'you', 'hasn', 'myself', 'up',
    'each', "mustn't", 'a', 'any', "it'd", 'she', 'are', 'but', 'having', 'before',
    'until', 'wasn', 'am', "shan't", 'couldn', 'weren', "i've", 'isn', "they'll",
    "he'd", 'be', 't', 'ours', "it's", "aren't", 'he', 's', 'we', 'they', 'doing',
    "wasn't", 'ma', 'being', 'under', 'why', "didn't", 'needn', 'of', 'between',
    'other', 'were', 'does', 'doesn', 'herself', 'that', "we'll", "weren't", 'just',
    'off', 'such', 'nor', 'will', 'them', "i'm", 'themselves', 'has', 'ourselves',
    'yours', 'wouldn', 'now', 'have', 'so', 'nt'
}

nlp = spacy.blank("en")
font_path = "C:\\Windows\\Fonts\\seguiemj.ttf"

def find_all_folder(folder_path: str):
    all_folders = os.listdir(folder_path)
    folders = [os.path.join(folder_path, f) for f in all_folders if os.path.isdir(os.path.join(folder_path, f))]
    return folders

def find_all_files(folder_path: str):
    all_files = os.listdir(folder_path)
    files = [os.path.join(folder_path, f) for f in all_files if f.endswith('.json')]
    return files

def make_unique_dir(base_dir, desired_name):
    candidate = os.path.join(base_dir, desired_name)
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate_i = os.path.join(base_dir, f"{desired_name}{i}")
        if not os.path.exists(candidate_i):
            return candidate_i
        i += 1

def read_json_files(files: list[str]):
    dfs = []

    for file in files:
        with open(file, 'rb') as f:
            data = orjson.loads(f.read())
        messages = data.get('messages')
        df = pd.json_normalize(messages)

        df = clean_data(df)

        dfs.append(df)

    combined_dataframe = pd.concat(dfs, ignore_index=True)
    combined_dataframe = combined_dataframe.sort_values(by='datetime')
    return combined_dataframe

def clean_data(df: pd.DataFrame):
    cols_to_keep = ['sender_name', 'datetime', 'content', 'reactions', 'share.link', 'share.share_text', "share.original_content_owner", 'photos', 'audio_files']

    df['datetime'] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')

    df = df[[col for col in cols_to_keep if col in df.columns]]

    df = df[~df['content'].str.lower().str.endswith('liked a message', na=False)]
    df = df[~df['content'].str.lower().str.strip().str.endswith('to your message', na=False)]
    df = df[~df['content'].str.contains('This poll is no longer available.', na=False)]

    attachment_mask = df['content'].str.contains('sent an attachment', na=False)
    df.loc[attachment_mask, 'content'] = df.loc[attachment_mask, 'content'].apply(preserve_attachment_phrases)

    emoji_mask = df['content'].apply(looks_double_encoded)
    df.loc[emoji_mask, 'content'] = df.loc[emoji_mask, 'content'].apply(fix_emoji_encoding)

    return df

def clean_text(text: str):
    text = url_pattern.sub('', text)
    text = punct_pattern.sub('', text)
    tokens = [token.text for token in nlp(text)]
    tokens = [word.strip() for word in tokens]

    output = [word for word in tokens if word not in stop_words]
    return ' '.join(output)

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
    s = attachment_pattern.sub(r'\1_sent_an_attachment', s)
    return s

def default(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records") 
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if not pd.isna(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [default(x) for x in obj]
    if pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")

def get_total_messages(df: pd.DataFrame):
    return df['content'].notna().sum()

def get_message_count(df: pd.DataFrame, text: str):
    return df['content'].str.contains(text, case=False, na=False).sum()

def count_messages_sent(df: pd.DataFrame):
    numbers = df.value_counts('sender_name')
    proportions = df.value_counts('sender_name', True).round(3)

    return {
        "message_per_user": numbers,
        "message_proportions": proportions
    }

def get_most_frequent_messages(df: pd.DataFrame):
    return df['content'].value_counts().head(20)

def get_messages_per_index(df: pd.DataFrame, index: str = "D"):
    return df.set_index("datetime").resample(index).size()

def most_active_days(df: pd.DataFrame, num: int = 10) -> list[dict]:
    s = get_messages_per_index(df, "D").sort_values(ascending=False).head(num)
    return [{"date": str(date.date()), "messages": int(count)} for date, count in s.items()]

def average_message_length(df: pd.DataFrame):
    df['content_length'] = df['content'].apply(lambda x: grapheme.length(x) if isinstance(x, str) else 0)
    avg_lengths = df.groupby('sender_name')['content_length'].mean().astype(int)
    
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
        "start": max_streak_start.date() if max_streak_start else None,
        "end": max_streak_end.date() if max_streak_end else None
    },
    "max_gap": {
        "length": max_gap,
        "start": max_gap_start.date() if max_gap_start else None,
        "end": max_gap_end.date() if max_gap_end else None
    }
    }


def day_time_graph(df: pd.DataFrame, output_dir: str):
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    
    pivot_table = df.pivot_table(
        index='day_of_week',
        columns='hour',
        values='datetime',
        aggfunc='count',
    )
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(index=days_order, fill_value=0)
    pivot_table = pivot_table.reindex(columns=np.arange(24), fill_value=0)
    pivot_table = pivot_table.fillna(0).astype(int)

    data = pivot_table.to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    cax = ax.imshow(data, aspect='auto', cmap='Blues')

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

    output_path = os.path.join(output_dir, "day_time_graph.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def build_freq_graph(df: pd.DataFrame, output_dir: str, index: str = "D"):
    frequency = get_messages_per_index(df, index)
    plt.figure(figsize=(12,6))
    frequency.plot()
    plt.title("Messages Sent Per Day")
    plt.xlabel("Date")
    plt.ylabel("Message Count")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "messages_per_day.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def generate_wordcloud(df: pd.DataFrame, output_dir: str):
    all_words = df['content'].dropna().apply(clean_text)
    all_words = all_words[all_words.str.len() > 0]

    freq = Counter(all_words)
    
    wordcloud = WordCloud(
        width=2000,
        height=1000,
        background_color='white',
        collocations=False,
        font_path=font_path,
        font_step=2,
        max_words=200,
        relative_scaling=0.25,
        min_font_size=40,
        random_state=22
    )
    wordcloud.generate_from_frequencies(freq)

    output_path = os.path.join(output_dir, "wordcloud.png")
    wordcloud.to_file(output_path)

analysis_functions = {
    "total_messages": get_total_messages,
    "messages_sent": count_messages_sent,
    "most_frequent_messages": get_most_frequent_messages,
    "most_active_days": most_active_days,
    "average_message_length": average_message_length,
    "message_streaks": get_message_streaks
}

plot_functions = [
    day_time_graph,
    build_freq_graph,
    generate_wordcloud
]

def process_folder(folder: str):
    files = find_all_files(folder)

    person_name = os.path.basename(folder).rsplit('_', 1)[0]
    person_result_dir = make_unique_dir(result_folder, person_name)
    os.makedirs(person_result_dir, exist_ok=True)

    print(f"Processing {person_name}...")
    data = read_json_files(files)

    results = {}
    for key, func in analysis_functions.items():
        try:
            results[key] = func(data)
        except Exception as e:
            print(f"Error in {key} for {person_name}: {e}")
            results[key] = None
    
    json_path = os.path.join(person_result_dir, "analysis_results.json")
    with open(json_path, 'wb') as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2, default=default))
    
    for plot_func in plot_functions:
        try:
            plot_func(data, person_result_dir)
        except Exception as e:
            print(f"Error in plotting function {plot_func.__name__} for {person_name}: {e}")


if __name__ == "__main__":
    word = "💔👎👍Hello🤓🤓👩‍👩‍👧"
    #print(len(word))
    #print(grapheme.length(word))
    start = time.time()

    # Comment off the option you don't use with #
    # Option 1: Process all inbox folders
    main_folder = r"YOUR-INSTAGRAM-DOWNLOAD-FOLDER-HERE"  # Path to the main inbox folder, it should end in "/messages/inbox"
    result_folder = r"YOUR-RESULT-FOLDER-HERE"  # Path of folder to store the results 
    folders = find_all_folder(main_folder)

    print("Processing all inbox folders...")
    with ThreadPoolExecutor() as executor:
        executor.map(process_folder, folders)

    
    # Option 2: Process a single folder
    single_folder = r"YOUR-SINGLE-FOLDER-HERE"  # Path to a specific folder (e.g., one DM or group chat)
    result_folder = r"YOUR-RESULT-FOLDER-HERE"  # Path to store the results
    print(f"Processing single folder: {os.path.basename(single_folder)}...")
    process_folder(single_folder)

    end = time.time()
    print(f"Execution Time: {end - start:.2f} seconds")
