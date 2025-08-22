import os
import time
from collections import Counter
from typing import Any
import multiprocessing as mp

import joblib
import grapheme
import numpy as np
import orjson
import pandas as pd
import regex as re
import spacy
from wordcloud import WordCloud

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from config import (
    URL_PATTERN, PUNCT_PATTERN, EMOJI_PATTERN,
    ATTACHMENT_PATTERN, STOP_WORDS, SYS_FONT_PATH,
    get_analysis_functions, get_plot_functions,
)


try:
    CLF = joblib.load('emotion_classifier.pkl')
    VECTORIZER = joblib.load('vectorizer.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or vectorizer: {e}. Please run the emotion_classifier.py script first.")
    CLF = None
    VECTORIZER = None
except Exception as e:
    print(f"Unexpected error: {e}")
    CLF = None
    VECTORIZER = None

NLP = spacy.blank("en")
result_folder = r"R:\Result"


class FileOperations:
    @staticmethod
    def find_all_folder(folder_path: str) -> list[str]:
        all_folders = os.listdir(folder_path)
        folders = [os.path.join(folder_path, f) for f in all_folders]
        folders = [f for f in folders if os.path.isdir(f)]
        return folders

    @staticmethod
    def find_all_files(folder_path: str) -> list[str]:
        all_files = os.listdir(folder_path)
        files = [os.path.join(folder_path, f) for f in all_files if f.endswith('.json')]
        return files

    @staticmethod
    def make_unique_dir(base_dir: str, desired_name: str) -> str:
        candidate = os.path.join(base_dir, desired_name)
        if not os.path.exists(candidate):
            return candidate
        i = 1
        while True:
            candidate_i = os.path.join(base_dir, f"{desired_name}{i}")
            if not os.path.exists(candidate_i):
                return candidate_i
            i += 1


class DataProcessor:
    @staticmethod
    def read_json_files(files: list[str]) -> pd.DataFrame:
        dfs = []

        for file in files:
            with open(file, 'rb') as f:
                data = orjson.loads(f.read())
            messages = data.get('messages')
            df = pd.json_normalize(messages)

            df = DataProcessor.clean_data(df)

            dfs.append(df)

        combined_dataframe = pd.concat(dfs, ignore_index=True)
        combined_dataframe = combined_dataframe.sort_values(by='datetime')
        return combined_dataframe

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        cols_to_keep = ['sender_name', 'datetime', 'content', 'reactions',
                         'share.link', 'share.share_text',
                         "share.original_content_owner", 'photos', 'audio_files']

        df['datetime'] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')

        df = df[[col for col in cols_to_keep if col in df.columns]]
        df = df[df['sender_name'] != 'Meta AI']

        df = df[~df['content'].str.lower().str.endswith('liked a message', na=False)]
        df = df[~df['content'].str.lower().str.strip().str.endswith('to your message', na=False)]
        df = df[~df['content'].str.contains('This poll is no longer available.', na=False)]

        attachment_mask = df['content'].str.contains('sent an attachment', na=False)
        df.loc[attachment_mask, 'content'] = df.loc[attachment_mask, 'content'].apply(TextProcessor.preserve_attachment_phrases)

        emoji_mask_content = df['content'].apply(TextProcessor.looks_double_encoded)
        df.loc[emoji_mask_content, 'content'] = df.loc[emoji_mask_content, 'content'].apply(TextProcessor.fix_emoji_encoding)

        emoji_mask_name = df['sender_name'].apply(TextProcessor.looks_double_encoded)
        df.loc[emoji_mask_name, 'sender_name'] = df.loc[emoji_mask_name, 'sender_name'].apply(TextProcessor.fix_emoji_encoding)

        return df


class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = URL_PATTERN.sub('', text)
        text = PUNCT_PATTERN.sub('', text)
        tokens = [token.text for token in NLP(text)]
        tokens = [word.strip() for word in tokens]

        output = [word for word in tokens if word not in STOP_WORDS]
        return ' '.join(output)

    @staticmethod
    def fix_emoji_encoding(s: str) -> str:
        try:
            return s.encode('latin1').decode('utf-8')
        except Exception:
            return s

    @staticmethod
    def extract_emojis(s: str) -> str:
        emojis = EMOJI_PATTERN.findall(s)
        return ''.join(emojis)

    @staticmethod
    def looks_double_encoded(s: str) -> bool:
        if not isinstance(s, str):
            return False
        return any(128 <= ord(c) <= 255 for c in s)

    @staticmethod
    def preserve_attachment_phrases(s: str) -> str:
        s = ATTACHMENT_PATTERN.sub(r'\1_sent_an_attachment', s)
        return s

    @staticmethod
    def filter_by_length(text: str, min_len: int=8, max_len: int=200) -> bool:
        length = len(text.split())
        return min_len <= length <= max_len

    @staticmethod
    def normalize_repeats(text: str) -> str:
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'([!?])\1{1,}', r'\1', text)
        return text


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


class Analyzer:
    @staticmethod
    def get_total_messages(df: pd.DataFrame) -> int:
        return df['content'].notna().sum()

    @staticmethod
    def get_message_count(df: pd.DataFrame, text: str) -> int:
        return df['content'].str.contains(text, case=False, na=False).sum()

    @staticmethod
    def count_messages_sent(df: pd.DataFrame) -> dict[str, int]:
        numbers = df.value_counts('sender_name')
        proportions = df.value_counts('sender_name', True).round(3)

        return {
            "message_per_user": numbers,
            "message_proportions": proportions
        }

    @staticmethod
    def get_most_frequent_messages(df: pd.DataFrame) -> pd.Series:
        return df['content'].value_counts().head(20)

    @staticmethod
    def get_most_frequent_emojis_sent(df: pd.DataFrame) -> dict[str, pd.Series]:
        emoji_counts_per_user = {}
        for user, group in df.groupby('sender_name'):
            all_emojis = group['content'].dropna().apply(TextProcessor.extract_emojis)
            all_emojis = ''.join(all_emojis)
            emoji_counts = Counter(all_emojis)
            emoji_counts_per_user[user] = pd.Series(emoji_counts).sort_values(ascending=False).head(5)

        return emoji_counts_per_user

    @staticmethod
    def get_most_frequent_emojis_reacted(df: pd.DataFrame):
        reacted_dict = {}
        rx = df['reactions'].explode().dropna()
        rx_norm = pd.json_normalize(rx)
        rx_norm['reaction'] = rx_norm['reaction'].apply(TextProcessor.fix_emoji_encoding)
        emoji_mask_actor = rx_norm['actor'].apply(TextProcessor.looks_double_encoded)
        rx_norm.loc[emoji_mask_actor, 'actor'] = rx_norm.loc[emoji_mask_actor, 'actor'].apply(TextProcessor.fix_emoji_encoding)

        rx_norm = rx_norm.groupby('actor')
        for actor, group in rx_norm:
            reacted_dict[actor] = group['reaction'].value_counts().head(5)

        return reacted_dict

    @staticmethod
    def get_messages_per_index(df: pd.DataFrame, index: str = "D") -> pd.Series:
        return df.resample(index, on="datetime").size()

    @staticmethod
    def most_active_days(df: pd.DataFrame, num: int = 10) -> list[dict]:
        s = Analyzer.get_messages_per_index(df, "D").sort_values(ascending=False).head(num)
        return [{"date": str(date.date()), "messages": int(count)} for date, count in s.items()]

    @staticmethod
    def average_message_length(df: pd.DataFrame) -> pd.Series:
        df['content_length'] = df['content'].apply(lambda x: grapheme.length(x) if isinstance(x, str) else 0)
        avg_lengths = df.groupby('sender_name')['content_length'].mean().astype(int)

        return avg_lengths

    @staticmethod
    def get_message_streaks(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        dates = df["datetime"].dt.normalize().drop_duplicates()
        data_range = pd.date_range(dates.min(), dates.max(), freq="D")
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

    @staticmethod
    def days_active(df: pd.DataFrame) -> int:
        dates = df['datetime'].dt.normalize().drop_duplicates()
        return dates.count() if not dates.empty else 0


class Visualizer:
    @staticmethod
    def day_time_graph(df: pd.DataFrame, output_dir: str) -> None:
        temp_df = df[['datetime']].copy()
        temp_df['day_of_week'] = temp_df['datetime'].dt.day_name()
        temp_df['hour'] = temp_df['datetime'].dt.hour

        pivot_table = temp_df.pivot_table(
            index='day_of_week',
            columns='hour',
            values='datetime',
            aggfunc='count',
            fill_value=0
        )
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(index=days_order, columns=np.arange(24), fill_value=0)

        data = pivot_table.to_numpy()

        fig, ax = plt.subplots(figsize=(14, 6))
        cax = ax.imshow(data, aspect='auto', cmap='Blues')

        ax.set_xticks(np.arange(24))
        ax.set_yticks(np.arange(7))
        ax.set_xticklabels(np.arange(24))
        ax.set_yticklabels(days_order)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        cbar = fig.colorbar(cax)
        cbar.set_label('Count')

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = int(data[i, j])
                if value > 0:
                    color = 'white' if value > data.max() * 0.5 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center',
                            color=color, fontsize=8)

        ax.set_title('Activity by Day and Hour')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')

        output_path = os.path.join(output_dir, "day_time_graph.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    @staticmethod
    def build_freq_graph(df: pd.DataFrame, output_dir: str, index: str = "D") -> None:
        frequency = Analyzer.get_messages_per_index(df, index)

        if len(frequency) <= 1:
            return

        plt.figure(figsize=(12,6))
        frequency.plot()
        plt.title("Messages Sent Per Day")
        plt.xlabel("Date")
        plt.ylabel("Message Count")
        try:
            plt.tight_layout()
        except:
            plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)

        output_path = os.path.join(output_dir, "messages_per_day.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    @staticmethod
    def generate_wordcloud(df: pd.DataFrame, output_dir: str) -> None:
        all_words = df['content'].dropna().apply(TextProcessor.clean_text)
        all_words = all_words[all_words.str.len() > 0]

        freq = Counter(all_words)

        wordcloud = WordCloud(
            width=2000,
            height=1000,
            background_color='white',
            collocations=False,
            font_path=SYS_FONT_PATH,
            font_step=2,
            max_words=200,
            relative_scaling=0.25,
            min_font_size=40
        )
        wordcloud.generate_from_frequencies(freq)

        output_path = os.path.join(output_dir, "wordcloud.png")
        wordcloud.to_file(output_path)

    @staticmethod
    def generate_pie_chart(df: pd.DataFrame, output_dir: str) -> None:
        message_count = Analyzer.count_messages_sent(df)['message_proportions'].head(12)
        print(message_count)
        labels = message_count.index.tolist()
        values = message_count.values.tolist()

        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Message % Piechart")
        plt.axis('equal')

        output_path = os.path.join(output_dir, "pie_chart.svg")
        plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.5)
        plt.close()

ANALYSIS_FUNCTIONS = get_analysis_functions(Analyzer)
PLOT_FUNCTIONS = get_plot_functions(Visualizer)


def process_folder(folder: str) -> None:
    files = FileOperations.find_all_files(folder)

    person_name = os.path.basename(folder).rsplit('_', 1)[0]
    data = DataProcessor.read_json_files(files)

    if data['content'].count() < 30 or data['datetime'].dt.normalize().drop_duplicates().count() < 3:
        print(f"Skipping {person_name}: Not enough messages")
        return

    print(f"Processing {person_name}")
    person_result_dir = FileOperations.make_unique_dir(result_folder, person_name)
    os.makedirs(person_result_dir, exist_ok=True)

    results = {}
    for key, func in ANALYSIS_FUNCTIONS.items():
        try:
            results[key] = func(data)
        except Exception as e:
            print(f"Error in {key} for {person_name}: {e}")
            results[key] = None

    json_path = os.path.join(person_result_dir, "analysis_results.json")
    with open(json_path, 'wb') as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2, default=default))
    
    for func in PLOT_FUNCTIONS:
        try:
            func(data, person_result_dir)
        except Exception as e:
            print(f"Error in plotting function {func.__name__} for {person_name}: {e}")


def emotions_game(df: pd.DataFrame) -> None:
    df_copy = df.copy()
    df_copy = df_copy.dropna(subset=['content'])
    df_copy['content'].apply(TextProcessor.clean_text)
    df_copy = df_copy[df_copy['content'].apply(TextProcessor.filter_by_length)]
    df_copy['content'] = df_copy['content'].apply(TextProcessor.normalize_repeats)

    insta_tfidf = VECTORIZER.transform(df_copy['content'])
    df_copy['predicted_emotion'] = CLF.predict(insta_tfidf)
    return df_copy[['sender_name', 'datetime', 'content', 'predicted_emotion']]


if __name__ == "__main__":
    word = "💔👎👍Hello🤓🤓👩‍👩‍👧"
    #print(len(word))
    #print(grapheme.length(word))
    start = time.time()

    # Comment off the option you don't use with #
    # Option 1: Process all inbox folders
    main_folder = r"YOUR-MAIN-FOLDER-HERE"  # Path to the main inbox folder, it should end in "/messages/inbox"
    result_folder = r"YOUR-RESULT-FOLDER-HERE"  # Path of folder to store the results

    folders = FileOperations.find_all_folder(main_folder)
    print("Processing all inbox folders")

    print(f"Processing {len(folders)} folders using {mp.cpu_count()} processes")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_folder, folders)

    # Option 2: Process a single folder
    # single_folder = r"YOUR-SINGLE-FOLDER-HERE"  # Path to a specific folder (e.g., one DM or group chat)
    # result_folder = r"YOUR-RESULT-FOLDER-HERE"  # Path to store the results
    # print(f"Processing single folder: {os.path.basename(single_folder)}")
    # process_folder(single_folder)

    end = time.time()
    print(f"Execution Time: {end - start:.2f} seconds")
