import os
import ast
import regex as re
import pandas as pd
import kagglehub

#pd.set_option('display.max_colwidth', None)
path = kagglehub.dataset_download("thedevastator/dailydialog-unlock-the-conversation-potential-in")

train_df = pd.read_csv(os.path.join(path, "train.csv"))
valid_df = pd.read_csv(os.path.join(path, "validation.csv"))
test_df  = pd.read_csv(os.path.join(path, "test.csv"))

def split_utterances(cell: list) -> list[str]:
    if isinstance(cell, list) and len(cell) == 1 and isinstance(cell[0], str):
        text = cell[0].strip()

        utterances = re.split(r'\s{2,}', text)

        return [utt.strip() for utt in utterances if utt.strip()]

    return []

def pre_process_df(df: pd.DataFrame) -> tuple[list[list[str]], list[list[int]]]:
    df['dialog'] = df['dialog'].apply(ast.literal_eval)
    dialogs = df['dialog'].apply(split_utterances).tolist()

    df['emotion'] = df['emotion'].apply(lambda cell: [int(x) for x in cell.strip('[]').split()])
    emotions = df['emotion'].tolist()

    return dialogs, emotions

def process_df(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    dialogs, emotions = pre_process_df(df)
    all_dialogs = []
    all_emotions = []
    for dialog, emotion in zip(dialogs, emotions):
        for text, emo in zip(dialog, emotion):
            all_dialogs.append(text)
            all_emotions.append(emo)
    return all_dialogs, all_emotions

train_dialogs, train_emotions = process_df(train_df)
valid_dialogs, valid_emotions = process_df(valid_df)
test_dialogs, test_emotions = process_df(test_df)
