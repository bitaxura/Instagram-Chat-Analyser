import os
import ast
import pandas as pd
import kagglehub

def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    single_speech = []
    for _, row in df.iterrows():
        dialog = ast.literal_eval(row['dialog'])
        emotion = list(map(int, row['emotion'].strip('[]').split()))
        for text, emo in zip(dialog, emotion):
            single_speech.append({"text": text.strip(), "label": emo})
    return pd.DataFrame(single_speech)

path = kagglehub.dataset_download("thedevastator/dailydialog-unlock-the-conversation-potential-in")

train_df = pd.read_csv(os.path.join(path, "train.csv"))
valid_df = pd.read_csv(os.path.join(path, "validation.csv"))
test_df  = pd.read_csv(os.path.join(path, "test.csv"))

train_single_speech_df = flatten_df(train_df)
valid_single_speech_df = flatten_df(valid_df)
test_single_speech_df  = flatten_df(test_df)