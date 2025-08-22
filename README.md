# Instagram DM Analyzer

This tool analyzes Instagram direct message (DM) data downloaded from your account. It supports bulk processing of all conversations or analyzing one folder at a time. The script outputs statistical summaries, visualizations, word clouds, and emotion analysis for each conversation.

## Features

* Counts total messages, messages per user, and frequent phrases
* Detects the longest message streaks and longest gaps
* Calculates average message lengths per user
* Emotion classification of messages using machine learning
* Generates:

  * Word cloud from meaningful words
  * Daily message frequency graph
  * Heatmap of message activity by day and hour
  * Emotion analysis predictions for messages
  * Pie chart of message split by user

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Note**: The emotion classification feature requires additional setup. You'll need to run `emotion_classifier.py` once to download the dataset and train the model before using the main analyzer.

---

## Project Structure

The project is organized as follows:

```
IG Analyzer/
├── main.py                 # Main script to process Instagram DM data
├── config.py               # Configuration file for setting up paths and parameters
├── emotion_classifier.py   # Script to train the emotion classification model
├── requirements.txt        # List of Python dependencies
├── README.md              # Project documentation
└── LICENSE                # License file
```

Each file serves a specific purpose:

- **main.py**: The entry point of the application. Contains the logic for processing all or single folders, including emotion analysis.
- **config.py**: Stores configurable parameters like folder paths, text processing patterns, and analysis functions.
- **emotion_classifier.py**: Downloads dataset and trains a machine learning model for emotion classification of messages.
- **requirements.txt**: Lists all the Python libraries required to run the project.
- **README.md**: Provides detailed instructions on how to use the tool.
- **LICENSE**: Specifies the licensing terms for the project.

---

## Initial Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Emotion Classification Model (One-time setup)
Before using the main analyzer, you need to train the emotion classification model:

```bash
python emotion_classifier.py
```

This will:
- Download the DailyDialog dataset from Kaggle
- Train a logistic regression model for emotion classification
- Save the trained model as `emotion_classifier.pkl` and `vectorizer.pkl`

**Note**: This step only needs to be done once. The trained model files will be reused for all future analyses.

---

## How to Get Your Instagram Data

To use this tool, you'll need to download your Instagram data (specifically your messages). Here's how to do it:
1. Open Instagram.

2. Go to Instagram **Settings**.  
    You can do this either on the **website** or the **mobile app**.

3. Then click on **Accounts Center**.

4. Navigate to:  
    **Your information and permissions** → **Download your information**.

5. On the **Download or transfer your information** page:  
    * If you want a backup of everything (messages, photos, stories, etc.), click **All your information**.  
    * If you only want your DMs, click **Some of your information**.

6. Scroll down to the section called **Your Instagram activity**.

7. Check the box next to **Messages**, then click **Next**.

8. Choose **Download to device**.

9. Set the **date range** for the messages you want to download.

10. Under **Format**, choose **JSON**.

11. Choose a **media quality** (any is fine for this project).

12. Click **Create files**.

Once you've submitted the request, Instagram will prepare your data. **It may take a few hours to a couple of days**, depending on how much you're downloading. You'll get a notification or email when it's ready.

When the download is complete:

* Extract the ZIP file you receive.
* Look for the folder named something like `messages/inbox`.
* That's the folder you'll point to in this project as either your `main_folder`.
* If you want to do 1 dm only navigate into `message/inbox` and put the folder path you want as your `single_folder`

---


## How to Use

1. Download your Instagram data and unzip it.

2. **Make sure you've completed the Initial Setup** (especially training the emotion model).

3. Open `main.py` and choose **only one** of the two options below:

   * If you want to process **all inbox folders**, use Option 1.
   * If you want to process **a single conversation folder**, use Option 2.
   * Comment out the unused block as shown below.

---

### Option 1: Analyze All Inbox Folders

This will process every conversation inside your main Instagram inbox folder.

Uncomment and fill out this block:

```python
main_folder = r"YOUR-MAIN-FOLDER-HERE"  # Path to the main inbox folder, it should end in "/messages/inbox"
result_folder = r"YOUR-RESULT-FOLDER-HERE"  # Path of folder to store the results
minimum_messages = 30 # Minimum number of messages in a conversation to be processed, choose based on your needs
minimum_days = 3 # Minimum number of active days in a conversation to be processed, choose based on your needs

folders = FileOperations.find_all_folder(main_folder)
print("Processing all inbox folders")

print(f"Processing {len(folders)} folders using {mp.cpu_count()} processes")

with mp.Pool(processes=mp.cpu_count()) as pool:
    pool.map(process_folder, folders)
```

And make sure the **single folder** block below is commented out:

```python
# single_folder = r"YOUR-SINGLE-FOLDER-HERE"  # Path to a specific folder (e.g., one DM or group chat)
# result_folder = r"YOUR-RESULT-FOLDER-HERE"  # Path to store the results
# print(f"Processing single folder: {os.path.basename(single_folder)}")
# process_folder(single_folder)
```

Then run the script:

```bash
python main.py
```

---

### Option 2: Analyze One Specific Folder

This will only analyze one folder (a single DM or group chat).

Uncomment and fill out this block:

```python
single_folder = r"YOUR-SINGLE-FOLDER-HERE"  # e.g., r"C:\Users\You\instagram\inbox\johndoe_123"
result_folder = r"YOUR-RESULT-FOLDER-HERE"  # e.g., r"C:\Users\You\results"

print(f"Processing single folder: {os.path.basename(single_folder)}")
process_folder(single_folder)
```

And make sure the **all inbox folders** block above it is commented out:

```python
# main_folder = r"YOUR-MAIN-FOLDER-HERE"  # Path to the main inbox folder, it should end in "/messages/inbox"
# result_folder = r"YOUR-RESULT-FOLDER-HERE"  # Path of folder to store the results 
# folders = FileOperations.find_all_folder(main_folder)
# print("Processing all inbox folders")
#
# print(f"Processing {len(folders)} folders using {mp.cpu_count()} processes")
#
# with mp.Pool(processes=mp.cpu_count()) as pool:
#     pool.map(process_folder, folders)
```

Then run the script:

```bash
python main.py
```

---


## Output Folder Structure

The `result_folder` will be populated with subfolders for each conversation processed. The structure differs depending on whether you run batch or single-folder processing:

### Example for Batch Processing

```
results/
├── JohnDoe/
│   ├── analysis_results.json
│   ├── day_time_graph.png
│   ├── messages_per_day.png
│   ├── wordcloud.png
│   └── pie_chart.svg
├── JaneSmith/
│   ├── analysis_results.json
│   ├── ...
```

Each folder name (like `JohnDoe`) comes from the original folder name up to the last underscore (e.g., `JohnDoe_1234` → `JohnDoe`).

### Example for Single Folder Processing

```
results/
└── MyChat/
    ├── analysis_results.json
    ├── day_time_graph.png
    ├── messages_per_day.png
    ├── wordcloud.png
    └── pie_chart.svg
```

If a folder name already exists, a numeric suffix is added automatically (e.g., `JohnDoe1`, `JohnDoe2`, etc.).

### Contents of `analysis_results.json`

```json
{
  "total_messages": 1234,
  "messages_sent": {
    "message_per_user": {
      "Alice": 567,
      "Bob": 667
    },
    "message_proportions": {
      "Alice": 0.459,
      "Bob": 0.541
    }
  },
  "most_frequent_messages": {
    "lol": 32,
    "yeah": 28,
    "okay": 25,
    "sure": 22,
    "thanks": 20
  },
  "most_frequent_emojis_sent": {
    "Alice": {
      "😂": 12,
      "❤️": 8,
      "👍": 5
    },
    "Bob": {
      "😎": 7,
      "🔥": 4,
      "😅": 3
    }
  },
  "most_frequent_emojis_reacted": {
    "Alice": {
      "❤️": 10,
      "😂": 6
    },
    "Bob": {
      "👍": 9,
      "😮": 4
    }
  },
  "most_active_days": [
    {
      "date": "2021-05-14",
      "messages": 40
    },
    {
      "date": "2021-03-20",
      "messages": 35
    },
    {
      "date": "2021-04-02",
      "messages": 33
    },
    {
      "date": "2021-06-18",
      "messages": 32
    },
    {
      "date": "2021-05-01",
      "messages": 30
    }
  ],
  "average_message_length": {
    "Alice": 12.3,
    "Bob": 15.8
  },
  "message_streaks": {
    "max_message_streak": {
      "length": 12,
      "start": "2021-01-02",
      "end": "2021-01-13"
    },
    "max_gap": {
      "length": 19,
      "start": "2021-02-10",
      "end": "2021-03-01"
    }
  },
  "days_active": 85
}
```

---

## Emotion Classification

The tool includes emotion classification functionality that analyzes the emotional content of messages. The emotion classifier uses a machine learning model trained on the DailyDialog dataset to predict emotions in text.

### Emotion Categories
The model classifies messages into the following emotion categories:
- 0: No emotion
- 1: Anger
- 2: Disgust
- 3: Fear
- 4: Happiness
- 5: Sadness
- 6: Surprise

### Usage
The emotion analysis is available through the `emotions_game()` function in `main.py`, which returns a DataFrame with predicted emotions for each message. This feature processes and normalizes text before classification to improve accuracy.

---

## Known Issues
* Chats with no messages or just images do not work at all
* Emotion classification requires the model files to be present (run `emotion_classifier.py` first)
* The model isn't very accurate yet

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.