import platform
import regex as re
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Glyph .* missing from font"
)

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
PUNCT_PATTERN = re.compile(r'[^\w\s\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]')
EMOJI_PATTERN = re.compile(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]')
ATTACHMENT_PATTERN = re.compile(r'\b(\w+)\s+sent an attachment\b')

if platform.system() == "Windows":
    SYS_FONT_PATH = r"C:\Windows\Fonts\seguiemj.ttf"
elif platform.system() == "Darwin":
    SYS_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
elif platform.system() == "Linux":
    SYS_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
else:
    SYS_FONT_PATH = None

STOP_WORDS = {
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

def get_analysis_functions(Analyzer):
    return {
        "total_messages": Analyzer.get_total_messages,
        "messages_sent": Analyzer.count_messages_sent,
        "most_frequent_messages": Analyzer.get_most_frequent_messages,
        "most_frequent_emojis_sent": Analyzer.get_most_frequent_emojis_sent,
        "most_frequent_emojis_reacted": Analyzer.get_most_frequent_emojis_reacted,
        "most_active_days": Analyzer.most_active_days,
        "average_message_length": Analyzer.average_message_length,
        "message_streaks": Analyzer.get_message_streaks,
        "days_active": Analyzer.days_active,
    }

def get_plot_functions(Visualizer):    
    return [
        Visualizer.day_time_graph,
        Visualizer.build_freq_graph,
        Visualizer.generate_wordcloud,
        Visualizer.generate_pie_chart
    ]
