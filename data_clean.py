"""
data_clean.py

Cleans raw WhatsApp chat export data so it is ready for text analysis.
Steps: drop system/media messages -> strip code blocks -> tokenize into
clean, lowercase words (removing links, emojis, phone numbers, etc.)

No classes here on purpose. Just plain functions, grouped by job:
    1. Constants / config
    2. Code-block detection
    3. Message-level cleaning (drop junk rows)
    4. Word-level cleaning (clean one token)
    5. Tokenizer (turn one message into a list of clean words)
"""

import re
import pandas as pd
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound


# ============================================================
# 1. CONSTANTS
# ============================================================

# Programming languages we treat as "code", not natural language text
CODE_LANGUAGES = {
    'C', 'C++', 'Java', 'Python', 'JavaScript', 'TypeScript',
    'PHP', 'Ruby', 'Go', 'Rust', 'Swift', 'Kotlin', 'Scala',
    'SQL', 'HTML', 'CSS', 'Bash', 'Shell', 'PowerShell',
    'R', 'MATLAB', 'Perl', 'Lua', 'Haskell', 'Dart',
    'Assembly', 'Makefile', 'Dockerfile', 'YAML', 'JSON', 'XML',
}

# WhatsApp system-style text we always want to throw away
DROP_EXACT_MESSAGES = {
    '<Media omitted>',
    'This message was deleted',
    'You deleted this message',
}

# Leftover WhatsApp metadata words (not real chat content)
META_TOKENS = {
    'omitted', 'attached', 'file', 'sticker', 'image', 'video',
    'audio', 'gif', 'document', 'deleted', 'missed', 'call',
    'null', 'undefined',
}

# One emoji/pictograph pattern, compiled once, reused everywhere.
# (Message-level cleaning needs a couple of extra ranges that word-level
# cleaning does not, so we keep a shared "core" and a "message" superset.)
_EMOJI_CORE = (
    '\U0001F600-\U0001F64F'   # emoticons
    '\U0001F300-\U0001F5FF'   # symbols & pictographs
    '\U0001F680-\U0001F6FF'   # transport & map
    '\U0001F1E0-\U0001F1FF'   # flags
    '\u2702-\u27B0'           # dingbats
    '\u24C2-\U0001F251'       # enclosed characters
    '\U0001F900-\U0001F9FF'   # supplemental symbols
    '\U0001FA00-\U0001FA6F'   # chess symbols etc.
    '\U0001FA70-\U0001FAFF'   # additional symbols
    '\u2500-\u2BEF'           # box drawings, misc technical
    '\u3000-\u303F'           # CJK symbols
    '\u3200-\u32FF'           # enclosed CJK
    '\uFE30-\uFE4F'           # CJK compatibility forms
)

EMOJI_PATTERN_MESSAGE = re.compile(
    '[' + _EMOJI_CORE + '\u2600-\u26FF' + '\u2068\u2069' + ']+',  # + weather/misc + WhatsApp control chars
    flags=re.UNICODE,
)

EMOJI_PATTERN_WORD = re.compile('[' + _EMOJI_CORE + ']+', flags=re.UNICODE)

# Invisible / zero-width unicode characters that sneak into WhatsApp exports
INVISIBLE_CHARS = [
    '\u200e', '\u200d', '\u200c', '\u200f',
    '\u200b', '\u00ad', '\ufeff', '\u200a',
    '\u2060', '\u2062', '\u2063', '\u2064',
]

# Curly quotes / dashes we normalize to spaces during tokenizing
SMART_PUNCT_CHARS = ['\u2018', '\u2019', '\u201c', '\u201d', '\u2013', '\u2014', '\u2015', '\u00b7']

URL_PATTERN = re.compile(r'https?://\S+|ftp://\S+', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'[\w\.\+\-]+@[\w\-]+\.[\w\.]+', re.IGNORECASE)

STRIP_PUNCT_CHARS = r"""[.?!,;:<>(){}\[\]\-\*_/\\|@~#%^&+=`'"…•·।]"""


# ============================================================
# 2. CODE-BLOCK DETECTION
# ============================================================

def is_code_block(text: str, min_lines: int = 2) -> bool:
    """Return True if pygments thinks this block of text is programming code."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < min_lines:
        return False

    try:
        lexer = guess_lexer(text)
        lexer_name = lexer.name

        if 'text' in lexer_name.lower():
            return False

        for lang in CODE_LANGUAGES:
            if lang.lower() in lexer_name.lower():
                return True

    except ClassNotFound:
        return False

    return False


def remove_code_from_message(text: str) -> str:
    """Split a message into blocks, drop code blocks, keep normal text."""
    blocks = re.split(r'\n{2,}', text.strip())
    clean_blocks = [block.strip() for block in blocks if not is_code_block(block)]
    return ' '.join(clean_blocks).strip()


# ============================================================
# 3. MESSAGE-LEVEL CLEANING (drop junk rows from the dataframe)
# ============================================================

def get_text_only_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with only real, human-typed text messages left."""
    df = df[df['sender'] != 'System']
    df = df[~df['message'].isin(DROP_EXACT_MESSAGES)]
    df = df.copy()

    df['message'] = df['message'].apply(lambda x: x.replace(' <This message was edited>', ''))
    df['message'] = df['message'].apply(lambda x: EMOJI_PATTERN_MESSAGE.sub(' ', x))
    df['message'] = df['message'].apply(remove_code_from_message)

    df = df[df['message'].str.strip() != '']
    return df.reset_index(drop=True)


# ============================================================
# 4. WORD-LEVEL CLEANING (clean one token)
# ============================================================

def clean_word(word: str) -> str:
    """
    Clean one word by removing links, emails, phone numbers, file names,
    punctuation, and other WhatsApp-specific noise.
    Returns the cleaned word, or an empty string if the word should be dropped.
    """
    # 0. Drop URLs and emails
    if re.search(r'https?://|ftp://', word, re.IGNORECASE):
        return ''
    if re.match(r'^[\w\.\+\-]+@[\w\-]+\.[\w\.]+$', word, re.IGNORECASE):
        return ''
    if re.search(r'\.(com|in|org|net|io|gov|co|me|app|online|edu|uk|us|info|biz)\b', word, re.IGNORECASE):
        return ''
    if re.search(
        r'(forms\.gle|drive\.google|docs\.google|meet\.google'
        r'|youtube\.com|youtu\.be|bit\.ly|tinyurl)',
        word, re.IGNORECASE,
    ):
        return ''

    # 1. Remove zero-width / invisible unicode characters
    for ch in INVISIBLE_CHARS:
        word = word.replace(ch, '')

    # 2. Strip emojis and pictographic symbols
    word = EMOJI_PATTERN_WORD.sub('', word)

    # 3. Remove WhatsApp mention prefixes (@~Name -> Name)
    word = re.sub(r'^[@~]+', '', word)

    # 4. Remove phone numbers
    if re.match(r'^\+?\d[\d\s\-]{7,}$', word):
        return ''

    # 5. Remove file/media references
    if re.search(r'\.(vcf|pdf|jpg|jpeg|png|mp4|mp3|zip|exe|apk|docx?|xlsx?|pptx?|csv|txt)$', word, re.IGNORECASE):
        return ''

    # 6. Remove poll artifacts
    poll_patterns = [
        r'^\(?\d+\s*votes?\)?$',
        r'^option:?$',
        r'^poll:?$',
        r'^\(\d+$',
        r'^\d+\)$',
    ]
    for pattern in poll_patterns:
        if re.match(pattern, word, re.IGNORECASE):
            return ''

    # 7. Remove date/time stamps
    if re.match(r'^\d{1,2}[:/\-]\d{1,2}([:/\-]\d{2,4})?$', word):
        return ''

    # 8. Remove standalone numbers and number+symbol combos
    if re.match(r'^[₹Rs\.]*\d[\d,./\-₹]*[/\-]?$', word):
        return ''

    # 9. Remove long numeric sequences (roll numbers etc)
    if re.match(r'^[A-Z]{1,5}\d{4,}$', word) or re.match(r'^\d{6,}$', word):
        return ''

    # 10. Clean embedded special characters
    word = re.sub(r'[@~|\\]', '', word)

    # 11. Strip leading/trailing punctuation clusters
    word = re.sub(r'^' + STRIP_PUNCT_CHARS + r'+', '', word)
    word = re.sub(STRIP_PUNCT_CHARS + r'+$', '', word)

    # 12. Remove symbols-only tokens
    if re.match(r'^[^a-zA-Z\u0900-\u09FF\u0980-\u09FF]+$', word):
        return ''

    # 13. Remove noise: single-character tokens (except a few valid English words)
    if len(word) == 1 and word not in ('a', 'A', 'I', 'i'):
        if not re.match(r'[\u0900-\u09FF]', word):
            return ''

    # 14. Normalize repeated punctuation (e.g. '...', '!!!')
    word = re.sub(r'[.?!]{2,}', ' ', word).strip()

    # 15. Remove leftover WhatsApp metadata words
    if word.lower() in META_TOKENS:
        return ''

    return word.strip()


# ============================================================
# 5. TOKENIZER (turn one message string into a list of clean words)
# ============================================================

def tokenize_message(message: str) -> list:
    """Turn one message string into a list of clean, lowercase words."""
    if not isinstance(message, str) or not message.strip():
        return []

    message = URL_PATTERN.sub(' ', message)
    message = EMAIL_PATTERN.sub(' ', message)

    text = message.lower()
    text = re.sub(r'[.?!,;:<>(){}\[\]/\\|@~#\n\t\r]', ' ', text)
    text = text.replace('_', ' ').replace('*', ' ')
    for ch in SMART_PUNCT_CHARS:
        text = text.replace(ch, ' ')

    cleaned_words = []
    for raw_token in text.split():
        cleaned = clean_word(raw_token)
        for sub_word in cleaned.split():
            if sub_word.strip():
                cleaned_words.append(sub_word.strip())

    return cleaned_words