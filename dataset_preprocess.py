import re
from datetime import datetime
import pandas as pd
import numpy as np


# =====================================
#  Line pattern and date/time helpers
# =====================================

# Regex that matches one WhatsApp chat line: date, time, sender (optional), message
CHAT_LINE_PATTERN = re.compile(
    r'^\[?((?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})|(?:\d{4}[/.-]\d{1,2}[/.-]\d{1,2})),\s*'  # date
    r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)'  # time
    r'(?:\]\s*|\s*-\s*)'                             # separator
    r'(?:([^:\n]+?):\s)?'                   # sender (optional, System messages have none)
    r'(.*)$'                                # message text
)

# Order to try when guessing which number is Day / Month / Year.
# WhatsApp usually exports as Month/Day/Year (e.g. 8/20/24), so we try that first.
DATE_FORMAT_PRIORITY = [
    ('M', 'D', 'Y'),
    ('D', 'M', 'Y'),
    ('Y', 'M', 'D'),
]


def normalize_date(date_str: str) -> str | None:
    """
    Normalize a date string into the standard ``DD-MM-YYYY`` format.

    The function accepts date strings separated by ``/``, ``-``, or ``.``
    (e.g., ``8/20/24``, ``20-08-2024``, ``2024.08.20``). It attempts to
    interpret the date using the formats defined in
    ``DATE_FORMAT_PRIORITY``. Two-digit years are automatically expanded
    to four digits.

    Args:
        date_str (str):
            Input date string to normalize.

    Returns:
        str | None:
            The normalized date in ``DD-MM-YYYY`` format if parsing is
            successful; otherwise, ``None``.

    Examples:
        >>> normalize_date("8/20/24")
        '20-08-2024'

        >>> normalize_date("20-08-2024")
        '20-08-2024'

        >>> normalize_date("2024.08.20")
        '20-08-2024'

        >>> normalize_date("invalid")
        None
    """
    if not date_str:
        return None
    date_str = str(date_str).strip()
    parts = re.split(r'[/.-]', date_str)
    if len(parts) != 3:
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None

    for order in DATE_FORMAT_PRIORITY:
        role_map = dict(zip(order, nums))
        day, month, year = role_map['D'], role_map['M'], role_map['Y']
        if year < 100:
            year += 2000 if year < 70 else 1900
        if not (1 <= month <= 12) or not (1 <= day <= 31):
            continue
        try:
            parsed_date = datetime(year, month, day)
            return parsed_date.strftime('%d-%m-%Y')
        except ValueError:
            continue
    return None


def normalize_time(time_str: str) -> str:
    """
    Normalize a time string into the standard 24-hour ``HH:MM`` format.

    The function accepts both 12-hour (AM/PM) and 24-hour time formats.
    If the input cannot be parsed using the supported formats, the original
    string is returned unchanged.

    Supported input formats:
        - ``HH:MM AM/PM`` (e.g., ``10:15 AM``)
        - ``HH:MMAM/PM`` (e.g., ``10:15AM``)
        - ``HH:MM`` (24-hour format, e.g., ``22:15``)

    Args:
        time_str (str):
            Input time string to normalize.

    Returns:
        str:
            The normalized time in ``HH:MM`` (24-hour) format if parsing
            is successful; otherwise, the original input string.

    Examples:
        >>> normalize_time("10:15 AM")
        '10:15'

        >>> normalize_time("10:15PM")
        '22:15'

        >>> normalize_time("22:15")
        '22:15'

        >>> normalize_time("invalid")
        'invalid'
    """
    
    time_str = time_str.strip()
    for fmt in ('%I:%M:%S %p', '%I:%M:%S%p', '%H:%M:%S %p', '%H:%M:%S', '%I:%M %p', '%I:%M%p', '%H:%M'):
        try:
            return datetime.strptime(time_str, fmt).strftime('%H:%M')
        except ValueError:
            continue
    return time_str

# =====================================
# Parse the chat export file
# =====================================

def parse_whatsapp_chat_from_file(file_path: str) -> list[dict]:
    """
    Parse a WhatsApp chat export into a structured list of message records.

    The function reads a WhatsApp exported ``.txt`` chat file and extracts
    individual messages using ``CHAT_LINE_PATTERN``. Multi-line messages are
    automatically merged into a single message. System-generated events
    (e.g., group notifications) that do not contain a sender are assigned
    the sender name ``"System"``.

    Each parsed message includes both the original and normalized date and
    time values.

    Args:
        file_path (str):
            Path to the exported WhatsApp chat text file.

    Returns:
        list[dict]:
            A list of dictionaries, where each dictionary contains the
            following keys:

            - ``date_raw`` (str): Original date string from the chat.
            - ``date_ddmmyyyy`` (str | None): Normalized date in
              ``DD-MM-YYYY`` format, or ``None`` if parsing fails.
            - ``time_raw`` (str): Original time string from the chat.
            - ``time_24hr`` (str): Time normalized to ``HH:MM`` (24-hour)
              format, or the original value if parsing fails.
            - ``sender`` (str): Sender's name, or ``"System"`` for
              system-generated messages.
            - ``message`` (str): Complete message text, including any
              merged multi-line content.

    Examples:
        >>> messages = parse_whatsapp_chat("chat.txt")
        >>> len(messages)
        250

        >>> messages[0]["sender"]
        'Alice'

        >>> messages[0]["message"]
        'Hello, how are you?'
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    messages = []
    current = None
    for line in lines:
        match = CHAT_LINE_PATTERN.match(line)
        if match:
            if current:
                current['message'] = current['message'].strip()
                messages.append(current)
            date, time_str, sender, text = match.groups()
            current = {
                'date_raw': date,
                'date_ddmmyyyy': normalize_date(date),
                'time_raw': time_str,
                'time_24hr': normalize_time(time_str),
                'sender': sender if sender else 'System',
                'message': text,
            }
        else:
            # This line is a continuation of the previous message (no date/time prefix)
            if current:
                current['message'] += '\n' + line

    if current:
        current['message'] = current['message'].strip()
        messages.append(current)

    return messages


def build_chat_dataframe_from_file(file_path: str) -> pd.DataFrame:
    """
    Full pipeline: parse the chat file and return a ready-to-use DataFrame
    with clean date/time columns and helper columns (day, month, year, hour...).
    """
    messages = parse_whatsapp_chat_from_file(file_path)
    df = pd.DataFrame(messages)

    df = df.rename(columns={
        'date_ddmmyyyy': 'date_formatted',
        'time_24hr': 'time_formatted',
    })

    df['date_formatted'] = pd.to_datetime(df['date_formatted'], format='%d-%m-%Y', errors='coerce')
    df['time_formatted'] = pd.to_datetime(df['time_formatted'], format='%H:%M', errors='coerce').dt.time

    df['day'] = df['date_formatted'].dt.day
    df['month'] = df['date_formatted'].dt.month_name()
    df['year'] = df['date_formatted'].dt.year
    df['day_name'] = df['date_formatted'].dt.day_name()

    df['hours'] = df['time_formatted'].apply(lambda t: t.hour if pd.notnull(t) else np.nan)
    df['minutes'] = df['time_formatted'].apply(lambda t: t.minute if pd.notnull(t) else np.nan)

    return df.reset_index(drop=True)


def parse_whatsapp_chat_from_string(content: str) -> list[dict]:
    """
    Parse a WhatsApp chat export from a string into a structured list of message records.
    """

    lines = content.split('\n')
    messages = []
    current = None
    for line in lines:
        match = CHAT_LINE_PATTERN.match(line)
        if match:
            if current:
                current['message'] = current['message'].strip()
                messages.append(current)
            date, time_str, sender, text = match.groups()
            current = {
                'date_raw': date,
                'date_ddmmyyyy': normalize_date(date),
                'time_raw': time_str,
                'time_24hr': normalize_time(time_str),
                'sender': sender if sender else 'System',
                'message': text,
            }
        else:
            # This line is a continuation of the previous message (no date/time prefix)
            if current:
                current['message'] += '\n' + line

    if current:
        current['message'] = current['message'].strip()
        messages.append(current)

    return messages


def build_chat_dataframe_from_string(content: str) -> pd.DataFrame:

    messages = parse_whatsapp_chat_from_string(content)
    df = pd.DataFrame(messages)

    if df.empty:
        return pd.DataFrame(columns=[
            'date_raw', 'date_formatted', 'time_raw', 'time_formatted',
            'sender', 'message', 'day', 'month', 'year', 'day_name',
            'hours', 'minutes',
        ])

    df = df.rename(columns={
        'date_ddmmyyyy': 'date_formatted',
        'time_24hr': 'time_formatted',
    })

    df['date_formatted'] = pd.to_datetime(df['date_formatted'], format='%d-%m-%Y', errors='coerce')
    df['time_formatted'] = pd.to_datetime(df['time_formatted'], format='%H:%M', errors='coerce').dt.time

    df['day'] = df['date_formatted'].dt.day
    df['month'] = df['date_formatted'].dt.month_name()
    df['year'] = df['date_formatted'].dt.year
    df['day_name'] = df['date_formatted'].dt.day_name()

    df['hours'] = df['time_formatted'].apply(lambda t: t.hour if pd.notnull(t) else np.nan)
    df['minutes'] = df['time_formatted'].apply(lambda t: t.minute if pd.notnull(t) else np.nan)

    return df.reset_index(drop=True)

# =====================================
# DataFram into Whatsapp txt
# =====================================
def dataframe_to_whatsapp_txt(df, output_file_path):
    """
    Reverses the build_chat_dataframe process by converting a DataFrame
    back into the standard WhatsApp .txt export format.
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Use raw columns if they exist to preserve original format,
            # otherwise fallback to formatted ones.
            date = row.get('date_raw', row['date_formatted'].strftime('%m/%d/%y'))
            time = row.get('time_raw', row['time_formatted'].strftime('%I:%M %p'))
            sender = row['sender']
            message = str(row['message'])

            if sender == 'System':
                # System messages don't have a 'Sender: ' prefix
                line = f"{date}, {time} - {message}\n"
            else:
                line = f"{date}, {time} - {sender}: {message}\n"

            f.write(line)
    print(f"Successfully saved DataFrame to {output_file_path}")

