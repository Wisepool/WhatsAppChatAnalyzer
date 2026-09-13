import re
import os
from datetime import datetime, timedelta
import pandas as pd
from urlextract import URLExtract
import plotly.express as px

from data_clean import (
    get_text_only_messages,
    tokenize_message
)
from collections import Counter
from config import BENGALI_FONT_PATH, STOPWORDS_FILE_PATHS
import numpy as np
from wordcloud import WordCloud
import emoji

# ===========================
# -- Material Color Palette --
# ===========================
# Google Material Design colors, used everywhere for a consistent look.
MATERIAL_COLORS = [
    "#E53935",  # Red 600
    "#1E88E5",  # Blue 600
    "#43A047",  # Green 600
    "#FB8C00",  # Orange 600
    "#8E24AA",  # Purple 600
    "#00ACC1",  # Cyan 600
    "#FDD835",  # Yellow 600
    "#3949AB",  # Indigo 600
    "#D81B60",  # Pink 600
    "#00897B",  # Teal 600
    "#F4511E",  # Deep Orange 600
    "#6D4C41",  # Brown 600
    "#7CB342",  # Light Green 600
    "#5E35B1",  # Deep Purple 600
    "#546E7A",  # Blue Grey 600
    "#C0CA33",  # Lime 600
    "#039BE5",  # Light Blue 600
    "#FFB300",  # Amber 600
]

# Continuous scale (light to dark Material Blue -> Purple), for heatmaps and
# any chart that colors by a numeric value.
MATERIAL_CONTINUOUS = [
    [0.0, "#E8EAF6"],   # Indigo 50
    [0.25, "#9FA8DA"],  # Indigo 200
    [0.5, "#5C6BC0"],   # Indigo 400
    [0.75, "#3949AB"],  # Indigo 600
    [1.0, "#1A237E"],   # Indigo 900
]

# Warm continuous scale (Amber -> Deep Orange -> Red), used where a "heat" feel fits.
MATERIAL_WARM = [
    [0.0, "#FFF8E1"],   # Amber 50
    [0.25, "#FFCA28"],  # Amber 400
    [0.5, "#FB8C00"],   # Orange 600
    [0.75, "#F4511E"],  # Deep Orange 600
    [1.0, "#B71C1C"],   # Red 900
]

# Green "activity" scale, kept for the GitHub-style heatmaps.
MATERIAL_GREEN = [
    [0.0, "#F1F8E9"],   # Light Green 50
    [0.05, "#C5E1A5"],  # Light Green 200
    [0.3, "#7CB342"],   # Light Green 600
    [0.6, "#43A047"],   # Green 600
    [1.0, "#1B5E20"],   # Green 900
]

# ===========================
# -- Other Functions --
# ===========================
def _person_label(person: str) -> str:
    """Turn 'all' into a readable label like 'All Members'."""
    return "All Members" if person == "all" else str(person)


def get_date_list(start_date, end_date, date_format="%d-%m-%Y"):
    # convert string to date object
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    
    date_list = []
    current = start
    
    while current <= end:
        date_list.append(current.strftime(date_format))
        current += timedelta(days=1)
    
    return date_list

# ============================
# --- Statistics Functions ---
# ============================

def get_total_members(df: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Return (count, list_of_names) of all members who sent at least one message.
    Excludes 'System' messages (case-insensitive).
    """
    # Use case-insensitive check for System and handle potential None values
    members = df[~df['sender'].fillna('').str.contains('^system$', case=False, na=False)]['sender'].unique()
    # Filter out any lingering None or empty strings
    members = [m for m in members if m and str(m).strip()]
    return len(members), members


def get_total_message_count(df: pd.DataFrame, person: str = "all") -> int:
    """Total number of messages sent (excluding System messages)."""
    if person == 'all':
        df = df[df['sender'] != 'System']
    else:
        df = df[df['sender'] == person]
    return df.shape[0]


def get_total_word_count(df: pd.DataFrame, person: str = "all") -> int:
    """Total number of words across all messages."""
    if person != 'all':
        df = df[df['sender'] == person]
    return df['message'].apply(lambda msg: len(msg.split())).sum()


def get_total_media_count(df: pd.DataFrame, person: str = "all") -> int:
    """Total number of media messages (photo, video, sticker etc)."""
    if person != 'all':
        df = df[df['sender'] == person]
    return df['message'].apply(lambda msg: "<Media omitted>" in msg).sum()


def get_total_link_count(df: pd.DataFrame, person: str = "all") -> tuple[int, pd.DataFrame]:
    """
    Find every link shared in the chat.
    Returns (total_count, links_dataframe).
    """
    extractor = URLExtract()
    links_data = []

    if person != 'all':
        df_filtered = df[df['sender'] == person].copy()
    else:
        df_filtered = df[df['sender'] != 'System'].copy()

    for _, row in df_filtered.iterrows():
        for url in extractor.find_urls(row['message']):
            links_data.append({
                'date': row['date_formatted'],
                'time': row['time_formatted'],
                'sender_name': row['sender'],
                'link': url,
            })

    links_df = pd.DataFrame(links_data)
    return len(links_data), links_df


def get_total_vcf_count(df: pd.DataFrame, person: str = "all") -> tuple[int, pd.DataFrame]:
    """
    Find every contact card (.vcf file) shared in the chat.
    Returns (total_count, vcf_dataframe).
    """
    if person != 'all':
        df_filtered = df[df['sender'] == person].copy()
    else:
        df_filtered = df[df['sender'] != 'System'].copy()

    vcf_mask = df_filtered['message'].str.contains(r'\.vcf \(file attached\)', case=False, na=False)
    vcf_df = df_filtered[vcf_mask].copy()

    return vcf_df.shape[0], vcf_df


def get_chat_date_range(df: pd.DataFrame, person: str = "all") -> tuple[str, str, str, str]:
    """
    Return (start_date, end_date, start_time, end_time) for the chat,
    or for one person if 'person' is given.
    """
    if person == 'all':
        df = df[df['sender'] != 'System']
    else:
        df = df[df['sender'] == person]

    start_date = df['date_formatted'].min().strftime('%d-%m-%Y')
    end_date = df['date_formatted'].max().strftime('%d-%m-%Y')
    start_time = df['time_formatted'].min().strftime('%H:%M')
    end_time = df['time_formatted'].max().strftime('%H:%M')

    return start_date, end_date, start_time, end_time

# ==================================
# -- Timeline Analysis Functions ---
# ==================================

def get_message_timeline(df: pd.DataFrame, person: str = 'all') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build daily and monthly message-count timelines.
    Returns (timeline_daily_df, timeline_monthly_df).
    """
    if person != 'all':
        df = df[df['sender'] == person].copy()
    else:
        df = df.copy()

    df = df[df['sender'] != 'System'].copy()

    # Daily timeline
    timeline_daily = (
        df.groupby('date_formatted').count()['message']
        .reset_index()
        .sort_values('date_formatted')
    )
    timeline_daily['time'] = (
        timeline_daily['date_formatted'].dt.day.astype(str) + '-' +
        timeline_daily['date_formatted'].dt.month.astype(str) + '-' +
        timeline_daily['date_formatted'].dt.year.astype(str)
    )

    # Monthly timeline
    df['month_num'] = df['date_formatted'].dt.month
    timeline_monthly = (
        df.groupby(['year', 'month_num', 'month']).count()['message']
        .reset_index()
        .sort_values(['year', 'month_num'])
    )
    timeline_monthly['time'] = timeline_monthly['month'] + '-' + timeline_monthly['year'].astype(str)

    return timeline_daily, timeline_monthly


def plot_daily_message_timeline(timeline_daily: pd.DataFrame) -> px.line:
    """Line chart of message count per day."""
    fig = px.line(
        timeline_daily,
        x='time',
        y='message',
        title='Daily Message Frequency Timeline',
        labels={'time': 'Date', 'message': 'Number of Messages'},
        markers=True,
        template='plotly_white',
        color_discrete_sequence=["#3949AB"],  # Indigo 600
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=6, color="#D81B60"))
    fig.update_layout(xaxis_tickangle=-45, height=500, hovermode='x unified')
    return fig


def plot_monthly_message_timeline(timeline_monthly: pd.DataFrame) -> px.line:
    """Line chart of message count per month."""
    fig = px.line(
        timeline_monthly,
        x='time',
        y='message',
        title='Monthly Message Frequency Timeline',
        labels={'time': 'Month-Year', 'message': 'Number of Messages'},
        markers=True,
        template='plotly_white',
        color_discrete_sequence=["#00897B"],  # Teal 600
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=6, color="#FB8C00"))
    fig.update_layout(xaxis_tickangle=-45, height=500, hovermode='x unified')
    return fig


def plot_most_active_day_and_month(df: pd.DataFrame, person: str = 'all') -> tuple[px.bar, px.bar]:
    """
    Bar chart of message count by day of week, and by month.
    Returns (fig_day, fig_month).
    """
    if person != 'all':
        df = df[df['sender'] == person].copy()
    else:
        df = df.copy()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    busy_day = df['day_name'].value_counts().reset_index()
    busy_day.columns = ['day_name', 'message_count']

    fig_day = px.bar(
        busy_day,
        x='day_name',
        y='message_count',
        title=f'Most Active Days: {_person_label(person)}',
        category_orders={'day_name': day_order},
        color='message_count',
        color_continuous_scale=MATERIAL_CONTINUOUS,
        text_auto=True,
    )
    fig_day.update_layout(xaxis_title='Day of Week', yaxis_title='Messages', template='plotly_white')

    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    busy_month = df['month'].value_counts().reset_index()
    busy_month.columns = ['month', 'message_count']

    fig_month = px.bar(
        busy_month,
        x='month',
        y='message_count',
        title=f'Most Active Months: {_person_label(person)}',
        category_orders={'month': month_order},
        color='message_count',
        color_continuous_scale=MATERIAL_WARM,
        text_auto=True,
    )
    fig_month.update_layout(xaxis_title='Month', yaxis_title='Messages', template='plotly_white')

    return fig_day, fig_month


# ==========================================
# --- Member Activity Analysis Functions ---
# ==========================================

def plot_most_active_members(df: pd.DataFrame, top_n: int = 20) -> tuple[pd.DataFrame, px.bar, px.pie]:
    """
    Build a horizontal bar chart and a pie chart of the members who sent the
    most messages.
    Returns (user_counts_df, fig_bar, fig_pie).
    """
    df_filtered = df[df['sender'] != 'System']

    user_counts = df_filtered['sender'].value_counts().reset_index()
    user_counts.columns = ['User', 'Message Count']

    if top_n != 'all' and isinstance(top_n, int):
        user_counts = user_counts.head(top_n)

    plot_df = user_counts.sort_values(by='Message Count', ascending=True)
    label = str(top_n) if top_n != 'all' else 'All'

    fig_bar = px.bar(
        plot_df,
        x='Message Count',
        y='User',
        orientation='h',
        title=f'Top {label} Active Users by Message Count',
        color='Message Count',
        color_continuous_scale=MATERIAL_CONTINUOUS,
        text_auto=True,
    )
    fig_bar.update_layout(
        yaxis_title='User Name',
        xaxis_title='Number of Messages',
        yaxis={'type': 'category'},
        margin=dict(l=200),
        height=max(400, len(user_counts) * 25),
        template='plotly_white',
    )

    fig_pie = px.pie(
        user_counts,
        values='Message Count',
        names='User',
        title=f'Message Share among Top {label} Users',
        hole=0.3,
        color_discrete_sequence=MATERIAL_COLORS,
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    return user_counts[['User', 'Message Count']], fig_bar, fig_pie


def get_chat_starters_and_enders(df: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, px.pie, px.pie]:
    """
    Find who sends the first and the last message on each date.
    Returns (starter_counts_df, ender_counts_df, fig_starters, fig_enders).
    """
    df_clean = df[df['sender'] != 'System'].copy()
    df_sorted = df_clean.sort_values(['date_formatted', 'time_formatted'])

    starters = df_sorted.groupby('date_formatted').first().reset_index()
    starter_counts = starters['sender'].value_counts().reset_index()
    starter_counts.columns = ['User', 'Count']

    enders = df_sorted.groupby('date_formatted').last().reset_index()
    ender_counts = enders['sender'].value_counts().reset_index()
    ender_counts.columns = ['User', 'Count']

    fig_starters = px.pie(
        starter_counts.head(top_n),
        values='Count',
        names='User',
        title=f'Top {top_n} Daily Conversation Starters (First Message of Day)',
        hole=0.4,
        color_discrete_sequence=MATERIAL_COLORS,
    )
    fig_starters.update_traces(textinfo='percent+label')

    fig_enders = px.pie(
        ender_counts.head(top_n),
        values='Count',
        names='User',
        title=f'Top {top_n} Daily Conversation Enders (Last Message of Day)',
        hole=0.4,
        color_discrete_sequence=MATERIAL_COLORS[::-1],
    )
    fig_enders.update_traces(textinfo='percent+label')

    return starter_counts, ender_counts, fig_starters, fig_enders


def plot_message_sunburst(df: pd.DataFrame) -> px.sunburst:
    """
    Full sunburst chart showing every member at the outer ring.
    Best used when the group has a small number of members.
    """
    df_clean = df[df['sender'] != 'System'].copy()

    sunburst_df = (
        df_clean.groupby(['year', 'month', 'day_name', 'sender'])
        .size()
        .reset_index(name='message_count')
    )

    fig = px.sunburst(
        sunburst_df,
        path=['year', 'month', 'day_name', 'sender'],
        values='message_count',
        title='Hierarchy of Messages: Year → Month → Weekday → Member',
        color='message_count',
        color_continuous_scale=MATERIAL_CONTINUOUS,
        height=660,
    )
    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    return fig

# ================
# --- Heatmaps ---
# ================

def plot_daily_activity_heatmap(df: pd.DataFrame, person: str = 'all', start_date: str = None, end_date: str = None, date_format: str = "%d-%m-%Y") -> px.density_heatmap:
    """
    GitHub-style heatmap: one cell per day, arranged by week (x) and
    day of week (y). Returns a Plotly figure, or None if no data matches.
    """
    if person != 'all':
        df = df[df['sender'] == person].copy()
    else:
        df = df.copy()

    if start_date:
        df = df[df['date_formatted'] >= pd.to_datetime(start_date, format=date_format)]
    if end_date:
        df = df[df['date_formatted'] <= pd.to_datetime(end_date, format=date_format)]

    if df.empty:
        print('No data found for the selected range/person.')
        return None

    daily_activity = df.groupby('date_formatted').size().reset_index(name='message_count')
    daily_activity['week'] = daily_activity['date_formatted'].dt.isocalendar().week
    daily_activity['year'] = daily_activity['date_formatted'].dt.year
    daily_activity['day_name'] = daily_activity['date_formatted'].dt.day_name()
    daily_activity['date_str'] = daily_activity['date_formatted'].dt.strftime('%d-%m-%Y')
    daily_activity['week_label'] = (
        daily_activity['year'].astype(str) + '-W' + daily_activity['week'].astype(str).str.zfill(2)
    )

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    custom_greens = MATERIAL_GREEN

    fig = px.density_heatmap(
        daily_activity,
        x='week_label',
        y='day_name',
        z='message_count',
        category_orders={'day_name': day_order},
        color_continuous_scale=custom_greens,
        title=f'Daily Message Activity Heatmap: {_person_label(person)}',
        hover_data={'week_label': True, 'day_name': True, 'message_count': True, 'date_str': True},
        labels={'week_label': 'Week', 'day_name': 'Day', 'message_count': 'Messages', 'date_str': 'Date'},
        text_auto=True,
    )
    fig.update_layout(
        xaxis_title='Weeks (Chronological)',
        yaxis_title='',
        template='plotly_white',
        height=450,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.update_traces(xgap=3, ygap=3)
    return fig


def plot_weekday_month_heatmap(df: pd.DataFrame, person: str = 'all') -> px.density_heatmap:
    """
    Heatmap of message count by (month, day of week).
    Returns a Plotly figure, or None if no data matches.
    """
    if person != 'all':
        df = df[df['sender'] == person].copy()
    else:
        df = df.copy()

    if df.empty:
        print('No data found for the selected person.')
        return None

    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    activity_matrix = df.groupby(['month', 'day_name']).size().reset_index(name='message_count')

    custom_greens = MATERIAL_GREEN

    fig = px.density_heatmap(
        activity_matrix,
        x='month',
        y='day_name',
        z='message_count',
        category_orders={'month': month_order, 'day_name': day_order},
        color_continuous_scale=custom_greens,
        title=f'Weekday vs Month Activity: {_person_label(person)}',
        labels={'month': 'Month', 'day_name': 'Day of Week', 'message_count': 'Messages'},
        text_auto=True,
    )
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='',
        template='plotly_white',
        height=500,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.update_traces(xgap=3, ygap=3)
    return fig


def plot_day_of_month_heatmap(df: pd.DataFrame, person: str = 'all') -> px.imshow:
    """
    Heatmap of message count for every day-of-month (1-31) against every month.
    Returns a Plotly figure, or None if no data matches.
    """
    if person != 'all':
        df = df[df['sender'] == person].copy()
    else:
        df = df.copy()

    if df.empty:
        print('No data found for the selected person.')
        return None

    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    matrix = df.groupby(['month', 'day']).size().unstack(fill_value=0)
    matrix = matrix.reindex(index=month_order, columns=range(1, 32), fill_value=0)

    custom_greens = MATERIAL_GREEN

    fig = px.imshow(
        matrix,
        labels=dict(x='Day of Month', y='Month', color='Messages'),
        x=list(range(1, 32)),
        y=month_order,
        color_continuous_scale=custom_greens,
        title=f'Activity Matrix: Day of Month vs Month ({_person_label(person)})',
        text_auto=True,
        aspect='auto',
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis_title='',
        template='plotly_white',
        height=700,
        margin=dict(l=100, r=20, t=50, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.update_traces(xgap=2, ygap=2)
    return fig


# ==================================
# -- Message Processing Functions --
# ==================================

def get_top_words(df: pd.DataFrame, stop_words_file_path: str | list, person: str = 'all') -> pd.DataFrame:
    """
    Return a DataFrame of (word, count), sorted from most to least frequent,
    with stopwords removed.
    Supports single file path (str/os.PathLike) or a list of paths.
    """
    if person != 'all':
        df = df[df['sender'] == person]

    df = get_text_only_messages(df)

    all_words = []
    for message in df['message']:
        all_words.extend(tokenize_message(message))

    # Handle both single path and list of paths
    if isinstance(stop_words_file_path, (str, os.PathLike)):
        paths = [stop_words_file_path]
    else:
        paths = stop_words_file_path

    stopwords = set()
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                stopwords.update(f.read().splitlines())

    filtered_words = [word for word in all_words if word not in stopwords]

    word_counts_df = pd.DataFrame(Counter(filtered_words).most_common(), columns=['word', 'count'])
    return word_counts_df.sort_values(by='count', ascending=False).reset_index(drop=True)


def plot_top_words_for_person(top_words_df: pd.DataFrame, top_n: int = 20) -> px.bar:

    if top_words_df.empty:
        return None

    top_words_df = top_words_df.head(top_n)

    fig = px.bar(
        top_words_df,
        x='count',
        y='word',
        orientation='h',
        title=f'Top {top_n} Most Frequent Words',
        labels={'count': 'Frequency Count', 'word': 'Words'},
        color='count',
        color_continuous_scale=MATERIAL_CONTINUOUS,
        text_auto=True,
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white',
        height=max(400, top_n * 25),
    )
    return fig


def make_word_cloud(df: pd.DataFrame, stop_words_file_path: str | list = STOPWORDS_FILE_PATHS, person: str = 'all', bengali_font_path: str | None = BENGALI_FONT_PATH,
                     width: int = 800, height: int = 800, max_words: int = 200) -> tuple[np.ndarray | None, dict, np.ndarray | None, dict]:
    """
    Build separate word clouds for Bengali script and English/Banglish script.
    Supports single file path (str/os.PathLike) or a list of paths for stopwords.
    """
    # get_top_words already handles the path/list logic
    word_df = get_top_words(df, stop_words_file_path, person=person)
    if word_df.empty or word_df['count'].sum() == 0:
        return None, {}, None, {}

    bengali_pattern = re.compile(r'[\u0980-\u09FF]')
    bengali_mask = word_df["word"].fillna("").apply(
        lambda x: bool(bengali_pattern.search(str(x)))
    )
    df_bengali = word_df[bengali_mask].copy()
    df_english = word_df[~bengali_mask].copy()

    freq_bengali = dict(zip(df_bengali['word'], df_bengali['count'])) if not df_bengali.empty else {}
    freq_english = dict(zip(df_english['word'], df_english['count'])) if not df_english.empty else {}

    common_kwargs = dict(
        background_color='white',
        width=width,
        height=height,
        min_font_size=10,
        max_words=max_words,
        colormap='turbo',
    )

    img_bengali = None
    if freq_bengali:
        try:
            wc_bengali = WordCloud(**common_kwargs, font_path=bengali_font_path)
            wc_bengali.generate_from_frequencies(freq_bengali)
            img_bengali = wc_bengali.to_array()
        except Exception:
            img_bengali = None

    img_english = None
    if freq_english:
        try:
            wc_english = WordCloud(**common_kwargs)
            wc_english.generate_from_frequencies(freq_english)
            img_english = wc_english.to_array()
        except Exception:
            img_english = None

    return img_bengali, freq_bengali, img_english, freq_english


def plot_word_frequency_bar(word_freq: dict, top_n: int = 20, label: str = '') -> px.bar:
    """
    Horizontal bar chart of the top N words from a {word: count} dictionary.
    'label' lets the caller say which language this chart is for (e.g. 'Bengali').
    Returns a Plotly figure, or None if word_freq is empty.
    """
    if not word_freq:
        return None

    top_items = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n]
    words, counts = zip(*top_items)
    plot_df = pd.DataFrame({'Word': words, 'Frequency': counts})

    title = f'Top {top_n} {label} Words by Frequency'.replace('  ', ' ')

    fig = px.bar(
        plot_df,
        x='Frequency',
        y='Word',
        orientation='h',
        title=title,
        color='Frequency',
        color_continuous_scale=MATERIAL_WARM,
        text='Frequency',
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white',
        height=max(400, top_n * 25),
    )
    return fig


def plot_message_length_distribution(df: pd.DataFrame, top_n: int = 10) -> px.box:
    """
    Box plot of message length (word count) for the top N most active senders.
    Returns a Plotly figure.
    """
    df_filtered = df[df['sender'] != 'System'].copy()
    df_filtered['word_count'] = df_filtered['message'].apply(lambda x: len(str(x).split()))

    top_senders = df_filtered['sender'].value_counts().head(top_n).index.tolist()
    df_plot = df_filtered[df_filtered['sender'].isin(top_senders)]

    fig = px.box(
        df_plot,
        x='sender',
        y='word_count',
        color='sender',
        title=f'Message Length Distribution for Top {top_n} Senders',
        labels={'sender': 'User', 'word_count': 'Word Count per Message'},
        points='outliers',
        template='plotly_white',
        color_discrete_sequence=MATERIAL_COLORS,
    )
    fig.update_layout(
        xaxis_title='Sender',
        yaxis_title='Number of Words',
        showlegend=False,
        height=600,
    )
    return fig


# ==================================
# -- Emoji Analysis --
# ==================================

def get_emoji_stats(df: pd.DataFrame, person: str = 'all') -> pd.DataFrame:
    """
    Count every emoji used in the chat.
    Returns a DataFrame with columns: emoji, count, percentage.
    """
    if person != 'all':
        df = df[df['sender'] == person]

    all_emojis = []
    for message in df['message']:
        if pd.isna(message):
            continue
        all_emojis.extend([item['emoji'] for item in emoji.emoji_list(message)])

    if not all_emojis:
        return pd.DataFrame(columns=['emoji', 'count', 'percentage'])

    emoji_counts = Counter(all_emojis)
    total_count = sum(emoji_counts.values())

    emoji_df = pd.DataFrame(emoji_counts.items(), columns=['emoji', 'count'])
    emoji_df['percentage'] = (emoji_df['count'] / total_count) * 100
    return emoji_df.sort_values(by='count', ascending=False).reset_index(drop=True)


def plot_top_emojis_pie(emoji_df, top_n=10):
    """Pie chart of the top N most used emojis."""
    top_emojis = emoji_df.head(top_n).copy()

    fig = px.pie(
        top_emojis,
        values='count',
        names='emoji',
        title=f'Top {top_n} Emojis by Percentage Share',
        hole=0.3,
        template='plotly_white',
        color_discrete_sequence=MATERIAL_COLORS,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def plot_hourly_activity(df, person='all'):
    """
    Stacked bar chart of message count per hour of day (0-23),
    with each sender shown as a separate color segment.
    Returns a Plotly figure, or None if there is no data.
    """
    df_clean = df[df['sender'] != 'System'].copy()
    if person != 'all':
        df_clean = df_clean[df_clean['sender'] == person]

    if df_clean.empty:
        print(f'No data found for: {person}')
        return None

    df_clean['hours'] = df_clean['hours'].astype(int)
    hourly_data = df_clean.groupby(['hours', 'sender']).size().reset_index(name='message_count')

    fig = px.bar(
        hourly_data,
        x='hours',
        y='message_count',
        color='sender',
        title=f'Hourly Message Distribution: {_person_label(person)}',
        labels={'hours': 'Hour of Day (24hr format)', 'message_count': 'Number of Messages', 'sender': 'Member'},
        template='plotly_white',
        text_auto=True,
        color_discrete_sequence=MATERIAL_COLORS,
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        barmode='stack',
        height=600,
        hovermode='x unified',
    )
    return fig


def get_day_period(hour):
    """Map an hour (0-23) to a named period of the day."""
    if 4 <= hour < 8:
        return 'Early Morning'
    elif 8 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Late Night'


def plot_weekly_schedule_heatmap(df, person='all'):
    """
    Modern-styled heatmap of activity: day of week (x) vs period of day (y).
    Returns a Plotly figure, or None if there is no data.
    """
    df_clean = df[df['sender'] != 'System'].copy()
    if person != 'all':
        df_clean = df_clean[df_clean['sender'] == person]

    if df_clean.empty:
        print(f'No data for {person}')
        return None

    df_clean['period'] = df_clean['hours'].apply(get_day_period)

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    period_order = ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Late Night']

    grid_df = df_clean.groupby(['day_name', 'period']).size().reset_index(name='count')
    pivot = grid_df.pivot(index='period', columns='day_name', values='count').fillna(0)
    pivot = pivot.reindex(index=period_order, columns=day_order)

    # Dark background with a Material amber-to-red "heat" scale.
    custom_colors = [
        [0.0, '#2b2b3d'],   # Inactive (dark slate)
        [0.15, '#3949AB'],  # Indigo 600 (low)
        [0.45, '#00ACC1'],  # Cyan 600 (medium-low)
        [0.7, '#FB8C00'],   # Orange 600 (medium-high)
        [1.0, '#E53935'],   # Red 600 (peak)
    ]

    fig = px.imshow(
        pivot,
        labels=dict(x='Day of Week', y='Time of Day', color='Messages'),
        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        y=period_order,
        color_continuous_scale=custom_colors,
        aspect='auto',
        text_auto=True,
    )

    fig.update_layout(
        title=dict(
            text=f'Weekly Activity Schedule: {_person_label(person)}',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center',
            y=0.97,          # pin title near the very top of the figure...
            yanchor='top',   # ...anchored from its top edge, not centered on it
        ),
        paper_bgcolor='#1a1c2c',
        plot_bgcolor='#1a1c2c',
        font=dict(color='white'),
        xaxis=dict(side='top', showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=150, r=50, t=140, b=50),  # more top margin so day labels
                                                 # (pushed up by side='top') sit
                                                 # below the title, not under it
        height=650,  # slightly taller to absorb the extra top margin
    )
    fig.update_traces(xgap=5, ygap=5)
    return fig