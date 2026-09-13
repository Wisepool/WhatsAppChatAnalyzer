import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)


import pandas as pd
from data_clean import get_text_only_messages


def get_last_n_day_messages(df: pd.DataFrame, last_n_day: int = 7) -> str:
    # 1. Remove system/media and empty messages
    df = get_text_only_messages(df)
    
    # 2. Ensure date column is in datetime format to perform calculations
    df['date_dt'] = pd.to_datetime(df['date_formatted'])
    
    # 3. Find the last date available in the chat log
    latest_date = df['date_dt'].max()
    start_date = latest_date - pd.Timedelta(days=last_n_day)
    
    # 4. Filter rows falling within the last N days
    df_filtered = df[df['date_dt'] > start_date].copy()
    
    if df_filtered.empty:
        return "No messages found in the specified time frame."

    # 5. Token Optimization: Create a clean short timestamp column
    df_filtered['timestamp'] = (
        df_filtered['date_formatted'].astype(str) + ' ' + 
        df_filtered['hours'].astype(str).str.zfill(2) + ':' + 
        df_filtered['minutes'].astype(str).str.zfill(2)
    )
    
    # 6. Group consecutive rows sent by the exact same user in the same minute
    group_condition = (
        (df_filtered['sender'] != df_filtered['sender'].shift()) | 
        (df_filtered['timestamp'] != df_filtered['timestamp'].shift())
    ).cumsum()
    
    compressed_df = (
        df_filtered.groupby([group_condition, 'timestamp', 'sender'])['message']
        .apply(lambda messages: ' | '.join(messages.astype(str)))
        .reset_index()
    )
    
    # 7. Generate the highly compact text block for Gemini API
    compressed_text_lines = []
    for _, row in compressed_df.iterrows():
        compressed_text_lines.append(f"[{row['timestamp']}] {row['sender']}: {row['message']}")
        
    return "\n".join(compressed_text_lines)
