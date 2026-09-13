from datetime import datetime
import streamlit as st
from dataset_preprocess import build_chat_dataframe_from_string
import utils
import config as cfg

from gemini_agent.gemini_halper import GeminiHelper
import gemini_agent.gemini_prompt as prompts
import gemini_agent.gemini_utils as gutils

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="WhatsApp Analyzer", layout="wide")
st.title("📱 WhatsApp Chat Dashboard")


# ----------------------------
# Cached helpers
# ----------------------------
# Caching by file content + params means widget interactions (sliders,
# selectboxes) that DON'T change the underlying data won't re-run heavy
# pandas/plot work on every Streamlit rerun. This is the #1 fix for the
# "freezes / errors on any parameter change" problem.

@st.cache_resource(show_spinner=False)
def get_gemini_agent():
    # cache_resource -> created once per session, not re-instantiated
    # every rerun (avoids re-auth / client re-init overhead).
    try:
        return GeminiHelper()
    except ValueError:
        return None


@st.cache_data(show_spinner="Parsing chat file...")
def load_chat_dataframe(file_bytes: bytes):
    content = file_bytes.decode("utf-8")
    return build_chat_dataframe_from_string(content)


@st.cache_data(show_spinner=False)
def cached_most_active_members(df, top_n):
    return utils.plot_most_active_members(df, top_n=top_n)


@st.cache_data(show_spinner=False)
def cached_chat_starters_and_enders(df, top_n):
    return utils.get_chat_starters_and_enders(df, top_n=top_n)


@st.cache_data(show_spinner=False)
def cached_top_words(df, stopword_paths, person):
    return utils.get_top_words(df, stopword_paths, person=person)


@st.cache_data(show_spinner="Building word clouds...")
def cached_word_cloud(df, stopword_paths, person, max_words):
    return utils.make_word_cloud(df, stopword_paths, person=person, max_words=max_words)


@st.cache_data(show_spinner=False)
def cached_emoji_stats(df, person):
    return utils.get_emoji_stats(df, person=person)


def safe_get_insight(agent, data_summary: str, prompt: str, cache_key: str):
    """
    Calls Gemini only once per unique cache_key (stored in session_state),
    and never lets a Gemini failure crash the whole app.
    """
    store = st.session_state.setdefault("_gemini_cache", {})

    if cache_key in store:
        return store[cache_key]

    try:
        insight = agent.get_insight(data_summary, prompt)
    except Exception as e:
        st.session_state["_gemini_last_error"] = str(e)
        insight = None

    store[cache_key] = insight
    return insight


# ----------------------------
# Session State
# ----------------------------
if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False

# ----------------------------
# Upload Chat
# ----------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload WhatsApp Chat (.txt)",
    type=["txt"]
)

if uploaded_file is not None:

    try:
        massages_df = load_chat_dataframe(uploaded_file.getvalue())
    except Exception as e:
        st.error(f"Could not parse the uploaded chat file: {e}")
        st.stop()

    if massages_df is None or massages_df.empty:
        st.warning("No messages could be parsed from this file.")
        st.stop()

    # Members
    member_count, member_list = utils.get_total_members(massages_df)
    user_list = ["Overall"] + sorted(member_list)

    selected_user = st.sidebar.selectbox(
        "Select Member for Analysis",
        user_list
    )

    is_ai_summary_on = st.sidebar.toggle(
        "AI Summary",
        value=False,
        key="ai_summary_toggle"
    )

    if st.sidebar.button("Show Analysis"):
        st.session_state.show_analysis = True

    # ----------------------------
    # Analysis
    # ----------------------------
    if st.session_state.show_analysis:

        selected_user = (
            "all"
            if selected_user == "Overall"
            else selected_user
        )

        st.header(
            f"Analysis for: {'All Members' if selected_user == 'all' else selected_user}"
        )

        # ==========================
        # Statistics
        # ==========================
        st.subheader("Statistics", divider="gray")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Messages", utils.get_total_message_count(massages_df, selected_user))
        col2.metric("Total Words", utils.get_total_word_count(massages_df, selected_user))
        col3.metric("Total Media", utils.get_total_media_count(massages_df, selected_user))
        col4.metric("Total Links", utils.get_total_link_count(massages_df, selected_user)[0])
        col5.metric("Total VCFs", utils.get_total_vcf_count(massages_df, selected_user)[0])

        # ==========================
        # Date Range
        # ==========================
        st.subheader("Chat Date Range", divider="gray")

        col1, col2 = st.columns(2)

        start_date = massages_df["date_formatted"].min().strftime("%d-%m-%Y")
        start_time = massages_df["time_formatted"].min().strftime("%I:%M %p")

        end_date = massages_df["date_formatted"].max().strftime("%d-%m-%Y")
        end_time = massages_df["time_formatted"].max().strftime("%I:%M %p")

        col1.metric("Chat Started On", f"{start_date} at {start_time}")
        col2.metric("Chat Ended On", f"{end_date} at {end_time}")

        # ==========================
        # Activity Analysis
        # ==========================
        if selected_user == "all":

            st.subheader("Activity Analysis", divider="gray")

            if member_count > 1:
                slider_val = st.slider(
                    "Select number of top active members",
                    min_value=1,
                    max_value=member_count,
                    value=min(20, member_count),
                    step=1,
                    key="activity_slider"
                )
            else:
                slider_val = 1

            # Cached: moving this slider only recomputes when slider_val changes,
            # not on every unrelated widget interaction elsewhere on the page.
            user_counts, fig_bar, fig_pie = cached_most_active_members(massages_df, slider_val)

            st.write(f"#### Top {slider_val} Active Members")
            st.plotly_chart(fig_bar, width='stretch')

            st.write(f"#### Top {slider_val} Members Distribution")
            st.plotly_chart(fig_pie, width='stretch')

            # ==========================
            # Chat Starters and Enders
            # ==========================
            st.subheader("Chat Starters and Enders", divider="gray")

            if member_count > 1:
                slider2_val = st.slider(
                    "Select number of top active members",
                    min_value=1,
                    max_value=member_count,
                    value=min(10, member_count),
                    step=1,
                    key="starter_ender_slider"
                )
            else:
                slider2_val = 1

            starters_df, enders_df, fig_starters, fig_enders = cached_chat_starters_and_enders(
                massages_df, slider2_val
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Chat Starters")
                st.plotly_chart(fig_starters, width='stretch')

            with col2:
                st.write("#### Chat Enders")
                st.plotly_chart(fig_enders, width='stretch')

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Chat Starters Data")
                st.dataframe(starters_df, width='stretch')

            with col2:
                st.write("#### Chat Enders Data")
                st.dataframe(enders_df, width='stretch')

            # ==========================
            # Sunburst Chart
            # ==========================
            st.subheader("Sunburst Chart (Year > Month > Weekday > Member)", divider="gray")
            sunburst_fig = utils.plot_message_sunburst(massages_df)
            st.plotly_chart(sunburst_fig, width='stretch')

        # ==========================
        # Timeline Analysis
        # ==========================
        st.subheader("Timeline Analysis", divider="gray")
        timeline_daily, timeline_monthly = utils.get_message_timeline(massages_df, selected_user)

        st.write("#### Daily Activity")
        st.plotly_chart(utils.plot_daily_message_timeline(timeline_daily), width='stretch')

        st.write("#### Monthly Activity")
        st.plotly_chart(utils.plot_monthly_message_timeline(timeline_monthly), width='stretch')

        # ==========================
        # Most Active Days and Months
        # ==========================
        st.subheader("Most Active Days and Months", divider="gray")
        fig_most_active_day, fig_most_active_month = utils.plot_most_active_day_and_month(
            massages_df, selected_user
        )

        st.write("#### Most Active Days")
        st.plotly_chart(fig_most_active_day, width='stretch')

        st.write("#### Most Active Months")
        st.plotly_chart(fig_most_active_month, width='stretch')

        # ======================
        # Heatmap
        # ======================
        st.subheader("Heatmap", divider="gray")

        st.write('### Daily Message Activity')
        DATE_FORMAT = "%d-%m-%Y"
        available_dates = utils.get_date_list(start_date, end_date, DATE_FORMAT)

        if not available_dates:
            st.warning("No valid dates found in this chat.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            selected_start_date = st.selectbox(
                "Select Start Date",
                available_dates,
                index=0,
                key="heatmap_start_date"
            )

        start_idx = available_dates.index(selected_start_date)

        with col2:
            selected_end_date = st.selectbox(
                "Select End Date",
                available_dates[start_idx:],
                index=len(available_dates[start_idx:]) - 1,
                key="heatmap_end_date"
            )

        start_dt = datetime.strptime(selected_start_date, DATE_FORMAT)
        end_dt = datetime.strptime(selected_end_date, DATE_FORMAT)

        if start_dt > end_dt:
            st.warning("Start date cannot be after end date.")
            selected_end_date = selected_start_date

        fig_daily_activity_heatmap = utils.plot_daily_activity_heatmap(
            massages_df,
            selected_user,
            start_date=selected_start_date,
            end_date=selected_end_date,
            date_format=DATE_FORMAT
        )
        st.plotly_chart(fig_daily_activity_heatmap, width='stretch')

        st.write('### Weekday vs Month Activity')
        weekday_month_heatmap = utils.plot_weekday_month_heatmap(massages_df, selected_user)
        st.plotly_chart(weekday_month_heatmap, width='stretch')

        st.write('### Day of Month vs Month')
        day_of_month_heatmap = utils.plot_day_of_month_heatmap(massages_df, selected_user)
        st.plotly_chart(day_of_month_heatmap, width='stretch')

        # ==========================
        # Message Analysis
        # ==========================
        st.subheader("Message Analysis", divider="gray")

        top_words_df = cached_top_words(massages_df, cfg.STOPWORDS_FILE_PATHS, selected_user)

        if top_words_df is None or top_words_df.empty:
            st.info("No word data available for this selection.")
        else:
            top_words_slider = st.slider(
                "Select number of top words to display",
                min_value=1,
                max_value=min(100, len(top_words_df)),
                value=min(10, len(top_words_df)),
                step=1,
                key="top_words_slider"
            )

            st.write(f"### Top {top_words_slider} Most Frequent Words")
            top_words_fig = utils.plot_top_words_for_person(top_words_df, top_n=top_words_slider)
            st.plotly_chart(top_words_fig, width='stretch')

            # Words Cloud
            st.write("### Word Cloud")

            no_words_slider = st.slider(
                "Select maximum number of words to display word cloud",
                min_value=1,
                max_value=min(400, len(top_words_df)),
                value=min(160, len(top_words_df)),
                step=1,
                key="no_words_limit_slider"
            )

            img_bengali, freq_bengali, img_english, freq_english = cached_word_cloud(
                massages_df,
                cfg.STOPWORDS_FILE_PATHS,
                selected_user,
                no_words_slider,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Top Bengali Word Cloud")
                if img_bengali is not None:
                    st.image(img_bengali, caption="Bengali Word Cloud")
                else:
                    st.info("No Bengali words were available to build a word cloud.")

            with col2:
                st.write("#### Top English Word Cloud")
                if img_english is not None:
                    st.image(img_english, caption="English Word Cloud")
                else:
                    st.info("No English words were available to build a word cloud.")

            max_freq_words = max(len(freq_bengali), len(freq_english))
            if max_freq_words > 0:
                slider_max = max(1, min(40, max_freq_words))
                no_freq_words_slider = st.slider(
                    "Select number of words to display word cloud",
                    min_value=1,
                    max_value=slider_max,
                    value=min(10, slider_max),
                    step=1,
                    key="no_freq_words_slider",
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"#### Top {no_freq_words_slider} Bengali Words Frequency")
                    fig_bengali = utils.plot_word_frequency_bar(freq_bengali, top_n=no_freq_words_slider)
                    if fig_bengali is not None:
                        st.plotly_chart(fig_bengali, width='stretch')
                    else:
                        st.info("No Bengali word frequency data available.")

                with col2:
                    st.write(f"#### Top {no_freq_words_slider} English Words Frequency")
                    fig_english = utils.plot_word_frequency_bar(freq_english, top_n=no_freq_words_slider)
                    if fig_english is not None:
                        st.plotly_chart(fig_english, width='stretch')
                    else:
                        st.info("No English word frequency data available.")

        # =========================
        # Message Length Distribution
        # =========================
        top_n_msg_distribution_slider = st.slider(
            "Select number of top members to display message length distribution",
            min_value=1,
            max_value=max(10, member_count),
            value=10,
            step=1,
            key="top_n_msg_distribution_slider"
        )
        st.write(f"### Message Length Distribution for Top {top_n_msg_distribution_slider} Senders")
        fig_msg_length = utils.plot_message_length_distribution(massages_df, top_n=top_n_msg_distribution_slider)
        st.plotly_chart(fig_msg_length, width='stretch')

        # =========================
        # Emoji Analysis
        # =========================
        st.subheader("Emoji Analysis", divider="gray")

        top_emojis_df = cached_emoji_stats(massages_df, selected_user)

        if top_emojis_df is None or top_emojis_df.empty:
            st.info("No emoji data available for this selection.")
        else:
            if len(top_emojis_df) > 1:
                top_n_emoji_slider = st.slider(
                    "Select number of top emojis to display",
                    min_value=1,
                    max_value=min(100, len(top_emojis_df)),
                    value=min(10, len(top_emojis_df)),
                    step=1,
                    key="top_n_emoji_slider"
                )
            else:
                top_n_emoji_slider = 1

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"### Top {top_n_emoji_slider} Emojis Distribution")
                fig_emoji = utils.plot_top_emojis_pie(top_emojis_df, top_n=top_n_emoji_slider)
                st.plotly_chart(fig_emoji, width='stretch')
            with col2:
                st.write("### Top Emojis")
                st.dataframe(top_emojis_df, width='stretch')

        st.subheader("Hourly and Time-of-Day Activity", divider="gray")
        st.write("### Hourly Activity")
        fig_hourly_activity = utils.plot_hourly_activity(massages_df, selected_user)
        st.plotly_chart(fig_hourly_activity, width='stretch')

        st.write("### Time-of-Day Activity")
        fig_weekly_schedule_heatmap = utils.plot_weekly_schedule_heatmap(massages_df, selected_user)
        st.plotly_chart(fig_weekly_schedule_heatmap, width='stretch')

        # ==========================
        # Last 7 Day Message Summary
        # ==========================
        if is_ai_summary_on:
            st.subheader("Last 7 Day Message Summary Powered by Gemini", divider="gray")

            gemini_agent = get_gemini_agent()
            if gemini_agent is None:
                st.info("AI Summary is unavailable until Gemini API keys are configured.")
                st.stop()

            data_summary = gutils.get_last_n_day_messages(massages_df, last_n_day=7)

            if not data_summary:
                st.info("No messages found in the last 7 days for this selection.")
            else:
                # Cache key ties the insight to the actual inputs that affect it,
                # so it's called once and reused across reruns/slider changes
                # instead of re-firing the Gemini API on every widget interaction.
                cache_key = f"msg_summary::{selected_user}::hash{hash(data_summary)}"

                with st.spinner("Generating 7-day summary with Gemini..."):
                    massages_summary = safe_get_insight(
                        gemini_agent,
                        data_summary,
                        prompts.WEEKLY_SUMMARY_PROMPT,
                        cache_key,
                    )

                if massages_summary is not None:
                    st.markdown(massages_summary)
                else:
                    err = st.session_state.get("_gemini_last_error")
                    st.info("Summary generation failed." + (f" ({err})" if err else " :/"))