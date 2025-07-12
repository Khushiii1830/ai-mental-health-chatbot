# app.py

import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime
import os
import plotly.express as px

# Title
st.title("💙 AI Mental Health Companion")

# Load emotion detection pipeline
emotion_detector = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None
)

# User input: Name
name = st.text_input("Enter your name:")

if name:
    st.write(f"Welcome, {name}! Share how you feel today.")

    # User input: Feelings
    user_feeling = st.text_area("Describe your day or feelings:")

    if st.button("Analyze My Mood"):
        if user_feeling.strip() == "":
            st.warning("Please enter something to analyze.")
        else:
            # Emotion prediction
            predictions = emotion_detector(user_feeling)[0]
            emotions = {item['label']: item['score'] for item in predictions}

            # Pie chart of emotions
            fig_pie = px.pie(
                names=list(emotions.keys()),
                values=list(emotions.values()),
                title="Your Emotion Distribution"
            )
            st.plotly_chart(fig_pie)

            # Top emotion
            top_emotion = max(emotions, key=emotions.get)

            # Load or create mood log
            if os.path.exists('mood_log.csv'):
                mood_data = pd.read_csv('mood_log.csv')
            else:
                mood_data = pd.DataFrame(columns=["Date", "Emotion"])

            # Personalized messages
            st.subheader("💡 Reflection and Suggestion")
            if top_emotion == "joy":
                st.success(f"😊 {name}, you seem joyful today! Keep embracing it.")
                st.info("Tip: Share your happiness with someone you care about.")
            elif top_emotion == "sadness":
                st.error(f"😔 {name}, it seems you're feeling sad.")
                st.info("Tip: Consider journaling or talking to a friend.")
            elif top_emotion == "anger":
                st.warning(f"😡 {name}, you might be feeling angry.")
                st.info("Tip: Try deep breathing or a short walk.")
            elif top_emotion == "fear":
                st.warning(f"😨 {name}, you're feeling fearful.")
                st.info("Tip: Try grounding exercises or talking about your fears.")
            elif top_emotion == "love":
                st.success(f"❤️ {name}, you're experiencing love today.")
                st.info("Tip: Let someone know you appreciate them.")
            elif top_emotion == "surprise":
                st.info(f"😲 {name}, you're surprised today.")
                st.info("Tip: Embrace the unexpected with curiosity.")
            else:
                st.info(f"🙂 {name}, your mood appears neutral today.")
                st.info("Tip: Enjoy a mindful moment or listen to music.")

            # Update mood log
            new_entry = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Emotion": top_emotion
            }
            mood_data = pd.concat([mood_data, pd.DataFrame([new_entry])], ignore_index=True)
            mood_data.to_csv('mood_log.csv', index=False)

            # Show mood history table
            st.subheader("📋 Your Mood History")
            st.dataframe(mood_data.style.highlight_max(axis=0, color='lightgreen'))

            # Emotion count bar chart
            st.subheader("📊 Mood Frequency")
            emotion_counts = mood_data['Emotion'].value_counts().reset_index()
            emotion_counts.columns = ['Emotion', 'Count']
            fig_bar = px.bar(
                emotion_counts,
                x='Emotion',
                y='Count',
                color='Emotion',
                text='Count',
                title="Recorded Emotion Counts"
            )
            st.plotly_chart(fig_bar)

            # Timeline of moods
            st.subheader("⏳ Mood Over Time")
            mood_data['Date'] = pd.to_datetime(mood_data['Date'])
            fig_line = px.line(
                mood_data,
                x='Date',
                y='Emotion',
                title="Mood Timeline",
                markers=True
            )
            st.plotly_chart(fig_line)

            # Insights summary
            st.subheader("🔍 Summary Insights")
            most_common = mood_data['Emotion'].mode()[0]
            total_entries = len(mood_data)
            last_emotion = mood_data.iloc[-1]['Emotion']

            st.write(f"✅ **Total days tracked:** {total_entries}")
            st.write(f"✅ **Most frequent emotion:** {most_common}")
            st.write(f"✅ **Last recorded emotion:** {last_emotion}")
