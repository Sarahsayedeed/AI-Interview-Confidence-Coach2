import streamlit as st
import cv2
import pandas as pd
import json
import os
from cv_tracker import record_and_analyze
from event_detector import detect_events
from llm_feedback import analyze_events

st.set_page_config(page_title="AI Interview Confidence Coach", layout="wide")

# Disclaimer
st.info("⚠️ **Disclaimer**: This tool analyzes body-language indicators as a practice aid. It is not a psychological diagnosis or a definitive judgment of your performance.")

st.title("AI Interview Confidence Coach")

# Language Toggle
language = st.radio("Select Language / اختر اللغة", ["ar", "en"], horizontal=True)

if 'recorded' not in st.session_state:
    st.session_state.recorded = False
if 'events' not in st.session_state:
    st.session_state.events = None
if 'llm_notes' not in st.session_state:
    st.session_state.llm_notes = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = "recorded_video.mp4"
if 'selected_time' not in st.session_state:
    st.session_state.selected_time = 0

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Practice")
    duration = st.slider("Select practice duration (seconds)", 10, 60, 30)
    
    if st.button("Start Recording", type="primary"):
        st.session_state.recorded = False
        st.session_state.events = None
        st.session_state.llm_notes = None
        st.session_state.selected_time = 0
        
        st_placeholder = st.empty()
        with st.spinner("Recording and analyzing your body language..."):
            features_df = record_and_analyze(duration_sec=duration, output_filename=st.session_state.video_path, st_placeholder=st_placeholder)
            
        st_placeholder.empty() # clear camera feed
        
        with st.spinner("Processing tension events..."):
            events = detect_events(features_df)
            st.session_state.events = events
            
        with st.spinner("Generating AI coaching notes..."):
            notes = analyze_events(events, language=language)
            st.session_state.llm_notes = notes
            st.session_state.recorded = True
            
    if st.session_state.recorded and os.path.exists(st.session_state.video_path):
        st.subheader("Review Recording")
        # Streamlit st.video supports start_time
        video_file = open(st.session_state.video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, start_time=int(st.session_state.selected_time))

with col2:
    if st.session_state.recorded:
        st.subheader("AI Feedback")
        
        if st.session_state.llm_notes:
            st.markdown(st.session_state.llm_notes)
            
        if isinstance(st.session_state.events, list) and len(st.session_state.events) > 0:
            st.subheader("Timeline Events")
            st.write("Click an event to jump to that moment in the video:")
            for i, ev in enumerate(st.session_state.events):
                event_name = ev.get("event", "Event").replace("_", " ").title()
                time_range = ev.get("timestamp_range", ["00:00"])[0]
                # convert "mm:ss" to seconds
                try:
                    m, s = map(int, time_range.split(":"))
                    sec = m * 60 + s
                except:
                    sec = 0
                
                if st.button(f"🔍 {event_name} at {time_range}", key=f"btn_{i}"):
                    st.session_state.selected_time = sec
                    st.rerun()
        else:
            st.success("Stable baseline detected. Great job!")
            
    else:
        st.write("Click 'Start Recording' to begin your practice session.")
