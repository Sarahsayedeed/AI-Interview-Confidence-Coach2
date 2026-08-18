import pandas as pd
import numpy as np

MIN_EVENT_DURATION_SECONDS = 0.5

def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def extract_runs(series, timestamps, event_name):
    events = []
    # Find consecutive True values
    run_starts = series[series & ~series.shift(1, fill_value=False)].index
    run_ends = series[series & ~series.shift(-1, fill_value=False)].index
    
    for start_idx, end_idx in zip(run_starts, run_ends):
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx]
        duration = end_time - start_time
        
        if duration >= MIN_EVENT_DURATION_SECONDS:
            event = {
                "event": event_name,
                "timestamp_range": [format_timestamp(start_time), format_timestamp(end_time)],
                "duration_seconds": round(duration, 1)
            }
            events.append(event)
    return events

def detect_events(features_df):
    if features_df.empty:
        return {"events": [], "baseline_stable": True}
        
    events = []
    
    cols_to_map = {
        "is_looking_away": "gaze_deviation",
        "is_touching_face": "hand_touched_face",
        "is_slouching": "posture_slump"
    }
    
    for col, event_name in cols_to_map.items():
        if col in features_df.columns:
            col_events = extract_runs(features_df[col], features_df.timestamp, event_name)
            events.extend(col_events)
            
    if not events:
        return {"events": [], "baseline_stable": True}
        
    return events
