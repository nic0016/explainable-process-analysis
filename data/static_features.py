import pandas as pd
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


def create_eventlog(xes_path):
    return xes_importer.apply(xes_path)


def extract_static_attributes(event_log, attribute_names=None):
    default_candidates = ['org:resource', 'org:role', 'concept:name']
    candidates = list(attribute_names) if attribute_names else default_candidates

    trace_count = len(event_log)
    present_counts = {attr: 0 for attr in candidates}
    for trace in event_log:
        attrs_in_trace = set()
        for event in trace:
            for attr in candidates:
                if attr in event:
                    attrs_in_trace.add(attr)
        for attr in attrs_in_trace:
            present_counts[attr] += 1

    static_attributes = [attr for attr in candidates if present_counts.get(attr, 0) == trace_count]
    if not static_attributes:
        static_attributes = candidates

    attribute_values = {attr: set() for attr in static_attributes}
    for trace in event_log:
        for event in trace:
            for attr in static_attributes:
                if attr in event:
                    attribute_values[attr].add(str(event[attr]))

    for attr in static_attributes:
        attribute_values[attr] = sorted(list(attribute_values[attr]))

    return attribute_values, static_attributes


def create_label_encoders(attribute_values, static_attributes):
    encoders = {}
    for attr in static_attributes:
        encoder = LabelEncoder()
        encoder.fit(attribute_values[attr])
        encoders[attr] = encoder
    return encoders


def encode_event_features(event, encoders, static_attributes):
    features = []
    for attr in static_attributes:
        if attr in event:
            encoded_value = encoders[attr].transform([str(event[attr])])[0]
            one_hot = np.zeros(len(encoders[attr].classes_))
            one_hot[encoded_value] = 1
            features.extend(one_hot)
        else:
            features.extend(np.zeros(len(encoders[attr].classes_)))
    return np.array(features)


def trace_duration_seconds(trace):
    timestamps = [event.get("time:timestamp", None) for event in trace if "time:timestamp" in event]
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)).total_seconds()


def build_static_features_dataframe(event_log, attribute_names=None):
    if not event_log:
        return pd.DataFrame(), {}, []
    
    attribute_values, static_attributes = extract_static_attributes(event_log, attribute_names)
    encoders = create_label_encoders(attribute_values, static_attributes)
    
    feature_dim = sum(len(encoders[attr].classes_) for attr in static_attributes)
    max_events = max(len(trace) for trace in event_log)
    
    columns = []
    for event_pos in range(max_events):
        for attr in static_attributes:
            for class_name in encoders[attr].classes_:
                columns.append(f"Event_{event_pos+1}_{attr}_{class_name}")
    columns.append("Total_Duration")
    
    rows = []
    trace_ids = []
    
    for trace_idx, trace in enumerate(event_log):
        trace_id = trace.attributes.get("concept:name", f"trace_{trace_idx}")
        duration = trace_duration_seconds(trace)
        
        row = np.zeros(max_events * feature_dim)
        ordered_events = sorted(trace, key=lambda evt: evt.get("time:timestamp", datetime.min))
        
        for event_idx, event in enumerate(ordered_events):
            if event_idx >= max_events:
                break
            event_features = encode_event_features(event, encoders, static_attributes)
            start_idx = event_idx * feature_dim
            end_idx = start_idx + feature_dim
            row[start_idx:end_idx] = event_features
        
        row = np.append(row, duration)
        rows.append(row)
        trace_ids.append(trace_id)
    
    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe.insert(0, "Trace_ID", trace_ids)
    
    return dataframe, encoders, static_attributes


def get_feature_info(encoders, static_attributes):
    info = {
        'static_attributes': static_attributes,
        'encoders': encoders,
        'feature_dimensions': {}
    }
    for attr in static_attributes:
        info['feature_dimensions'][attr] = len(encoders[attr].classes_)
    info['total_feature_dim'] = sum(info['feature_dimensions'].values())
    return info


def head_with_static_features(dataframe, n=5, events=2):
    feature_columns = [col for col in dataframe.columns if col.startswith("Event_")]
    
    selected_columns = ["Trace_ID"]
    for event_pos in range(1, events + 1):
        event_cols = [col for col in feature_columns if col.startswith(f"Event_{event_pos}_")]
        selected_columns.extend(event_cols[:5])
    selected_columns.append("Total_Duration")
    
    existing_columns = [col for col in selected_columns if col in dataframe.columns]
    return dataframe[existing_columns].head(n)
