"""
Static Event Attribute Feature Extraction for XES Event Logs

Transforms categorical event attributes (org:resource, org:role, concept:name) 
into high-dimensional feature vectors using one-hot encoding.
Feature Space: R^(K×T) where K = max trace length, T = total attribute dimensions
"""

import pandas as pd
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


def create_eventlog(xes_path):
    """Load XES event log using PM4Py importer."""
    return xes_importer.apply(xes_path)


def extract_static_attributes(event_log, attribute_names=None):
    """
    Extract unique values for selected static attributes.

    Parameters:
        attribute_names: list[str] | None
            If None, uses default attributes ['org:resource','org:role','concept:name']
            and filters to those present in every trace.
    """
    print("Extracting static event attributes...")

    default_candidates = ['org:resource', 'org:role', 'concept:name']
    candidates = list(attribute_names) if attribute_names else default_candidates

    # Find attributes present in every trace
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
        # Fallback: use all candidates if none present in all traces
        static_attributes = candidates

    attribute_values = {attr: set() for attr in static_attributes}
    for trace in event_log:
        for event in trace:
            for attr in static_attributes:
                if attr in event:
                    attribute_values[attr].add(str(event[attr]))

    for attr in static_attributes:
        attribute_values[attr] = sorted(list(attribute_values[attr]))

    print("Selected static attributes (present in all traces if possible):")
    for attr, values in attribute_values.items():
        print(f"  {attr}: {len(values)} unique values")

    return attribute_values, static_attributes


def create_label_encoders(attribute_values, static_attributes):
    """Create LabelEncoder objects for categorical attributes."""
    encoders = {}
    
    for attr in static_attributes:
        encoder = LabelEncoder()
        encoder.fit(attribute_values[attr])
        encoders[attr] = encoder
    
    return encoders


def encode_event_features(event, encoders, static_attributes):
    """Transform event to one-hot encoded feature vector."""
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
    """Calculate trace duration in seconds from min/max timestamps."""
    timestamps = [event.get("time:timestamp", None) for event in trace if "time:timestamp" in event]
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)).total_seconds()


def build_static_features_dataframe(event_log, attribute_names=None):
    """Build DataFrame with static event features."""
    if not event_log:
        return pd.DataFrame(), {}, []
    
    print("=== Building DataFrame with static event features ===")
    
    # Extract static attributes
    attribute_values, static_attributes = extract_static_attributes(event_log, attribute_names)
    
    # Create Label Encoders
    encoders = create_label_encoders(attribute_values, static_attributes)
    
    # Calculate feature dimensions
    feature_dim = sum(len(encoders[attr].classes_) for attr in static_attributes)
    max_events = max(len(trace) for trace in event_log)
    
    print(f"Feature dimensions:")
    print(f"  Static attributes: {len(static_attributes)}")
    print(f"  Feature dimension per event: {feature_dim}")
    print(f"  Max events per trace: {max_events}")
    print(f"  Total features per trace: {max_events * feature_dim}")
    
    # Create column names
    columns = []
    for event_pos in range(max_events):
        for attr in static_attributes:
            for class_name in encoders[attr].classes_:
                columns.append(f"Event_{event_pos+1}_{attr}_{class_name}")
    columns.append("Total_Duration")
    
    # Build data matrix
    rows = []
    trace_ids = []
    
    print("Processing traces...")
    for trace_idx, trace in enumerate(event_log):
        trace_id = trace.attributes.get("concept:name", f"trace_{trace_idx}")
        duration = trace_duration_seconds(trace)
        
        # Initialize row with zeros
        row = np.zeros(max_events * feature_dim)
        
        # Sort events chronologically
        ordered_events = sorted(trace, key=lambda evt: evt.get("time:timestamp", datetime.min))
        
        # Encode events to features
        for event_idx, event in enumerate(ordered_events):
            if event_idx >= max_events:
                break
            
            # Extract features for this event
            event_features = encode_event_features(event, encoders, static_attributes)
            
            # Place features in row
            start_idx = event_idx * feature_dim
            end_idx = start_idx + feature_dim
            row[start_idx:end_idx] = event_features
        
        # Add duration
        row = np.append(row, duration)
        rows.append(row)
        trace_ids.append(trace_id)
        
        if (trace_idx + 1) % 1000 == 0:
            print(f"  {trace_idx + 1}/{len(event_log)} traces processed...")
    
    # Create DataFrame
    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe.insert(0, "Trace_ID", trace_ids)
    
    print(f"✓ DataFrame created: {dataframe.shape}")
    
    return dataframe, encoders, static_attributes


def get_feature_info(encoders, static_attributes):
    """Return information about features."""
    info = {
        'static_attributes': static_attributes,
        'encoders': encoders,
        'feature_dimensions': {}
    }
    
    for attr in static_attributes:
        info['feature_dimensions'][attr] = len(encoders[attr].classes_)
    
    total_dim = sum(info['feature_dimensions'].values())
    info['total_feature_dim'] = total_dim
    
    return info


def head_with_static_features(dataframe, n=5, events=2):
    """Display first n traces with first events event positions."""
    feature_columns = [col for col in dataframe.columns if col.startswith("Event_")]
    
    # Find columns for first 'events' event positions
    selected_columns = ["Trace_ID"]
    for event_pos in range(1, events + 1):
        event_cols = [col for col in feature_columns if col.startswith(f"Event_{event_pos}_")]
        selected_columns.extend(event_cols[:5])  # First 5 features per event for overview
    
    selected_columns.append("Total_Duration")
    
    # Filter only existing columns
    existing_columns = [col for col in selected_columns if col in dataframe.columns]
    
    return dataframe[existing_columns].head(n)
