"""
XES Event Log Loading and Encoding

Transforms XES event logs into ML-ready formats:
- 3D tensors [N, K, M] for sequence models (CNN, ResNet, TCN, Transformer)
- Flattened DataFrames for tabular models (XGBoost)
- Split-aware normalization to prevent data leakage
"""

import pandas as pd
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import zscore


def create_eventlog(xes_path):
    """Load XES event log using PM4Py importer."""
    return xes_importer.apply(xes_path)


def build_event_reference(event_log):
    """Extract unique event names from event log."""
    events = set()
    for trace in event_log:
        for event in trace:
            name = event.get("concept:name", None)
            if name:
                events.add(name)
    return sorted(events)


def build_event_id_mapping(event_names):
    """Create mapping from event names to IDs."""
    return {name: idx + 1 for idx, name in enumerate(event_names)}


def trace_duration_seconds(trace):
    """Calculate trace duration in seconds from min/max timestamps."""
    timestamps = [event.get("time:timestamp", None) for event in trace if "time:timestamp" in event]
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)).total_seconds()


def normalize_durations_split_aware(durations, train_indices):
    """
    Normalize durations using ONLY training set statistics.
    This prevents data leakage from test set.
    
    Args:
        durations: np.ndarray of all durations
        train_indices: np.ndarray of training indices
        
    Returns:
        normalized_durations: np.ndarray with z-score normalized values
        train_mean: mean computed from training set only
        train_std: std computed from training set only
    """
    durations = np.array(durations)
    train_durations = durations[train_indices]
    
    train_mean = np.mean(train_durations)
    train_std = np.std(train_durations)
    
    if train_std > 0:
        # Apply training statistics to ALL data (train + test)
        normalized_durations = (durations - train_mean) / train_std
    else:
        # Fallback: no normalization if no variance
        normalized_durations = durations
    
    return normalized_durations, train_mean, train_std


def build_dataframe(event_log, train_indices=None):
    """
    Build dataframe with OneHot encoded events and normalized labels.
    Returns 3-dimensional arrays [nb_data, max_seq_length, nb_events] with 0/1 values.
    
    Args:
        event_log: PM4Py event log
        train_indices: Optional array of training indices for split-aware normalization.
                      If None, uses all data for normalization (old behavior).
                      If provided, uses only training data statistics (prevents data leakage).
    
    Returns:
        encoded_sequences: np.ndarray [N, K, M] with OneHot encoded events
        normalized_durations: np.ndarray [N] with z-normalized durations
        trace_ids: list of trace identifiers
        encoder: OneHotEncoder for event names
    """
    if not event_log:
        return pd.DataFrame(), None, None, None
    
    # Get unique event names and create OneHotEncoder
    event_names = build_event_reference(event_log)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    encoder.fit(np.array(event_names).reshape(-1, 1))
    
    max_events = max(len(trace) for trace in event_log)
    nb_events = len(event_names)
    
    print(f"Dataset info:")
    print(f"  Number of traces: {len(event_log)}")
    print(f"  Max sequence length: {max_events}")
    print(f"  Number of unique events: {nb_events}")
    print(f"  Output shape: [{len(event_log)}, {max_events}, {nb_events}]")
    if train_indices is not None:
        print(f"  Using split-aware normalization (train_n={len(train_indices)}) ✓")
    
    # Initialize 3D array for encoded sequences
    encoded_sequences = np.zeros((len(event_log), max_events, nb_events), dtype=np.int32)
    durations = []
    trace_ids = []
    
    for trace_idx, trace in enumerate(event_log):
        trace_id = trace.attributes.get("concept:name", f"trace_{trace_idx}")
        duration = trace_duration_seconds(trace)
        
        # Sort events chronologically
        ordered_events = sorted(trace, key=lambda evt: evt.get("time:timestamp", 0))
        
        # Encode each event in the sequence
        for event_idx, event in enumerate(ordered_events):
            if event_idx >= max_events:
                break
            name = event.get("concept:name", None)
            if name and name in event_names:
                # OneHot encode the event name
                one_hot = encoder.transform([[name]])[0]
                encoded_sequences[trace_idx, event_idx, :] = one_hot
        
        durations.append(duration)
        trace_ids.append(trace_id)
    
    # Normalize durations using z-score (split-aware if train_indices provided)
    durations = np.array(durations)
    
    if train_indices is not None:
        # SPLIT-AWARE: Use ONLY training data for normalization statistics (no data leakage!)
        normalized_durations, train_mean, train_std = normalize_durations_split_aware(
            durations, train_indices
        )
        print(f"Duration normalization (split-aware - no data leakage):")
        print(f"  Train mean: {train_mean:.2f}, Train std: {train_std:.2f}")
        print(f"  Original range: [{np.min(durations):.2f}, {np.max(durations):.2f}]")
        print(f"  Normalized range: [{np.min(normalized_durations):.2f}, {np.max(normalized_durations):.2f}]")
    else:
        # OLD BEHAVIOR: Use all data for normalization (backwards compatible)
        if len(durations) > 1 and np.std(durations) > 0:
            normalized_durations = zscore(durations)
            print(f"Duration normalization (using all data - old behavior):")
            print(f"  Original range: [{np.min(durations):.2f}, {np.max(durations):.2f}]")
            print(f"  Normalized range: [{np.min(normalized_durations):.2f}, {np.max(normalized_durations):.2f}]")
            print(f"  Mean: {np.mean(normalized_durations):.4f}, Std: {np.std(normalized_durations):.4f}")
        else:
            normalized_durations = durations
            print("Warning: Could not normalize durations (insufficient variance)")
    
    return encoded_sequences, normalized_durations, trace_ids, encoder


def head_with_events(encoded_sequences, normalized_durations, trace_ids, encoder, n=5, k=5):
    """Display first n traces with first k events in a readable format."""
    if encoded_sequences is None:
        return "No data available"
    
    event_names = encoder.categories_[0] if encoder else []
    
    print(f"First {n} traces (showing first {k} events):")
    print("=" * 80)
    
    for i in range(min(n, len(trace_ids))):
        print(f"Trace {i+1}: {trace_ids[i]}")
        print(f"  Duration (normalized): {normalized_durations[i]:.4f}")
        print(f"  Events:")
        
        for j in range(min(k, encoded_sequences.shape[1])):
            # Find which event is encoded (should have value 1)
            event_vector = encoded_sequences[i, j, :]
            if np.sum(event_vector) > 0:  # If there's an event at this position
                event_idx = np.argmax(event_vector)
                event_name = event_names[event_idx] if event_idx < len(event_names) else "Unknown"
                print(f"    Position {j+1}: {event_name}")
            else:
                print(f"    Position {j+1}: (empty)")
        print()
    
    print(f"Data shape: {encoded_sequences.shape}")
    print(f"Event encoding: OneHot with {len(event_names)} unique events")
    return None


def convert_to_dataframe(encoded_sequences, normalized_durations, trace_ids, encoder):
    """
    Convert 3D encoded sequences back to DataFrame format for compatibility.
    Creates flattened feature columns for each event position and attribute.
    """
    if encoded_sequences is None:
        return pd.DataFrame()
    
    event_names = encoder.categories_[0] if encoder else []
    nb_traces, max_events, nb_events = encoded_sequences.shape
    
    # Create column names
    columns = []
    for event_pos in range(max_events):
        for event_name in event_names:
            columns.append(f"Event_{event_pos+1}_{event_name}")
    columns.append("Total_Duration_Normalized")
    
    # Flatten the 3D array to 2D
    flattened_data = encoded_sequences.reshape(nb_traces, -1)
    
    # Add normalized durations
    data_with_duration = np.column_stack([flattened_data, normalized_durations])
    
    # Create DataFrame
    dataframe = pd.DataFrame(data_with_duration, columns=columns)
    dataframe.insert(0, "Trace_ID", trace_ids)
    
    return dataframe


def load_encoded_with_static_attributes(xes_path, attribute_names=None):
    """
    Load XES log and create DataFrame combining event-ID encoding with static event attributes.

    Returns:
        dataframe: DataFrame with columns [Trace_ID, Event_*_..., Total_Duration_Normalized, Event_*_org:*_*, ...]
        encoder: OneHotEncoder for event IDs
        static_encoders: Dict[attribute -> LabelEncoder]
        static_attributes: List of used static attributes
    """
    # Load log and create event-ID encoding
    event_log = create_eventlog(xes_path)
    encoded_sequences, normalized_durations, trace_ids, encoder = build_dataframe(event_log)
    df_events = convert_to_dataframe(encoded_sequences, normalized_durations, trace_ids, encoder)

    # Create static feature matrix and merge
    from .static_features import build_static_features_dataframe

    df_static, static_encoders, static_attributes = build_static_features_dataframe(event_log, attribute_names)

    # Remove unnormalized duration from static matrix
    if "Total_Duration" in df_static.columns:
        df_static = df_static.drop(columns=["Total_Duration"])

    # Merge on Trace_ID
    merged = df_events.merge(df_static, on="Trace_ID", how="left").fillna(0)

    return merged, encoder, static_encoders, static_attributes


def build_dataframe_with_static(event_log, train_indices=None, static_attributes=None):
    """
    Build dataframe with OneHot encoded events PLUS static event attributes.
    Returns 3-dimensional arrays [N, K, M+S] where:
    - N = number of traces
    - K = max sequence length
    - M = number of unique event types (one-hot)
    - S = sum of all static attribute dimensions (one-hot per attribute)
    
    Args:
        event_log: PM4Py event log
        train_indices: Optional array of training indices for split-aware normalization.
        static_attributes: List of static attributes to include (default: auto-detect)
    
    Returns:
        encoded_sequences: np.ndarray [N, K, M+S] with event+static features
        normalized_durations: np.ndarray [N] with z-normalized durations
        trace_ids: list of trace identifiers
        encoder: OneHotEncoder for event names
        static_encoders: dict of LabelEncoders for each static attribute
        static_attribute_names: list of used static attributes
    """
    if not event_log:
        return None, None, None, None, {}, []
    
    # 1. Base Event Encoding
    event_names = build_event_reference(event_log)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    encoder.fit(np.array(event_names).reshape(-1, 1))
    
    max_events = max(len(trace) for trace in event_log)
    nb_event_types = len(event_names)
    
    # 2. Identify static attributes
    if static_attributes is None:
        # Auto-detect: Find attributes present in most events
        static_attributes = []
        candidate_attrs = ['org:resource', 'org:role', 'concept:name']
        
        for attr in candidate_attrs:
            count = sum(1 for trace in event_log for event in trace if attr in event)
            total_events = sum(len(trace) for trace in event_log)
            if count / total_events > 0.8:
                static_attributes.append(attr)
    
    print(f"Dataset info (with static features):")
    print(f"  Number of traces: {len(event_log)}")
    print(f"  Max sequence length: {max_events}")
    print(f"  Number of unique event types: {nb_event_types}")
    print(f"  Static attributes: {static_attributes}")
    if train_indices is not None:
        print(f"  Using split-aware normalization (train_n={len(train_indices)}) ✓")
    
    # 3. Extract unique values for static attributes
    static_values = {attr: set() for attr in static_attributes}
    for trace in event_log:
        for event in trace:
            for attr in static_attributes:
                if attr in event:
                    static_values[attr].add(str(event[attr]))
    
    # 4. Create LabelEncoders for static attributes
    static_encoders = {}
    static_dims = {}
    for attr in static_attributes:
        encoder_static = LabelEncoder()
        encoder_static.fit(sorted(static_values[attr]))
        static_encoders[attr] = encoder_static
        static_dims[attr] = len(encoder_static.classes_)
        print(f"    {attr}: {len(encoder_static.classes_)} unique values")
    
    total_static_dim = sum(static_dims.values())
    total_feature_dim = nb_event_types + total_static_dim
    
    print(f"  Total feature dimension per event: {nb_event_types} (events) + {total_static_dim} (static) = {total_feature_dim}")
    print(f"  Output shape: [{len(event_log)}, {max_events}, {total_feature_dim}]")
    
    # 5. Create 3D array [N, K, M+S]
    encoded_sequences = np.zeros((len(event_log), max_events, total_feature_dim), dtype=np.float32)
    trace_ids = []
    durations = []
    
    for trace_idx, trace in enumerate(event_log):
        trace_id = trace.attributes.get("concept:name", f"trace_{trace_idx}")
        trace_ids.append(trace_id)
        
        duration = trace_duration_seconds(trace)
        durations.append(duration)
        
        # Sort events chronologically
        ordered_events = sorted(trace, key=lambda evt: evt.get("time:timestamp", None) or 0)
        
        for event_idx, event in enumerate(ordered_events):
            if event_idx >= max_events:
                break
            
            # Event One-Hot (first M dimensions)
            event_name = event.get("concept:name", "UNKNOWN")
            if event_name in event_names:
                event_onehot = encoder.transform([[event_name]])[0].astype(np.float32)
            else:
                event_onehot = np.zeros(nb_event_types, dtype=np.float32)
            
            # Static Features One-Hot (next S dimensions)
            static_onehot = []
            for attr in static_attributes:
                if attr in event:
                    attr_value = str(event[attr])
                    if attr_value in static_encoders[attr].classes_:
                        encoded_val = static_encoders[attr].transform([attr_value])[0]
                        one_hot = np.zeros(static_dims[attr], dtype=np.float32)
                        one_hot[encoded_val] = 1
                        static_onehot.extend(one_hot)
                    else:
                        static_onehot.extend(np.zeros(static_dims[attr], dtype=np.float32))
                else:
                    # Attribute missing -> zero vector
                    static_onehot.extend(np.zeros(static_dims[attr], dtype=np.float32))
            
            # Concatenate Event + Static Features
            full_features = np.concatenate([event_onehot, np.array(static_onehot, dtype=np.float32)])
            encoded_sequences[trace_idx, event_idx, :] = full_features
    
    # 6. Normalize durations (split-aware if train_indices provided)
    durations = np.array(durations)
    
    if train_indices is not None:
        normalized_durations, train_mean, train_std = normalize_durations_split_aware(
            durations, train_indices
        )
        normalized_durations = normalized_durations.astype(np.float32)
        print(f"Duration normalization (split-aware - no data leakage):")
        print(f"  Train mean: {train_mean:.2f}, Train std: {train_std:.2f}")
        print(f"  Original range: [{durations.min():.2f}, {durations.max():.2f}]")
        print(f"  Normalized range: [{normalized_durations.min():.2f}, {normalized_durations.max():.2f}]")
    else:
        normalized_durations = zscore(durations).astype(np.float32)
        print(f"Duration normalization (using all data - old behavior):")
        print(f"  Original range: [{durations.min():.2f}, {durations.max():.2f}]")
        print(f"  Normalized range: [{normalized_durations.min():.2f}, {normalized_durations.max():.2f}]")
        print(f"  Mean: {normalized_durations.mean():.4f}, Std: {normalized_durations.std():.4f}")
    
    return encoded_sequences, normalized_durations, trace_ids, encoder, static_encoders, static_attributes
