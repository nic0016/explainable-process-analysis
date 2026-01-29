import pandas as pd
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import zscore


def create_eventlog(xes_path):
    return xes_importer.apply(xes_path)


def build_event_reference(event_log):
    events = set()
    for trace in event_log:
        for event in trace:
            name = event.get("concept:name", None)
            if name:
                events.add(name)
    return sorted(events)


def build_event_id_mapping(event_names):
    return {name: idx + 1 for idx, name in enumerate(event_names)}


def trace_duration_seconds(trace):
    timestamps = [event.get("time:timestamp", None) for event in trace if "time:timestamp" in event]
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)).total_seconds()


def normalize_durations_split_aware(durations, train_indices):
    durations = np.array(durations)
    train_durations = durations[train_indices]
    
    train_mean = np.mean(train_durations)
    train_std = np.std(train_durations)
    
    if train_std > 0:
        normalized_durations = (durations - train_mean) / train_std
    else:
        normalized_durations = durations
    
    return normalized_durations, train_mean, train_std


def build_dataframe(event_log, train_indices=None):
    if not event_log:
        return pd.DataFrame(), None, None, None
    
    event_names = build_event_reference(event_log)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    encoder.fit(np.array(event_names).reshape(-1, 1))
    
    max_events = max(len(trace) for trace in event_log)
    nb_events = len(event_names)
    
    print(f"Dataset info:")
    print(f"  Traces: {len(event_log)}, Max length: {max_events}, Events: {nb_events}")
    if train_indices is not None:
        print(f"  Split-aware normalization (train_n={len(train_indices)})")
    
    encoded_sequences = np.zeros((len(event_log), max_events, nb_events), dtype=np.int32)
    durations = []
    trace_ids = []
    
    for trace_idx, trace in enumerate(event_log):
        trace_id = trace.attributes.get("concept:name", f"trace_{trace_idx}")
        duration = trace_duration_seconds(trace)
        
        ordered_events = sorted(trace, key=lambda evt: evt.get("time:timestamp", 0))
        
        for event_idx, event in enumerate(ordered_events):
            if event_idx >= max_events:
                break
            name = event.get("concept:name", None)
            if name and name in event_names:
                one_hot = encoder.transform([[name]])[0]
                encoded_sequences[trace_idx, event_idx, :] = one_hot
        
        durations.append(duration)
        trace_ids.append(trace_id)
    
    durations = np.array(durations)
    
    if train_indices is not None:
        normalized_durations, train_mean, train_std = normalize_durations_split_aware(durations, train_indices)
    else:
        if len(durations) > 1 and np.std(durations) > 0:
            normalized_durations = zscore(durations)
        else:
            normalized_durations = durations
    
    return encoded_sequences, normalized_durations, trace_ids, encoder


def head_with_events(encoded_sequences, normalized_durations, trace_ids, encoder, n=5, k=5):
    if encoded_sequences is None:
        return "No data available"
    
    event_names = encoder.categories_[0] if encoder else []
    
    print(f"First {n} traces (showing first {k} events):")
    print("=" * 80)
    
    for i in range(min(n, len(trace_ids))):
        print(f"Trace {i+1}: {trace_ids[i]}")
        print(f"  Duration (normalized): {normalized_durations[i]:.4f}")
        
        for j in range(min(k, encoded_sequences.shape[1])):
            event_vector = encoded_sequences[i, j, :]
            if np.sum(event_vector) > 0:
                event_idx = np.argmax(event_vector)
                event_name = event_names[event_idx] if event_idx < len(event_names) else "Unknown"
                print(f"    Position {j+1}: {event_name}")
        print()
    
    return None


def convert_to_dataframe(encoded_sequences, normalized_durations, trace_ids, encoder):
    if encoded_sequences is None:
        return pd.DataFrame()
    
    event_names = encoder.categories_[0] if encoder else []
    nb_traces, max_events, nb_events = encoded_sequences.shape
    
    columns = []
    for event_pos in range(max_events):
        for event_name in event_names:
            columns.append(f"Event_{event_pos+1}_{event_name}")
    columns.append("Total_Duration_Normalized")
    
    flattened_data = encoded_sequences.reshape(nb_traces, -1)
    data_with_duration = np.column_stack([flattened_data, normalized_durations])
    
    dataframe = pd.DataFrame(data_with_duration, columns=columns)
    dataframe.insert(0, "Trace_ID", trace_ids)
    
    return dataframe


def load_encoded_with_static_attributes(xes_path, attribute_names=None):
    event_log = create_eventlog(xes_path)
    encoded_sequences, normalized_durations, trace_ids, encoder = build_dataframe(event_log)
    df_events = convert_to_dataframe(encoded_sequences, normalized_durations, trace_ids, encoder)

    from .static_features import build_static_features_dataframe

    df_static, static_encoders, static_attributes = build_static_features_dataframe(event_log, attribute_names)

    if "Total_Duration" in df_static.columns:
        df_static = df_static.drop(columns=["Total_Duration"])

    merged = df_events.merge(df_static, on="Trace_ID", how="left").fillna(0)

    return merged, encoder, static_encoders, static_attributes


def build_dataframe_with_static(event_log, train_indices=None, static_attributes=None):
    if not event_log:
        return None, None, None, None, {}, []
    
    event_names = build_event_reference(event_log)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    encoder.fit(np.array(event_names).reshape(-1, 1))
    
    max_events = max(len(trace) for trace in event_log)
    nb_event_types = len(event_names)
    
    if static_attributes is None:
        static_attributes = []
        candidate_attrs = ['org:resource', 'org:role', 'concept:name']
        
        for attr in candidate_attrs:
            count = sum(1 for trace in event_log for event in trace if attr in event)
            total_events = sum(len(trace) for trace in event_log)
            if count / total_events > 0.8:
                static_attributes.append(attr)
    
    print(f"Dataset info (with static features):")
    print(f"  Traces: {len(event_log)}, Max length: {max_events}, Events: {nb_event_types}")
    print(f"  Static attributes: {static_attributes}")
    
    static_values = {attr: set() for attr in static_attributes}
    for trace in event_log:
        for event in trace:
            for attr in static_attributes:
                if attr in event:
                    static_values[attr].add(str(event[attr]))
    
    static_encoders = {}
    static_dims = {}
    for attr in static_attributes:
        encoder_static = LabelEncoder()
        encoder_static.fit(sorted(static_values[attr]))
        static_encoders[attr] = encoder_static
        static_dims[attr] = len(encoder_static.classes_)
    
    total_static_dim = sum(static_dims.values())
    total_feature_dim = nb_event_types + total_static_dim
    
    encoded_sequences = np.zeros((len(event_log), max_events, total_feature_dim), dtype=np.float32)
    trace_ids = []
    durations = []
    
    for trace_idx, trace in enumerate(event_log):
        trace_id = trace.attributes.get("concept:name", f"trace_{trace_idx}")
        trace_ids.append(trace_id)
        
        duration = trace_duration_seconds(trace)
        durations.append(duration)
        
        ordered_events = sorted(trace, key=lambda evt: evt.get("time:timestamp", None) or 0)
        
        for event_idx, event in enumerate(ordered_events):
            if event_idx >= max_events:
                break
            
            event_name = event.get("concept:name", "UNKNOWN")
            if event_name in event_names:
                event_onehot = encoder.transform([[event_name]])[0].astype(np.float32)
            else:
                event_onehot = np.zeros(nb_event_types, dtype=np.float32)
            
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
                    static_onehot.extend(np.zeros(static_dims[attr], dtype=np.float32))
            
            full_features = np.concatenate([event_onehot, np.array(static_onehot, dtype=np.float32)])
            encoded_sequences[trace_idx, event_idx, :] = full_features
    
    durations = np.array(durations)
    
    if train_indices is not None:
        normalized_durations, _, _ = normalize_durations_split_aware(durations, train_indices)
        normalized_durations = normalized_durations.astype(np.float32)
    else:
        normalized_durations = zscore(durations).astype(np.float32)
    
    return encoded_sequences, normalized_durations, trace_ids, encoder, static_encoders, static_attributes
