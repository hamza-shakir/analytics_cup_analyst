"""
segments.py v2 - Match Segmentation Functions

Core functionality for breaking matches into tactical periods:
- By game state (winning/drawing/losing)
- By time windows (15-min blocks, custom)
- Around goals (before/after context)
- Around substitutions

Returns kloppy datasets (filtered) for maximum flexibility.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple, Optional
from .load_data_v2 import (
    load_tracking_dataset,
    load_event_data,
    to_long_dataframe
)


# ----------------------------------------------------------
# Core Segmentation: By Game State
# ----------------------------------------------------------

def by_game_state(
    match_id: Union[int, str],
    state: str = 'drawing',
    team: str = 'home',
    return_format: str = 'dataset'
) -> Union['TrackingDataset', pd.DataFrame]:
    """
    Get frames where specified team is in given game state.
    
    Args:
        match_id: Match identifier
        state: 'winning', 'drawing', or 'losing'
        team: 'home' or 'away' (which team's perspective)
        return_format: 'dataset' (kloppy) or 'dataframe' (long format)
    
    Returns:
        Filtered dataset/dataframe containing only frames in specified state
    
    Example:
        >>> # Get frames where home team is winning
        >>> winning_segment = by_game_state(1886347, state='winning', team='home')
        >>> print(f"Frames when winning: {len(winning_segment.records)}")
        
        >>> # Get as DataFrame
        >>> winning_df = by_game_state(1886347, state='winning', team='home', 
        ...                             return_format='dataframe')
    
    Note:
        Game state is determined from score progression events.
        State changes at the exact frame where a goal is scored.
    """
    # Validate inputs
    if state not in ['winning', 'drawing', 'losing']:
        raise ValueError(f"state must be 'winning', 'drawing', or 'losing', got '{state}'")
    if team not in ['home', 'away']:
        raise ValueError(f"team must be 'home' or 'away', got '{team}'")
    
    # Load data
    dataset = load_tracking_dataset(match_id)
    events_df = load_event_data(match_id)
    
    if events_df.empty:
        # No events, entire match is drawing
        if state == 'drawing':
            return dataset if return_format == 'dataset' else to_long_dataframe(dataset, match_id)
        else:
            # Return empty dataset
            return dataset.filter(lambda f: False) if return_format == 'dataset' else pd.DataFrame()
    
    # Determine frame ranges for each game state
    state_windows = _get_game_state_windows(events_df, team)
    
    # Filter to requested state
    target_frames = []
    for window in state_windows:
        if window['state'] == state:
            target_frames.extend(range(window['start_frame'], window['end_frame'] + 1))
    
    if not target_frames:
        # No frames in this state
        return dataset.filter(lambda f: False) if return_format == 'dataset' else pd.DataFrame()
    
    # Filter dataset to these frames
    target_frames_set = set(target_frames)
    filtered_dataset = dataset.filter(
        lambda frame: frame.frame_id in target_frames_set
    )
    
    if return_format == 'dataframe':
        return to_long_dataframe(filtered_dataset, match_id)
    
    return filtered_dataset


def get_all_game_states(
    match_id: Union[int, str],
    team: str = 'home',
    return_format: str = 'dataset'
) -> Dict[str, Union['TrackingDataset', pd.DataFrame]]:
    """
    Get segments for ALL game states (drawing, winning, losing).
    
    Convenience function that calls by_game_state() for each state.
    
    Args:
        match_id: Match identifier
        team: 'home' or 'away' (which team's perspective)
        return_format: 'dataset' or 'dataframe'
    
    Returns:
        Dictionary mapping state -> filtered dataset/dataframe
        Keys: 'drawing', 'winning', 'losing'
    
    Example:
        >>> segments = get_all_game_states(1886347, team='home')
        >>> for state, segment in segments.items():
        ...     print(f"{state}: {len(segment.records)} frames")
        drawing: 25000 frames
        winning: 3000 frames
        losing: 0 frames
    """
    states = ['drawing', 'winning', 'losing']
    
    result = {}
    for state in states:
        segment = by_game_state(match_id, state=state, team=team, return_format=return_format)
        
        # Only include states that have frames
        if return_format == 'dataset':
            if len(segment.records) > 0:
                result[state] = segment
        else:
            if not segment.empty:
                result[state] = segment
    
    return result


def get_game_state_summary(
    match_id: Union[int, str],
    team: str = 'home'
) -> pd.DataFrame:
    """
    Get summary of time spent in each game state.
    
    Args:
        match_id: Match identifier
        team: 'home' or 'away'
    
    Returns:
        DataFrame with columns: state, frames, minutes, percentage
    
    Example:
        >>> summary = get_game_state_summary(1886347, team='home')
        >>> print(summary)
             state  frames  minutes  percentage
        0  drawing   25000     41.7        79.3
        1  winning    6522     10.9        20.7
        2   losing       0      0.0         0.0
    """
    dataset = load_tracking_dataset(match_id)
    segments = get_all_game_states(match_id, team=team, return_format='dataset')
    
    frame_rate = dataset.metadata.frame_rate
    total_frames = len(dataset.records)
    
    summary_data = []
    for state in ['drawing', 'winning', 'losing']:
        if state in segments:
            frames = len(segments[state].records)
        else:
            frames = 0
        
        minutes = frames / frame_rate / 60
        percentage = 100 * frames / total_frames if total_frames > 0 else 0
        
        summary_data.append({
            'state': state,
            'frames': frames,
            'minutes': round(minutes, 1),
            'percentage': round(percentage, 1)
        })
    
    return pd.DataFrame(summary_data)


# ----------------------------------------------------------
# Time-Based Segmentation
# ----------------------------------------------------------

def by_time_window(
    match_id: Union[int, str],
    start_minute: float,
    end_minute: float,
    return_format: str = 'dataset'
) -> Union['TrackingDataset', pd.DataFrame]:
    """
    Get frames within a specific time window.
    
    Args:
        match_id: Match identifier
        start_minute: Start time in minutes (e.g., 0, 15, 30)
        end_minute: End time in minutes (e.g., 15, 30, 45)
        return_format: 'dataset' or 'dataframe'
    
    Returns:
        Filtered dataset/dataframe for the time window
    
    Example:
        >>> # Get first 15 minutes
        >>> first_15 = by_time_window(1886347, start_minute=0, end_minute=15)
        
        >>> # Get 30-45 minute period
        >>> mid_half = by_time_window(1886347, start_minute=30, end_minute=45)
    """
    dataset = load_tracking_dataset(match_id)
    
    # Convert minutes to seconds for filtering
    start_seconds = start_minute * 60
    end_seconds = end_minute * 60
    
    # Filter frames by timestamp
    filtered_dataset = dataset.filter(
        lambda frame: start_seconds <= frame.timestamp.total_seconds() <= end_seconds
    )
    
    if return_format == 'dataframe':
        return to_long_dataframe(filtered_dataset, match_id)
    
    return filtered_dataset


def by_time_windows(
    match_id: Union[int, str],
    window_minutes: int = 15,
    return_format: str = 'dataset'
) -> Dict[str, Union['TrackingDataset', pd.DataFrame]]:
    """
    Segment match into regular time windows.
    
    Args:
        match_id: Match identifier
        window_minutes: Length of each window in minutes (default: 15)
        return_format: 'dataset' or 'dataframe'
    
    Returns:
        Dictionary mapping time label -> filtered dataset/dataframe
        Keys like: '0-15', '15-30', '30-45', etc.
    
    Example:
        >>> windows = by_time_windows(1886347, window_minutes=15)
        >>> for label, segment in windows.items():
        ...     print(f"{label}: {len(segment.records)} frames")
        0-15: 9000 frames
        15-30: 9000 frames
        30-45: 9000 frames
        ...
    """
    dataset = load_tracking_dataset(match_id)
    
    # Calculate total duration
    total_seconds = max(frame.timestamp.total_seconds() for frame in dataset.records)
    total_minutes = total_seconds / 60
    
    # Create windows
    windows = {}
    start = 0
    
    while start < total_minutes:
        end = min(start + window_minutes, total_minutes)
        label = f"{int(start)}-{int(end)}"
        
        segment = by_time_window(match_id, start, end, return_format=return_format)
        
        # Only include if has frames
        if return_format == 'dataset':
            if len(segment.records) > 0:
                windows[label] = segment
        else:
            if not segment.empty:
                windows[label] = segment
        
        start = end
    
    return windows


# ----------------------------------------------------------
# Goal-Context Segmentation
# ----------------------------------------------------------

def around_goal(
    match_id: Union[int, str],
    goal_index: int = 0,
    before_minutes: float = 5,
    after_minutes: float = 5,
    return_format: str = 'dataset'
) -> Dict[str, Union['TrackingDataset', pd.DataFrame]]:
    """
    Get frames before and after a goal.
    
    Args:
        match_id: Match identifier
        goal_index: Which goal (0 = first goal, 1 = second goal, etc.)
        before_minutes: Minutes before goal to include
        after_minutes: Minutes after goal to include
        return_format: 'dataset' or 'dataframe'
    
    Returns:
        Dictionary with keys 'before' and 'after'
    
    Example:
        >>> # Analyze 5 mins before/after first goal
        >>> goal_context = around_goal(1886347, goal_index=0, 
        ...                             before_minutes=5, after_minutes=5)
        >>> before = goal_context['before']
        >>> after = goal_context['after']
    """
    dataset = load_tracking_dataset(match_id)
    events_df = load_event_data(match_id)
    
    if events_df.empty:
        raise ValueError("No events found for this match")
    
    # Get goal events (score changes, excluding 0-0 start)
    score_pairs = events_df[["home_team_score", "away_team_score"]].apply(tuple, axis=1)
    mask = score_pairs.ne(score_pairs.shift()).fillna(False)
    goal_events = events_df[mask].reset_index(drop=True)
    
    if len(goal_events) <= goal_index + 1:  # +1 because first is 0-0
        raise ValueError(f"Goal index {goal_index} not found (only {len(goal_events)-1} goals)")
    
    # Get goal frame (skip 0-0 start)
    goal_frame = int(goal_events.iloc[goal_index + 1]['frame_start'])
    frame_rate = dataset.metadata.frame_rate
    
    # Calculate frame ranges
    before_frames = int(before_minutes * 60 * frame_rate)
    after_frames = int(after_minutes * 60 * frame_rate)
    
    before_start = max(0, goal_frame - before_frames)
    before_end = goal_frame
    after_start = goal_frame
    after_end = goal_frame + after_frames
    
    # Filter datasets
    before_segment = dataset.filter(
        lambda f: before_start <= f.frame_id < before_end
    )
    after_segment = dataset.filter(
        lambda f: after_start <= f.frame_id <= after_end
    )
    
    result = {}
    if return_format == 'dataframe':
        result['before'] = to_long_dataframe(before_segment, match_id)
        result['after'] = to_long_dataframe(after_segment, match_id)
    else:
        result['before'] = before_segment
        result['after'] = after_segment
    
    return result


def all_goals_context(
    match_id: Union[int, str],
    before_minutes: float = 5,
    after_minutes: float = 5,
    return_format: str = 'dataset'
) -> List[Dict[str, Union['TrackingDataset', pd.DataFrame]]]:
    """
    Get before/after context for ALL goals in the match.
    
    Args:
        match_id: Match identifier
        before_minutes: Minutes before each goal
        after_minutes: Minutes after each goal
        return_format: 'dataset' or 'dataframe'
    
    Returns:
        List of dictionaries, one per goal, each with 'before' and 'after' keys
    
    Example:
        >>> all_contexts = all_goals_context(1886347)
        >>> for i, context in enumerate(all_contexts):
        ...     print(f"Goal {i+1}:")
        ...     print(f"  Before: {len(context['before'].records)} frames")
        ...     print(f"  After: {len(context['after'].records)} frames")
    """
    events_df = load_event_data(match_id)
    
    if events_df.empty:
        return []
    
    # Count goals
    score_pairs = events_df[["home_team_score", "away_team_score"]].apply(tuple, axis=1)
    mask = score_pairs.ne(score_pairs.shift()).fillna(False)
    goal_events = events_df[mask].reset_index(drop=True)
    num_goals = len(goal_events) - 1  # Exclude 0-0 start
    
    contexts = []
    for i in range(num_goals):
        try:
            context = around_goal(
                match_id, 
                goal_index=i,
                before_minutes=before_minutes,
                after_minutes=after_minutes,
                return_format=return_format
            )
            contexts.append(context)
        except ValueError:
            continue
    
    return contexts


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def _get_game_state_windows(
    events_df: pd.DataFrame,
    team: str = 'home'
) -> List[Dict]:
    """
    Internal function to determine game state windows from events.
    
    Returns list of dicts with: state, start_frame, end_frame
    """
    # Find score change events
    score_pairs = events_df[["home_team_score", "away_team_score"]].apply(tuple, axis=1)
    
    # Find first 0-0
    zero_zero_idx = events_df.loc[
        (events_df["home_team_score"] == 0) & (events_df["away_team_score"] == 0)
    ].index
    
    if len(zero_zero_idx) > 0:
        start_idx = zero_zero_idx[0]
    else:
        start_idx = events_df.index[0]
    
    # Get score changes from start
    df_from_start = events_df.loc[start_idx:].copy()
    score_pairs = df_from_start[["home_team_score", "away_team_score"]].apply(tuple, axis=1)
    mask = score_pairs.ne(score_pairs.shift()).fillna(True)
    score_events = df_from_start[mask].reset_index(drop=True)
    
    # Build windows
    windows = []
    
    for i in range(len(score_events)):
        home_score = int(score_events.iloc[i]['home_team_score'])
        away_score = int(score_events.iloc[i]['away_team_score'])
        start_frame = int(score_events.iloc[i]['frame_start'])
        
        # Determine end frame
        if i < len(score_events) - 1:
            end_frame = int(score_events.iloc[i + 1]['frame_start']) - 1
        else:
            # Last window extends to end of available events
            end_frame = int(events_df['frame_start'].max()) + 10000  # Generous buffer
        
        # Determine state from team's perspective
        if team == 'home':
            if home_score > away_score:
                state = 'winning'
            elif home_score < away_score:
                state = 'losing'
            else:
                state = 'drawing'
        else:  # away
            if away_score > home_score:
                state = 'winning'
            elif away_score < home_score:
                state = 'losing'
            else:
                state = 'drawing'
        
        windows.append({
            'state': state,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'home_score': home_score,
            'away_score': away_score
        })
    
    return windows