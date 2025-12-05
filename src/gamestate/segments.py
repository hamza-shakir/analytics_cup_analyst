"""
Core functionality for breaking matches into tactical periods:
- By game state (winning/drawing/losing)
- By time windows (15-min blocks, custom) - WITH MATCH TIME SUPPORT
- Around goals (before/after context)

Returns kloppy TrackingDataset objects only.
Use to_long_dataframe() if you need DataFrame format.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple, Optional
from .load_data import (
    load_tracking_dataset,
    load_event_data,
    to_long_dataframe
)


# ----------------------------------------------------------
# Helper: Get Full Match
# ----------------------------------------------------------

def get_full_match(match_id: Union[int, str], only_alive: bool = True) -> 'TrackingDataset':
    """
    Get full match as TrackingDataset for filtering.
    
    This is the recommended starting point for filter-based analysis.
    Loads the entire match tracking data which can then be filtered
    by game state, phase type, possession, etc.
    
    Args:
        match_id: Match identifier
        only_alive: If True, only include frames where ball is in play (default: True)
    
    Returns:
        TrackingDataset containing entire match
    
    Example - Simple usage:
        >>> import gamestate as gs
        >>> 
        >>> # Get full match
        >>> segment = gs.get_full_match(1886347)
        >>> print(f"Total frames: {len(segment.records)}")
    
    Example - With filters:
        >>> # Start with full match
        >>> segment = gs.get_full_match(1886347)
        >>> 
        >>> # Apply filters
        >>> segment = gs.filter_by_game_state(segment, 'winning', 'home', 1886347)
        >>> segment = gs.filter_by_phase_type(segment, 'build_up', 'home', 1886347)
        >>> 
        >>> # Analyze
        >>> positions = gs.average_positions(segment, team='home', match_id=1886347)
    
    Note:
        This function is equivalent to load_tracking_dataset() but provides
        a clearer starting point for filter-based workflows.
    """
    return load_tracking_dataset(match_id, only_alive=only_alive)


# ----------------------------------------------------------
# Core Segmentation: By Game State
# ----------------------------------------------------------

def segment_by_game_state(
    match_id: Union[int, str],
    state: str = 'drawing',
    team: str = 'home'
) -> 'TrackingDataset':
    """
    Get frames where specified team is in given game state.
    
    CONVENIENCE FUNCTION: This is equivalent to:
        segment = get_full_match(match_id)
        segment = filter_by_game_state(segment, state, team, match_id)
    
    For stacking multiple filters, use filter_by_game_state() directly.
    
    Args:
        match_id: Match identifier
        state: 'winning', 'drawing', or 'losing'
        team: 'home' or 'away' (which team's perspective)
    
    Returns:
        TrackingDataset containing only frames in specified state
    
    Example - Simple usage (recommended for single filter):
        >>> import gamestate as gs
        >>> 
        >>> # Get frames where home team is winning
        >>> winning_segment = gs.segment_by_game_state(1886347, state='winning', team='home')
        >>> print(f"Frames when winning: {len(winning_segment.records)}")
        >>> 
        >>> # Analyze
        >>> positions = gs.average_positions(winning_segment, team='home', match_id=1886347)
    
    Example - Compositional approach (recommended for multiple filters):
        >>> # Start with full match
        >>> segment = gs.get_full_match(1886347)
        >>> 
        >>> # Stack filters
        >>> segment = gs.filter_by_game_state(segment, 'winning', 'home', 1886347)
        >>> segment = gs.filter_by_phase_type(segment, 'build_up', 'home', 1886347)
        >>> segment = gs.filter_by_third(segment, 'defensive', 'home', 1886347)
        >>> 
        >>> # Analyze stacked filters
        >>> positions = gs.average_positions(segment, team='home', match_id=1886347)
    
    Note:
        Game state is determined from score progression events.
        State changes at the exact frame where a goal is scored.
    """
    # Validate inputs
    if state not in ['winning', 'drawing', 'losing']:
        raise ValueError(f"state must be 'winning', 'drawing', or 'losing', got '{state}'")
    if team not in ['home', 'away']:
        raise ValueError(f"team must be 'home' or 'away', got '{team}'")
    
    # Import here to avoid circular dependency
    try:
        from .filters import filter_by_game_state as filter_func
        
        # Use filter-based approach
        full = get_full_match(match_id)
        return filter_func(full, state, team, match_id)
    
    except ImportError:
        # Fallback to original implementation if filters not available yet
        # This ensures backwards compatibility during development
        dataset = load_tracking_dataset(match_id)
        events_df = load_event_data(match_id)
        
        if events_df.empty:
            # No events, entire match is drawing
            if state == 'drawing':
                return dataset
            else:
                # Return empty dataset
                return dataset.filter(lambda f: False)
        
        # Determine frame ranges for each game state
        state_windows = _get_game_state_windows(events_df, team)
        
        # Filter to requested state
        target_frames = []
        for window in state_windows:
            if window['state'] == state:
                target_frames.extend(range(window['start_frame'], window['end_frame'] + 1))
        
        if not target_frames:
            # No frames in this state
            return dataset.filter(lambda f: False)
        
        # Filter dataset to these frames
        target_frames_set = set(target_frames)
        filtered_dataset = dataset.filter(
            lambda frame: frame.frame_id in target_frames_set
        )
        
        return filtered_dataset


def get_all_game_states(
    match_id: Union[int, str],
    team: str = 'home'
) -> Dict[str, 'TrackingDataset']:
    """
    Get segments for ALL game states (drawing, winning, losing).
    
    Convenience function that calls filter_by_game_state() for each state.
    
    Args:
        match_id: Match identifier
        team: 'home' or 'away' (which team's perspective)
    
    Returns:
        Dictionary mapping state -> TrackingDataset
        Keys: 'drawing', 'winning', 'losing' (only includes states with data)
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> segments = gs.get_all_game_states(1886347, team='home')
        >>> for state, segment in segments.items():
        ...     print(f"{state}: {len(segment.records)} frames")
        drawing: 25000 frames
        winning: 3000 frames
        losing: 0 frames
        >>> 
        >>> # Compare metrics across states
        >>> for state, segment in segments.items():
        ...     comp = gs.team_compactness(segment, team='home', match_id=1886347)
        ...     print(f"{state}: width={comp['width']:.1f}m")
    """
    # Import here to avoid circular dependency
    try:
        from .filters import filter_by_game_state as filter_func
        
        # Use filter-based approach
        full = get_full_match(match_id)
        states = ['drawing', 'winning', 'losing']
        
        result = {}
        for state in states:
            segment = filter_func(full, state, team, match_id)
            
            # Only include states that have frames
            if len(segment.records) > 0:
                result[state] = segment
        
        return result
    
    except ImportError:
        # Fallback to calling segment_by_game_state
        states = ['drawing', 'winning', 'losing']
        
        result = {}
        for state in states:
            segment = segment_by_game_state(match_id, state=state, team=team)
            
            # Only include states that have frames
            if len(segment.records) > 0:
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
        DataFrame with columns: state, frames, tracking_minutes, percentage
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> summary = gs.get_game_state_summary(1886347, team='home')
        >>> print(summary)
             state  frames  tracking_minutes  percentage
        0  drawing   25000              41.7        79.3
        1  winning    6522              10.9        20.7
        2   losing       0               0.0         0.0
    
    Note:
        'tracking_minutes' shows actual tracking data duration.
        This excludes dead ball time if only_alive=True was used.
        For match clock time analysis, use segment_by_time_windows with use_match_time=True.
    """
    dataset = load_tracking_dataset(match_id)
    segments = get_all_game_states(match_id, team=team)
    
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
            'tracking_minutes': round(minutes, 1),
            'percentage': round(percentage, 1)
        })
    
    return pd.DataFrame(summary_data)


# ----------------------------------------------------------
# Time-Based Segmentation (WITH MATCH TIME SUPPORT)
# ----------------------------------------------------------

def segment_by_time_window(
    match_id: Union[int, str],
    start_minute: float,
    end_minute: float,
    use_match_time: bool = True
) -> 'TrackingDataset':
    """
    Get frames within a specific time window.
    
    Args:
        match_id: Match identifier
        start_minute: Start time in minutes (e.g., 0, 15, 30)
        end_minute: End time in minutes (e.g., 15, 30, 45)
        use_match_time: If True, uses match clock time; if False, uses data time
    
    Returns:
        TrackingDataset for the time window
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Get first 15 minutes of MATCH TIME (default)
        >>> first_15 = gs.segment_by_time_window(1886347, start_minute=0, end_minute=15)
        >>> 
        >>> # Get 30-45 minute period
        >>> mid_half = gs.segment_by_time_window(1886347, start_minute=30, end_minute=45)
        >>> 
        >>> # Use data time instead (old behavior)
        >>> first_15_data = gs.segment_by_time_window(1886347, start_minute=0, end_minute=15, 
        ...                                             use_match_time=False)
        >>> 
        >>> # Convert to DataFrame if needed
        >>> first_15_df = gs.to_long_dataframe(first_15, 1886347)
    
    Note:
        - use_match_time=True: Uses match clock (0-90 mins), aligns with how analysts think
        - use_match_time=False: Uses tracking data duration, excludes dead ball time
    """
    dataset = load_tracking_dataset(match_id)
    
    if use_match_time:
        # Use match clock time from events
        events_df = load_event_data(match_id)
        
        if events_df.empty:
            print("Warning: No event data available, falling back to data time")
            use_match_time = False
        else:
            # Create frame-to-minute mapping
            frame_to_minute = _create_frame_to_minute_mapping(events_df)
            
            # Filter frames by match time
            filtered_dataset = dataset.filter(
                lambda frame: (
                    frame.frame_id in frame_to_minute and
                    start_minute <= frame_to_minute[frame.frame_id] <= end_minute
                )
            )
            return filtered_dataset
    
    # Use data time (original implementation)
    start_seconds = start_minute * 60
    end_seconds = end_minute * 60
    
    filtered_dataset = dataset.filter(
        lambda frame: start_seconds <= frame.timestamp.total_seconds() <= end_seconds
    )
    
    return filtered_dataset


def segment_by_time_windows(
    match_id: Union[int, str],
    window_minutes: int = 15,
    use_match_time: bool = True,
    max_minute: int = 90
) -> Dict[str, 'TrackingDataset']:
    """
    Segment match into regular time windows.
    
    Args:
        match_id: Match identifier
        window_minutes: Length of each window in minutes (default: 15)
        use_match_time: If True, uses match clock (0-90); if False, uses data time
        max_minute: Maximum minute to consider (default: 90 for regular time)
    
    Returns:
        Dictionary mapping time label -> TrackingDataset
        Keys like: '0-15', '15-30', '30-45', etc.
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Standard 15-min windows using match clock (DEFAULT)
        >>> windows = gs.segment_by_time_windows(1886347, window_minutes=15)
        >>> for label, segment in windows.items():
        ...     print(f"{label}: {len(segment.records)} frames")
        0-15: 8500 frames
        15-30: 9200 frames
        30-45: 8800 frames
        45-60: 4200 frames
        60-75: 3100 frames
        75-90: 2800 frames
        >>> 
        >>> # Use data time instead (old behavior)
        >>> windows_data = gs.segment_by_time_windows(1886347, use_match_time=False)
        0-15: 9000 frames
        15-30: 9000 frames
        30-45: 9000 frames
        45-52: 4500 frames
    
    Note:
        - use_match_time=True (DEFAULT): Creates windows for full match (0-90 mins)
        - use_match_time=False: Creates windows based on available tracking data
    """
    if use_match_time:
        # Use match clock time (0-90 or 0-max_minute)
        total_minutes = max_minute
    else:
        # Use data time (current implementation)
        dataset = load_tracking_dataset(match_id)
        total_seconds = max(frame.timestamp.total_seconds() for frame in dataset.records)
        total_minutes = total_seconds / 60
    
    # Create windows
    windows = {}
    start = 0
    
    while start < total_minutes:
        end = min(start + window_minutes, total_minutes)
        label = f"{int(start)}-{int(end)}"
        
        segment = segment_by_time_window(
            match_id, start, end, 
            use_match_time=use_match_time
        )
        
        # Only include if has frames
        if len(segment.records) > 0:
            windows[label] = segment
        
        start = end
    
    return windows


# ----------------------------------------------------------
# Goal-Context Segmentation
# ----------------------------------------------------------

def segment_around_goal(
    match_id: Union[int, str],
    goal_index: int = 0,
    before_minutes: float = 5,
    after_minutes: float = 5
) -> Dict[str, 'TrackingDataset']:
    """
    Get frames before and after a goal.
    
    Args:
        match_id: Match identifier
        goal_index: Which goal (0 = first goal, 1 = second goal, etc.)
        before_minutes: Minutes before goal to include
        after_minutes: Minutes after goal to include
    
    Returns:
        Dictionary with keys 'before' and 'after', each containing TrackingDataset
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Analyze 5 mins before/after first goal
        >>> goal_context = gs.segment_around_goal(1886347, goal_index=0, 
        ...                                        before_minutes=5, after_minutes=5)
        >>> before = goal_context['before']
        >>> after = goal_context['after']
        >>> 
        >>> # Convert to DataFrame if needed
        >>> before_df = gs.to_long_dataframe(before, 1886347)
        >>> after_df = gs.to_long_dataframe(after, 1886347)
    
    Note:
        Time windows are frame-based (not match clock based).
        This is reasonable for goal context analysis where you want
        consistent amounts of tracking data.
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
    
    return {
        'before': before_segment,
        'after': after_segment
    }


def all_goals_context(
    match_id: Union[int, str],
    before_minutes: float = 5,
    after_minutes: float = 5
) -> List[Dict[str, 'TrackingDataset']]:
    """
    Get before/after context for ALL goals in the match.
    
    Args:
        match_id: Match identifier
        before_minutes: Minutes before each goal
        after_minutes: Minutes after each goal
    
    Returns:
        List of dictionaries, one per goal, each with 'before' and 'after' keys
        containing TrackingDataset objects
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> all_contexts = gs.all_goals_context(1886347)
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
            context = segment_around_goal(
                match_id, 
                goal_index=i,
                before_minutes=before_minutes,
                after_minutes=after_minutes
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


def _create_frame_to_minute_mapping(events_df: pd.DataFrame) -> Dict[int, float]:
    """
    Create mapping from frame_id to match minute.
    
    Uses event data to interpolate frame timestamps to match clock time.
    
    Args:
        events_df: Events DataFrame with frame_start and minute_start
    
    Returns:
        Dictionary mapping frame_id -> match_minute
    """
    # Sort events by frame
    events_sorted = events_df[['frame_start', 'minute_start']].sort_values('frame_start').drop_duplicates()
    
    frame_to_minute = {}
    
    # For each consecutive pair of events, interpolate
    for i in range(len(events_sorted) - 1):
        start_frame = int(events_sorted.iloc[i]['frame_start'])
        start_minute = float(events_sorted.iloc[i]['minute_start'])
        end_frame = int(events_sorted.iloc[i + 1]['frame_start'])
        end_minute = float(events_sorted.iloc[i + 1]['minute_start'])
        
        # Linear interpolation for frames between events
        frame_diff = end_frame - start_frame
        minute_diff = end_minute - start_minute
        
        if frame_diff > 0:
            for frame_id in range(start_frame, end_frame + 1):
                progress = (frame_id - start_frame) / frame_diff
                minute = start_minute + (progress * minute_diff)
                frame_to_minute[frame_id] = minute
    
    # Handle frames after last event (assume continuous play)
    if len(events_sorted) > 0:
        last_frame = int(events_sorted.iloc[-1]['frame_start'])
        last_minute = float(events_sorted.iloc[-1]['minute_start'])
        
        # Estimate frame rate from events
        if len(events_sorted) > 1:
            total_frames = int(events_sorted.iloc[-1]['frame_start'] - events_sorted.iloc[0]['frame_start'])
            total_minutes = float(events_sorted.iloc[-1]['minute_start'] - events_sorted.iloc[0]['minute_start'])
            if total_minutes > 0:
                frames_per_minute = total_frames / total_minutes
            else:
                frames_per_minute = 600  # Default ~10fps * 60s
        else:
            frames_per_minute = 600
        
        # Extend mapping for frames after last event (up to +10 minutes)
        for offset in range(0, int(10 * frames_per_minute)):
            frame_id = last_frame + offset
            minute = last_minute + (offset / frames_per_minute)
            frame_to_minute[frame_id] = minute
    
    return frame_to_minute