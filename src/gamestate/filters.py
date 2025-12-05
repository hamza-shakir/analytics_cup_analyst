"""
filters.py - Phase-Based Filtering

Filter tracking segments by tactical context:
- Phase types (attack phases: build_up, create, finish, etc.)
- Defensive phases (high_block, medium_block, low_block, etc.)
- Pitch thirds (defensive, middle, attacking)
- Game states (winning, drawing, losing)

All filters operate on existing TrackingDataset segments and return
filtered TrackingDataset objects.
"""

import pandas as pd
import numpy as np
from typing import Union, Set
from .load_data import (
    load_phases_data,
    load_event_data,
    load_tracking_dataset
)
from .helpers import has_frames


# ----------------------------------------------------------
# Filter 1: By IP (In-Possession) Phase Type
# ----------------------------------------------------------

def filter_by_ip_phase(
    segment: 'TrackingDataset',
    phase_type: str,
    team: str,
    match_id: Union[int, str]
) -> 'TrackingDataset':
    """
    Filter segment to frames in specified possession phase type.
    
    Phase types capture the tactical intent of possession:
    - 'build_up': Building from back
    - 'create': Creating chances in final third
    - 'finish': Finishing/shooting sequences
    - 'quick_break': Counter-attacks
    - 'transition': Transition play
    - 'direct': Direct/long ball play
    - 'chaotic': Chaotic/scrambled play
    - 'set_play': Set pieces
    
    Args:
        segment: TrackingDataset to filter
        phase_type: One of the phase types listed above
        team: 'home' or 'away' - which team's phases to filter
        match_id: Match identifier
    
    Returns:
        TrackingDataset containing only frames from specified phase type
        Returns empty segment if no phases match or phases data unavailable
    
    Example - Simple filtering:
        >>> import gamestate as gs
        >>> 
        >>> # Get time segment
        >>> segment = gs.segment_by_time_window(1886347, 0, 45)
        >>> 
        >>> # Filter to build-up phases
        >>> buildup = gs.filter_by_phase_type(segment, 'build_up', team='home', match_id=1886347)
        >>> 
        >>> if gs.has_frames(buildup):
        ...     positions = gs.average_positions(buildup, team='home', match_id=1886347)
        ...     print(f"Build-up: width={positions['avg_y'].max() - positions['avg_y'].min():.1f}m")
    
    Example - Compare phase types:
        >>> # Get full match
        >>> segment = gs.get_full_match(1886347)
        >>> 
        >>> # Compare different phases
        >>> for phase in ['build_up', 'create', 'finish', 'quick_break']:
        ...     filtered = gs.filter_by_phase_type(segment, phase, team='home', match_id=1886347)
        ...     if gs.has_frames(filtered):
        ...         comp = gs.team_compactness(filtered, team='home', match_id=1886347)
        ...         print(f"{phase}: width={comp['width']:.1f}m, depth={comp['depth']:.1f}m")
    
    Example - Stack with other filters:
        >>> # Start with time window
        >>> segment = gs.segment_by_time_window(1886347, 0, 45)
        >>> 
        >>> # Filter to winning
        >>> segment = gs.filter_by_game_state(segment, 'winning', team='home', match_id=1886347)
        >>> 
        >>> # Filter to build-up
        >>> segment = gs.filter_by_phase_type(segment, 'build_up', team='home', match_id=1886347)
        >>> 
        >>> # Result: "Build-up when winning in first half"
        >>> if gs.has_frames(segment):
        ...     positions = gs.average_positions(segment, team='home', match_id=1886347)
    
    Note:
        Requires phases CSV data from SkillCorner. If unavailable,
        returns empty segment with warning.
    """
    # Validate inputs
    valid_phases = [
        'build_up', 'create', 'finish', 'quick_break',
        'transition', 'direct', 'chaotic', 'set_play'
    ]
    if phase_type not in valid_phases:
        raise ValueError(
            f"phase_type must be one of {valid_phases}, got '{phase_type}'"
        )
    if team not in ['home', 'away']:
        raise ValueError(f"team must be 'home' or 'away', got '{team}'")
    
    # Load phases data
    phases_df = load_phases_data(match_id)
    
    if phases_df.empty:
        print(f"⚠️  No phases data available for match {match_id}")
        print(f"   Returning empty segment")
        return segment.filter(lambda f: False)
    
    # Get team ID
    team_id = phases_df[f'{team}_team.id'].iloc[0]
    
    # Filter to team's possession phases of specified type
    target_phases = phases_df[
        (phases_df['team_in_possession_id'] == team_id) &
        (phases_df['team_in_possession_phase_type'] == phase_type)
    ]
    
    if len(target_phases) == 0:
        print(f"⚠️  No {phase_type} phases found for {team} team")
        return segment.filter(lambda f: False)
    
    # Get frame ranges from phases
    target_frames = _extract_frame_set_from_phases(target_phases)
    
    # Filter segment to these frames
    filtered_segment = segment.filter(
        lambda frame: frame.frame_id in target_frames
    )
    
    return filtered_segment


# ----------------------------------------------------------
# Filter 2: By OOP (Out-of-Possession) Phase Type
# ----------------------------------------------------------

def filter_by_oop_phase(
    segment: 'TrackingDataset',
    phase_type: str,
    team: str,
    match_id: Union[int, str]
) -> 'TrackingDataset':
    """
    Filter segment to frames in specified defensive phase type.
    
    Defensive phase types capture how the team defends:
    - 'high_block': Pressing high up the pitch
    - 'medium_block': Mid-block defensive shape
    - 'low_block': Deep defensive block
    - 'defending_direct': Defending direct/long balls
    - 'defending_set_play': Defending set pieces
    - 'defending_transition': Defending transitions
    - 'defending_quick_break': Defending counter-attacks
    - 'chaotic': Chaotic defensive situations
    
    Args:
        segment: TrackingDataset to filter
        defensive_type: One of the defensive types listed above
        team: 'home' or 'away' - which team's defensive phases
        match_id: Match identifier
    
    Returns:
        TrackingDataset containing only frames from specified defensive phase
        Returns empty segment if no phases match or phases data unavailable
    
    Example - Analyze defensive line by block type:
        >>> import gamestate as gs
        >>> 
        >>> # Get full match
        >>> segment = gs.get_full_match(1886347)
        >>> 
        >>> # Compare defensive blocks
        >>> for block in ['high_block', 'medium_block', 'low_block']:
        ...     filtered = gs.filter_by_defensive_phase(segment, block, team='home', match_id=1886347)
        ...     if gs.has_frames(filtered):
        ...         line = gs.defensive_line_height(filtered, team='home', match_id=1886347)
        ...         print(f"{block}: line at {line['median_defensive_line_x']:.1f}m from goal")
    
    Example - Stack with time filter:
        >>> # First 45 minutes
        >>> segment = gs.segment_by_time_window(1886347, 0, 45)
        >>> 
        >>> # Filter to high block
        >>> high_block = gs.filter_by_defensive_phase(segment, 'high_block', team='home', match_id=1886347)
        >>> 
        >>> # Analyze
        >>> if gs.has_frames(high_block):
        ...     comp = gs.team_compactness(high_block, team='home', match_id=1886347)
        ...     line = gs.defensive_line_height(high_block, team='home', match_id=1886347)
    
    Note:
        When team is OUT of possession, they are the defending team.
        This filter gets frames where OPPONENT has the ball and TEAM is defending.
    """
    # Validate inputs
    valid_defensive = [
        'high_block', 'medium_block', 'low_block',
        'defending_direct', 'defending_set_play',
        'defending_transition', 'defending_quick_break',
        'chaotic'
    ]
    if phase_type not in valid_defensive:
        raise ValueError(
            f"phase_type must be one of {valid_defensive}, got '{phase_type}'"
        )
    if team not in ['home', 'away']:
        raise ValueError(f"team must be 'home' or 'away', got '{team}'")
    
    # Load phases data
    phases_df = load_phases_data(match_id)
    
    if phases_df.empty:
        print(f"⚠️  No phases data available for match {match_id}")
        print(f"   Returning empty segment")
        return segment.filter(lambda f: False)
    
    # Get team ID and opponent ID
    team_id = phases_df[f'{team}_team.id'].iloc[0]
    opponent_id = phases_df[f'{"away" if team == "home" else "home"}_team.id'].iloc[0]
    
    # Filter to phases where OPPONENT has possession and TEAM is defending
    # (team_out_of_possession_phase_type describes the team OUT of possession)
    target_phases = phases_df[
        (phases_df['team_in_possession_id'] == opponent_id) &
        (phases_df['team_out_of_possession_phase_type'] == phase_type)
    ]
    
    if len(target_phases) == 0:
        print(f"⚠️  No {phase_type} phases found for {team} team")
        return segment.filter(lambda f: False)
    
    # Get frame ranges from phases
    target_frames = _extract_frame_set_from_phases(target_phases)
    
    # Filter segment to these frames
    filtered_segment = segment.filter(
        lambda frame: frame.frame_id in target_frames
    )
    
    return filtered_segment


# ----------------------------------------------------------
# Filter 3: By Pitch Third
# ----------------------------------------------------------

def filter_by_pitch_third(segment, pitch_third, team, match_id):
    """
    Filter segment to phases that START in specified pitch third.
    
    Phases are classified by where they START, even if they progress
    into other thirds during the phase. For example, a build-up starting
    in the defensive third that progresses to the attacking third will
    have all its frames included when filtering for 'defensive'.
    
    Args:
        segment: TrackingDataset to filter
        pitch_third: Pitch zone - 'defensive', 'middle', or 'attacking'
        team: 'home' or 'away'
        match_id: Match identifier
    
    Returns:
        TrackingDataset: Filtered segment containing only frames from phases
                        that started in the specified third
    
    Example:
        >>> # Get phases starting in defensive third
        >>> segment = segment_by_time_window(1886347, 0, 45)
        >>> segment = filter_by_pitch_third(segment, pitch_third='defensive', 
        ...                                 team='home', match_id=1886347)
        >>> print(f"Frames: {len(segment.records)}")
        
        >>> # Combine with phase filter
        >>> segment = filter_by_ip_phase(segment, 'build_up', 'home', 1886347)
        >>> segment = filter_by_pitch_third(segment, pitch_third='defensive',
        ...                                 team='home', match_id=1886347)
        >>> # Now have: build-up phases starting in defensive third
    
    Note:
        Uses 'third_start' column from phases data, which indicates where
        the phase originated. Phases can span multiple thirds.
    """
    
    # Validate input
    if not has_frames(segment):
        return segment
    
    valid_thirds = ['defensive', 'middle', 'attacking']
    if pitch_third not in valid_thirds:
        raise ValueError(
            f"pitch_third must be one of {valid_thirds}, got '{pitch_third}'"
        )
    
    # Load phases data
    phases_df = load_phases_data(match_id)
    if phases_df.empty:
        print("⚠️  No phases data available")
        return segment.filter(lambda f: False)
    
    # Get team IDs
    dataset = load_tracking_dataset(match_id)
    home_team_id = dataset.metadata.teams[0].team_id
    away_team_id = dataset.metadata.teams[1].team_id
    team_id = home_team_id if team == 'home' else away_team_id
    other_team_id = away_team_id if team == 'home' else home_team_id
    
    # Map user-friendly names to data values
    third_mapping = {
        'defensive': 'defensive_third',
        'middle': 'middle_third',
        'attacking': 'attacking_third'
    }
    
    third_value = third_mapping[pitch_third]
    
    # Filter to phases that START in specified third for specified team
    target_phases = phases_df[
        (phases_df['team_in_possession_id'] == team_id) &
        (phases_df['third_start'] == third_value)
    ]
    
    if target_phases.empty:
        print(f"⚠️  No phases starting in {pitch_third} third for {team} team")
        return segment.filter(lambda f: False)
    
    print(f"✅ Found {len(target_phases)} phases starting in {pitch_third} third for {team} team")
    
    # Extract target frames
    target_frames = set()
    for _, phase in target_phases.iterrows():
        frame_start = phase['frame_start']
        frame_end = phase['frame_end']
        target_frames.update(range(frame_start, frame_end + 1))
    
    # Check for overlaps with other team's phases (same third)
    # This handles the edge case where both teams have phases starting 
    # in the same third at overlapping times
    other_phases = phases_df[
        (phases_df['team_in_possession_id'] == other_team_id) &
        (phases_df['third_start'] == third_value)
    ]
    
    if not other_phases.empty:
        # Extract other team frames
        other_frames = set()
        for _, phase in other_phases.iterrows():
            other_frames.update(range(phase['frame_start'], phase['frame_end'] + 1))
        
        # Find ambiguous frames
        ambiguous_frames = target_frames.intersection(other_frames)
        
        if ambiguous_frames:
            print(f"⚠️  Found {len(ambiguous_frames)} overlapping frames (excluding them)")
            # Remove ambiguous frames
            target_frames = target_frames - ambiguous_frames
    
    if not target_frames:
        print(f"⚠️  No frames remaining after removing overlaps")
        return segment.filter(lambda f: False)
    
    # Filter segment
    filtered_segment = segment.filter(lambda f: f.frame_id in target_frames)
    
    return filtered_segment


# ----------------------------------------------------------
# Filter 4: By Game State
# ----------------------------------------------------------

def filter_by_game_state(
    segment: 'TrackingDataset',
    state: str,
    team: str,
    match_id: Union[int, str]
) -> 'TrackingDataset':
    """
    Filter segment to frames where team is in specified game state.
    
    Game states from team's perspective:
    - 'winning': Team is ahead on score
    - 'drawing': Score is level
    - 'losing': Team is behind on score
    
    Args:
        segment: TrackingDataset to filter
        state: 'winning', 'drawing', or 'losing'
        team: 'home' or 'away' - which team's perspective
        match_id: Match identifier
    
    Returns:
        TrackingDataset containing only frames in specified game state
        Returns empty segment if no frames match
    
    Example - Compare tactics by game state:
        >>> import gamestate as gs
        >>> 
        >>> # Get full match
        >>> segment = gs.get_full_match(1886347)
        >>> 
        >>> # Compare game states
        >>> for state in ['winning', 'drawing', 'losing']:
        ...     filtered = gs.filter_by_game_state(segment, state, team='home', match_id=1886347)
        ...     if gs.has_frames(filtered):
        ...         comp = gs.team_compactness(filtered, team='home', match_id=1886347)
        ...         print(f"When {state}: width={comp['width']:.1f}m")
    
    Example - Build-up when losing:
        >>> # Get time window
        >>> segment = gs.segment_by_time_window(1886347, 0, 90)
        >>> 
        >>> # Filter to losing
        >>> segment = gs.filter_by_game_state(segment, 'losing', team='home', match_id=1886347)
        >>> 
        >>> # Filter to build-up
        >>> segment = gs.filter_by_phase_type(segment, 'build_up', team='home', match_id=1886347)
        >>> 
        >>> # Analyze: "How does team build up when losing?"
        >>> if gs.has_frames(segment):
        ...     positions = gs.average_positions(segment, team='home', match_id=1886347)
    
    Note:
        Uses event data (scores) to determine game state.
        State changes at exact frame where goal is scored.
    """
    # Validate inputs
    if state not in ['winning', 'drawing', 'losing']:
        raise ValueError(
            f"state must be 'winning', 'drawing', or 'losing', got '{state}'"
        )
    if team not in ['home', 'away']:
        raise ValueError(f"team must be 'home' or 'away', got '{team}'")
    
    # Load event data
    events_df = load_event_data(match_id)
    
    if events_df.empty:
        # No events, entire match is drawing
        if state == 'drawing':
            return segment
        else:
            return segment.filter(lambda f: False)
    
    # Determine frame ranges for each game state
    state_windows = _get_game_state_windows(events_df, team)
    
    # Get frames for requested state
    target_frames = set()
    for window in state_windows:
        if window['state'] == state:
            target_frames.update(range(window['start_frame'], window['end_frame'] + 1))
    
    if not target_frames:
        print(f"⚠️  No frames found for state '{state}' (team={team})")
        return segment.filter(lambda f: False)
    
    # Filter segment to these frames
    filtered_segment = segment.filter(
        lambda frame: frame.frame_id in target_frames
    )
    
    return filtered_segment


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def _extract_frame_set_from_phases(phases_df: pd.DataFrame) -> Set[int]:
    """
    Extract set of frame IDs from phases DataFrame.
    
    Args:
        phases_df: DataFrame with frame_start and frame_end columns
    
    Returns:
        Set of frame IDs covered by these phases
    """
    target_frames = set()
    
    for _, phase in phases_df.iterrows():
        frame_start = int(phase['frame_start'])
        frame_end = int(phase['frame_end'])
        target_frames.update(range(frame_start, frame_end + 1))
    
    return target_frames


def _get_game_state_windows(
    events_df: pd.DataFrame,
    team: str
) -> list:
    """
    Internal function to determine game state windows from events.
    
    Args:
        events_df: Events DataFrame with scores
        team: 'home' or 'away'
    
    Returns:
        List of dicts with: state, start_frame, end_frame, home_score, away_score
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