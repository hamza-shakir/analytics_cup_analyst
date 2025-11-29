"""
context.py - Match Context Functions (Kloppy Native)

Provides high-level match context:
- summary: Teams, scores, possession
- score_progression: Goal timeline

Uses kloppy datasets for cleaner, more efficient code.
"""

import pandas as pd
import numpy as np
from .load_data import (
    load_tracking_dataset,
    load_event_data,
    get_metadata
)


def summary(match_id):
    """
    Return two-row summary table (home and away) with match context.
    
    Columns:
    - home_away_status: 'Home' or 'Away'
    - team_id
    - team_name
    - goals_scored
    - possesion_pct
    
    Args:
        match_id: Match identifier
    
    Returns:
        pandas.DataFrame: Two-row summary (home, away)
    """
    # Load data
    dataset = load_tracking_dataset(match_id)
    events_df = load_event_data(match_id)
    
    # Get team info from kloppy metadata
    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]
    
    # Goals scored: get final score from events
    if not events_df.empty and 'home_team_score' in events_df.columns:
        final_home = int(events_df['home_team_score'].iloc[-1])
        final_away = int(events_df['away_team_score'].iloc[-1])
    else:
        final_home = 0
        final_away = 0
    
    # Possession: count frames by ball_owning_team
    poss_home = sum(1 for frame in dataset.records 
                    if frame.ball_owning_team == home_team)
    poss_away = sum(1 for frame in dataset.records 
                    if frame.ball_owning_team == away_team)
    total_frames = len(dataset.records)
    
    if total_frames > 0:
        poss_home_pct = round(100 * poss_home / total_frames)
        poss_away_pct = round(100 * poss_away / total_frames)
    else:
        poss_home_pct = 0
        poss_away_pct = 0
    
    # Build result DataFrame
    summary_df = pd.DataFrame([
        {
            'home_away_status': 'Home',
            'team_id': home_team.team_id,
            'team_name': home_team.name,
            'goals_scored': final_home,
            'possesion_pct': poss_home_pct,
        },
        {
            'home_away_status': 'Away',
            'team_id': away_team.team_id,
            'team_name': away_team.name,
            'goals_scored': final_away,
            'possesion_pct': poss_away_pct,
        }
    ])
    
    return summary_df


def score_progression(match_id):
    """
    Get timeline of when goals were scored.
    
    Returns DataFrame showing score changes throughout the match.
    
    Args:
        match_id: Match identifier
    
    Returns:
        pandas.DataFrame: Goal events with scores and timing
    
    Example:
        >>> goals = score_progression(1886347)
        >>> print(goals)
          home_team.name  home_team_score  away_team_score  away_team.name  minute_start  frame_start
        0    Auckland FC                0                0   Newcastle...             0           28
        1    Auckland FC                1                0   Newcastle...            83        50600
        2    Auckland FC                2                0   Newcastle...            88        53812
    """
    # Load event data
    events_df = load_event_data(match_id)
    
    if events_df.empty:
        return pd.DataFrame()
    
    # Get team names from metadata
    meta = get_metadata(match_id)
    
    # Select relevant columns
    columns = [
        "home_team.name",
        "home_team_score",
        "away_team_score",
        "away_team.name",
        "minute_start",
        "frame_start"
    ]
    
    # Filter to columns that exist
    available_cols = [col for col in columns if col in events_df.columns]
    
    # Find score changes: keep first row (0-0) and rows where score changes
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
    
    result = df_from_start[mask][available_cols].reset_index(drop=True)
    
    return result
