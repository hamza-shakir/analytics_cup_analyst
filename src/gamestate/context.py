"""
context.py - Match Context Functions (Kloppy Native)

Provides high-level match context:
- summary: Teams, scores, possession, substitutions
- score_progression: Goal timeline
- substitutions: Player changes

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
    - n_subs
    - possesion_pct
    
    Args:
        match_id: Match identifier
    
    Returns:
        pandas.DataFrame: Two-row summary (home, away)
    
    Example:
        >>> summary_df = summary(1886347)
        >>> print(summary_df)
          home_away_status  team_id      team_name  goals_scored  n_subs  possesion_pct
        0             Home     1805    Auckland FC             2       3             45
        1             Away     1629   Newcastle...             0       2             55
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
    
    # Substitutions count
    subs_df = substitutions(match_id)
    if not subs_df.empty:
        n_subs_home = int(subs_df[subs_df['team_name'] == home_team.name]['sub_in_id'].notna().sum())
        n_subs_away = int(subs_df[subs_df['team_name'] == away_team.name]['sub_in_id'].notna().sum())
    else:
        n_subs_home = 0
        n_subs_away = 0
    
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
            'n_subs': n_subs_home,
            'possesion_pct': poss_home_pct,
        },
        {
            'home_away_status': 'Away',
            'team_id': away_team.team_id,
            'team_name': away_team.name,
            'goals_scored': final_away,
            'n_subs': n_subs_away,
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


def substitutions(match_id):
    """
    Detect substitutions by analyzing when players appear/disappear from frames.
    
    Returns DataFrame with substitution information:
    - team_name
    - sub_minute (estimated)
    - sub_out_id, sub_out_name
    - sub_in_id, sub_in_name
    
    Args:
        match_id: Match identifier
    
    Returns:
        pandas.DataFrame: Substitution events
    
    Example:
        >>> subs = substitutions(1886347)
        >>> print(subs[['team_name', 'sub_minute', 'sub_out_name', 'sub_in_name']])
          team_name  sub_minute  sub_out_name  sub_in_name
        0  Auckland FC          65   Player A    Player B
    
    Note:
        Detection is based on frame presence. A substitution is inferred when:
        - A player stops appearing in frames (sub out)
        - A new player starts appearing in frames (sub in)
        Around the same time for the same team.
    """
    # Load dataset
    dataset = load_tracking_dataset(match_id)
    wide_df = dataset.to_df()
    
    # Get team info
    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]
    
    # Build player metadata lookup: player_id -> team and info
    player_info = {}
    for team in [home_team, away_team]:
        for player in team.players:
            player_info[str(player.player_id)] = {
                'team_name': team.name,
                'team_id': team.team_id,
                'name': player.name,
                'jersey_no': player.jersey_no,
            }
    
    # Find player appearance windows
    # For each player, find first and last frame they appear
    player_windows = []
    
    for col in wide_df.columns:
        if col.endswith('_x'):
            player_id = col.replace('_x', '')
            
            if not player_id.isdigit():
                continue
            
            # Find frames where player is present
            present_frames = wide_df[wide_df[col].notna()]['frame_id'].values
            
            if len(present_frames) == 0:
                continue
            
            first_frame = int(present_frames[0])
            last_frame = int(present_frames[-1])
            total_frames = len(present_frames)
            
            # Get player info
            info = player_info.get(player_id, {})
            
            player_windows.append({
                'player_id': player_id,
                'team_name': info.get('team_name'),
                'name': info.get('name'),
                'jersey_no': info.get('jersey_no'),
                'first_frame': first_frame,
                'last_frame': last_frame,
                'total_frames': total_frames
            })
    
    players_df = pd.DataFrame(player_windows)
    
    if players_df.empty:
        return pd.DataFrame(columns=[
            "team_name", "sub_minute", "sub_out_id", "sub_out_name",
            "sub_in_id", "sub_in_name"
        ])
    
    # Identify subs: players who didn't play full match
    max_frames = players_df['total_frames'].max()
    
    # Sub out: started from frame 0-100 but didn't finish
    sub_outs = players_df[
        (players_df['first_frame'] < 100) & 
        (players_df['total_frames'] < max_frames * 0.95)
    ].copy()
    
    # Sub in: didn't start from beginning
    sub_ins = players_df[
        players_df['first_frame'] > 100
    ].copy()
    
    # Estimate sub minute from frame (frame_rate is in metadata)
    frame_rate = dataset.metadata.frame_rate
    sub_outs['sub_minute'] = (sub_outs['last_frame'] / frame_rate / 60).round(0).astype(int)
    sub_ins['sub_minute'] = (sub_ins['first_frame'] / frame_rate / 60).round(0).astype(int)
    
    # Pair subs by team and similar timing
    paired_subs = []
    
    for team in players_df['team_name'].unique():
        if pd.isna(team):
            continue
        
        team_outs = sub_outs[sub_outs['team_name'] == team].sort_values('sub_minute')
        team_ins = sub_ins[sub_ins['team_name'] == team].sort_values('sub_minute')
        
        # Simple pairing: match by order and proximity
        for i, out_row in team_outs.iterrows():
            # Find closest sub_in within ±5 minutes
            candidates = team_ins[
                (team_ins['sub_minute'] >= out_row['sub_minute'] - 5) &
                (team_ins['sub_minute'] <= out_row['sub_minute'] + 5)
            ]
            
            if not candidates.empty:
                in_row = candidates.iloc[0]
                
                paired_subs.append({
                    'team_name': team,
                    'sub_minute': out_row['sub_minute'],
                    'sub_out_id': out_row['player_id'],
                    'sub_out_name': out_row['name'],
                    'sub_in_id': in_row['player_id'],
                    'sub_in_name': in_row['name'],
                })
                
                # Remove matched sub_in to avoid duplicate pairing
                team_ins = team_ins[team_ins['player_id'] != in_row['player_id']]
        
        # Add unpaired outs
        for i, out_row in team_outs.iterrows():
            if not any(s['sub_out_id'] == out_row['player_id'] for s in paired_subs):
                paired_subs.append({
                    'team_name': team,
                    'sub_minute': out_row['sub_minute'],
                    'sub_out_id': out_row['player_id'],
                    'sub_out_name': out_row['name'],
                    'sub_in_id': None,
                    'sub_in_name': None,
                })
        
        # Add unpaired ins
        for i, in_row in team_ins.iterrows():
            paired_subs.append({
                'team_name': team,
                'sub_minute': in_row['sub_minute'],
                'sub_out_id': None,
                'sub_out_name': None,
                'sub_in_id': in_row['player_id'],
                'sub_in_name': in_row['name'],
            })
    
    if not paired_subs:
        return pd.DataFrame(columns=[
            "team_name", "sub_minute", "sub_out_id", "sub_out_name",
            "sub_in_id", "sub_in_name"
        ])
    
    result = pd.DataFrame(paired_subs).sort_values(['team_name', 'sub_minute']).reset_index(drop=True)
    
    return result