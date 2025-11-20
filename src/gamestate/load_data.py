"""
load_data.py - Kloppy-native data loading

This module loads SkillCorner tracking and event data using kloppy.
Returns kloppy dataset objects for maximum flexibility.
Provides helpers to convert to pandas when needed.
"""

import pandas as pd
import numpy as np
from kloppy import skillcorner
from kloppy.domain import Orientation


# ----------------------------------------------------------
# Module-level cache
# ----------------------------------------------------------
_dataset_cache = {}
_event_cache = {}


# ----------------------------------------------------------
# Core Loading Functions
# ----------------------------------------------------------

def load_tracking_dataset(match_id, only_alive=True, normalize_orientation=False):
    """
    Load SkillCorner tracking data using kloppy.
    
    Args:
        match_id: Match identifier (int or str)
        only_alive: If True, only include frames where ball is in play
        normalize_orientation: If True, transform so home team always attacks left->right
    
    Returns:
        kloppy.TrackingDataset: Dataset with frames, metadata, etc.
    
    Example:
        >>> dataset = load_tracking_dataset(1886347)
        >>> print(f"Loaded {len(dataset.records)} frames")
        >>> print(f"Home: {dataset.metadata.teams[0].name}")
    """
    cache_key = f"{match_id}_{only_alive}_{normalize_orientation}"
    
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]
    
    # Load with kloppy
    dataset = skillcorner.load_open_data(
        match_id=str(match_id),
        only_alive=only_alive,
        coordinates="skillcorner"  # Keep SkillCorner's native coordinates
    )
    
    # Normalize orientation if requested
    if normalize_orientation:
        dataset = dataset.transform(
            to_orientation=Orientation.HOME_AWAY
        )
    
    _dataset_cache[cache_key] = dataset
    return dataset


def load_event_data(match_id):
    """
    Load SkillCorner dynamic events CSV.
    
    Args:
        match_id: Match identifier
    
    Returns:
        pandas.DataFrame: Events with scores, game states, etc.
    
    Note:
        This loads the dynamic_events.csv which contains score changes
        and game state information. Not full event data.
    """
    if match_id in _event_cache:
        return _event_cache[match_id]
    
    event_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"
    
    try:
        events_df = pd.read_csv(event_url)
    except:
        print(f"Warning: Could not load events for match {match_id}")
        return pd.DataFrame()
    
    # Get team info from tracking dataset
    dataset = load_tracking_dataset(match_id)
    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]
    
    # Add team identifiers
    events_df['home_team.id'] = home_team.team_id
    events_df['away_team.id'] = away_team.team_id
    events_df['home_team.name'] = home_team.name
    events_df['away_team.name'] = away_team.name
    
    # Create home/away scores
    events_df['home_team_score'] = events_df.apply(
        lambda row: row['team_score'] if row['team_id'] == home_team.team_id else row['opponent_team_score'],
        axis=1
    )
    events_df['away_team_score'] = events_df.apply(
        lambda row: row['opponent_team_score'] if row['team_id'] == home_team.team_id else row['team_score'],
        axis=1
    )
    
    _event_cache[match_id] = events_df
    return events_df


# ----------------------------------------------------------
# Metadata Access Helpers
# ----------------------------------------------------------

def get_metadata(match_id):
    """
    Get match metadata in easily accessible format.
    
    Returns:
        dict: Contains teams, players, frame_rate, etc.
    
    Example:
        >>> meta = get_metadata(1886347)
        >>> print(meta['home_team']['name'])
        >>> print(f"Frame rate: {meta['frame_rate']} fps")
    """
    dataset = load_tracking_dataset(match_id)
    metadata = dataset.metadata
    
    return {
        'home_team': {
            'id': metadata.teams[0].team_id,
            'name': metadata.teams[0].name,
            'players': [
                {
                    'id': p.player_id,
                    'name': p.name,
                    'full_name': p.full_name,
                    'jersey_no': p.jersey_no,
                    'position': str(p.starting_position) if p.starting_position else None,
                    'is_gk': p.starting_position and 'Goalkeeper' in str(p.starting_position)
                }
                for p in metadata.teams[0].players
            ]
        },
        'away_team': {
            'id': metadata.teams[1].team_id,
            'name': metadata.teams[1].name,
            'players': [
                {
                    'id': p.player_id,
                    'name': p.name,
                    'full_name': p.full_name,
                    'jersey_no': p.jersey_no,
                    'position': str(p.starting_position) if p.starting_position else None,
                    'is_gk': p.starting_position and 'Goalkeeper' in str(p.starting_position)
                }
                for p in metadata.teams[1].players
            ]
        },
        'frame_rate': metadata.frame_rate,
        'total_frames': len(dataset.records),
        'duration_minutes': len(dataset.records) / metadata.frame_rate / 60
    }


def list_players(match_id, team=None):
    """
    Get list of players in the match.
    
    Args:
        match_id: Match identifier
        team: 'home', 'away', or None (both teams)
    
    Returns:
        pandas.DataFrame: Player information
    
    Example:
        >>> players = list_players(1886347, team='home')
        >>> print(players[['jersey_no', 'name', 'position']])
    """
    meta = get_metadata(match_id)
    
    players = []
    
    if team in [None, 'home']:
        for p in meta['home_team']['players']:
            players.append({
                **p,
                'team': 'home',
                'team_name': meta['home_team']['name'],
                'team_id': meta['home_team']['id']
            })
    
    if team in [None, 'away']:
        for p in meta['away_team']['players']:
            players.append({
                **p,
                'team': 'away',
                'team_name': meta['away_team']['name'],
                'team_id': meta['away_team']['id']
            })
    
    return pd.DataFrame(players)


# ----------------------------------------------------------
# DataFrame Conversion Helpers
# ----------------------------------------------------------

def to_wide_dataframe(dataset):
    """
    Convert kloppy dataset to wide-format DataFrame.
    
    Wide format: 1 row per frame, columns for each player (player_id_x, player_id_y)
    
    Args:
        dataset: kloppy TrackingDataset
    
    Returns:
        pandas.DataFrame: Wide format tracking data
    
    Note:
        This is kloppy's native format. Most efficient for frame-based operations.
    """
    return dataset.to_df()


def to_long_dataframe(dataset, match_id):
    """
    Convert kloppy dataset to long-format DataFrame.
    
    Long format: 1 row per player per frame
    Similar to your original enriched_tracking_data format.
    
    Args:
        dataset: kloppy TrackingDataset
        match_id: Match identifier
    
    Returns:
        pandas.DataFrame: Long format with columns like x, y, player_id, etc.
    
    Note:
        Use this if you need player-centric analysis or compatibility with
        existing code that expects long format.
    """
    wide_df = dataset.to_df()
    
    # Extract player IDs from columns
    player_columns = {}
    for col in wide_df.columns:
        if '_x' in col and col.replace('_x', '').isdigit():
            player_id = col.replace('_x', '')
            player_columns[player_id] = {
                'x': col,
                'y': f'{player_id}_y',
                'd': f'{player_id}_d',
                's': f'{player_id}_s'
            }
    
    # Get metadata
    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]
    
    # Build player metadata lookup
    player_meta = {}
    for team in dataset.metadata.teams:
        for player in team.players:
            player_meta[str(player.player_id)] = {
                'player_id': player.player_id,
                'short_name': player.name,
                'number': player.jersey_no,
                'team_id': team.team_id,
                'team_name': team.name,
                'is_gk': player.starting_position and 'Goalkeeper' in str(player.starting_position),
                'position': str(player.starting_position) if player.starting_position else None,
            }
    
    # Convert to long format
    long_rows = []
    
    for idx, row in wide_df.iterrows():
        frame_data = {
            'frame': row['frame_id'],
            'timestamp': row['timestamp'],
            'period': row['period_id'],
            'ball_x': row['ball_x'],
            'ball_y': row['ball_y'],
            'ball_z': row.get('ball_z'),
            'ball_state': row.get('ball_state'),
        }
        
        # Possession group
        if pd.notna(row.get('ball_owning_team_id')):
            if str(row['ball_owning_team_id']) == str(home_team.team_id):
                frame_data['possession_group'] = 'home team'
            elif str(row['ball_owning_team_id']) == str(away_team.team_id):
                frame_data['possession_group'] = 'away team'
        else:
            frame_data['possession_group'] = None
        
        # Add row for each visible player
        for player_id, cols in player_columns.items():
            if pd.notna(row.get(cols['x'])):
                player_row = frame_data.copy()
                player_row.update({
                    'player_id': player_id,
                    'x': row[cols['x']],
                    'y': row[cols['y']],
                    'is_detected': True,
                })
                
                # Add player metadata
                if player_id in player_meta:
                    player_row.update(player_meta[player_id])
                
                long_rows.append(player_row)
    
    long_df = pd.DataFrame(long_rows)
    
    # Add match metadata
    long_df['match_id'] = match_id
    long_df['home_team.name'] = home_team.name
    long_df['away_team.name'] = away_team.name
    long_df['home_team.id'] = home_team.team_id
    long_df['away_team.id'] = away_team.team_id
    
    return long_df


# ----------------------------------------------------------
# Quick Info Functions
# ----------------------------------------------------------

def print_match_info(match_id):
    """
    Print a quick summary of the match.
    
    Example:
        >>> print_match_info(1886347)
        Match: Auckland FC vs Newcastle United Jets FC
        Frames: 31,500 (21.0 minutes)
        Frame rate: 25.0 fps
        Home players: 14
        Away players: 15
    """
    meta = get_metadata(match_id)
    
    print(f"\n{'='*60}")
    print(f"MATCH INFO - ID: {match_id}")
    print(f"{'='*60}")
    print(f"Match: {meta['home_team']['name']} vs {meta['away_team']['name']}")
    print(f"Frames: {meta['total_frames']:,} ({meta['duration_minutes']:.1f} minutes)")
    print(f"Frame rate: {meta['frame_rate']} fps")
    print(f"Home players: {len(meta['home_team']['players'])}")
    print(f"Away players: {len(meta['away_team']['players'])}")
    print(f"{'='*60}\n")


# ----------------------------------------------------------
# Backward Compatibility (Optional)
# ----------------------------------------------------------

def load_enriched_tracking_data(match_id):
    """
    Backward compatible function that returns long-format DataFrame.
    
    This matches your original function signature for easy migration.
    
    Args:
        match_id: Match identifier
    
    Returns:
        pandas.DataFrame: Long format tracking data
    """
    dataset = load_tracking_dataset(match_id)
    return to_long_dataframe(dataset, match_id)