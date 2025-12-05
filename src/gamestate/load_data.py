"""
load_data.py - Kloppy-native data loading WITH PHASES SUPPORT

This module loads SkillCorner tracking, event, and phases data using kloppy.
Returns kloppy dataset objects for maximum flexibility.
Provides helpers to convert to pandas when needed.
"""

import pandas as pd
import numpy as np
import requests
from kloppy import skillcorner
from kloppy.domain import Orientation


# ----------------------------------------------------------
# Module-level cache
# ----------------------------------------------------------
_dataset_cache = {}
_event_cache = {}
_phases_cache = {}  # NEW: Cache for phases data


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
    
    # Try loading with different methods for compatibility
    dataset = None
    last_error = None
    
    # Method 1: Without coordinates parameter (newer kloppy versions)
    try:
        dataset = skillcorner.load_open_data(
            match_id=str(match_id),
            only_alive=only_alive
        )
    except Exception as e:
        last_error = e
        
        # Method 2: With coordinates parameter (older kloppy versions)
        try:
            dataset = skillcorner.load_open_data(
                match_id=str(match_id),
                only_alive=only_alive,
                coordinates="skillcorner"
            )
            last_error = None
        except Exception as e2:
            last_error = e2
    
    # If both methods failed, raise clear error
    if dataset is None:
        error_msg = f"""
Failed to load tracking data for match {match_id}.

Error: {str(last_error)}

Possible solutions:
1. Update kloppy: pip install --upgrade kloppy
2. Check match ID exists: https://github.com/SkillCorner/opendata
3. Check your internet connection

Kloppy version issues:
- If using kloppy >= 3.0: Should work without 'coordinates' parameter
- If using kloppy < 3.0: May need 'coordinates="skillcorner"' parameter
"""
        raise Exception(error_msg)
    
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


def load_phases_data(match_id):
    """
    Load SkillCorner phases-of-play CSV.
    
    Args:
        match_id: Match identifier
    
    Returns:
        pandas.DataFrame: Phases with tactical context, spatial data, outcomes
    
    Columns include:
        Possession:
            - team_in_possession_id, team_in_possession_shortname
        
        Phase types:
            - team_in_possession_phase_type: build_up, create, finish, direct, 
              chaotic, set_play, transition, quick_break
            - team_out_of_possession_phase_type: high_block, medium_block, low_block,
              defending_direct, defending_set_play, defending_transition, 
              defending_quick_break, chaotic
        
        Spatial data:
            - x_start, x_end, y_start, y_end
            - channel_id_start, channel_id_end, channel_start, channel_end
            - third_id_start, third_id_end, third_start, third_end
            - penalty_area_start, penalty_area_end
        
        Team shape:
            - team_in_possession_width_start/end
            - team_in_possession_length_start/end
            - team_out_of_possession_width_start/end
            - team_out_of_possession_length_start/end
        
        Temporal:
            - frame_start, frame_end
            - time_start, time_end
            - minute_start, second_start
            - duration
            - period
        
        Outcomes:
            - team_possession_loss_in_phase (bool)
            - team_possession_lead_to_goal (bool)
            - team_possession_lead_to_shot (bool)
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Load phases data
        >>> phases = gs.load_phases_data(1886347)
        >>> print(f"Loaded {len(phases)} possession phases")
        >>> 
        >>> # Get all build-up phases for home team
        >>> home_team_id = phases['home_team.id'].iloc[0]
        >>> home_buildups = phases[
        ...     (phases['team_in_possession_id'] == home_team_id) &
        ...     (phases['team_in_possession_phase_type'] == 'build_up')
        ... ]
        >>> print(f"Home team build-up phases: {len(home_buildups)}")
        >>> 
        >>> # Analyze defensive phases
        >>> high_blocks = phases[
        ...     phases['team_out_of_possession_phase_type'] == 'high_block'
        ... ]
        >>> print(f"High block defensive phases: {len(high_blocks)}")
    
    Note:
        Phases data may not be available for all matches.
        If unavailable, returns empty DataFrame with warning.
    """
    # Check if already cached
    if match_id in _phases_cache:
        return _phases_cache[match_id]
    
    # Try loading from SkillCorner opendata
    phases_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_phases_of_play.csv"
    
    try:
        phases_df = pd.read_csv(phases_url)
        print(f"✅ Loaded {len(phases_df)} phases for match {match_id}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load phases data for match {match_id}")
        print(f"   Error: {str(e)[:100]}")
        print(f"   Note: Phases data may not be available for all matches")
        print(f"   Phase-based filters will return empty segments for this match")
        return pd.DataFrame()
    
    # Get team info from tracking dataset
    try:
        dataset = load_tracking_dataset(match_id)
        home_team = dataset.metadata.teams[0]
        away_team = dataset.metadata.teams[1]
        
        # Add team identifiers for easier filtering
        phases_df['home_team.id'] = home_team.team_id
        phases_df['away_team.id'] = away_team.team_id
        phases_df['home_team.name'] = home_team.name
        phases_df['away_team.name'] = away_team.name
        
        print(f"   Home: {home_team.name}")
        print(f"   Away: {away_team.name}")
    except Exception as e:
        print(f"⚠️  Warning: Could not add team metadata: {e}")
    
    # Cache and return
    _phases_cache[match_id] = phases_df
    return phases_df


# ----------------------------------------------------------
# Metadata Access Helpers
# ----------------------------------------------------------

def get_metadata(match_id):
    """
    Get match metadata in easily accessible format.
    
    Uses hybrid approach:
    - SkillCorner JSON for reliable player information (especially GK detection)
    - Kloppy dataset for accurate frame_rate and frame counts
    
    Returns:
        dict: Contains teams, players, frame_rate, etc.
    
    Example:
        >>> meta = get_metadata(1886347)
        >>> print(meta['home_team']['name'])
        >>> print(f"Frame rate: {meta['frame_rate']} fps")
        >>> 
        >>> # Get GK for home team
        >>> home_gk = [p for p in meta['home_team']['players'] if p['is_gk']]
        >>> print(f"Goalkeeper: {home_gk[0]['name']} (#{home_gk[0]['jersey_no']})")
    """
    # ========== LOAD KLOPPY DATASET FOR FRAME INFO ==========
    dataset = load_tracking_dataset(match_id)
    metadata = dataset.metadata
    # ========================================================
    
    # ========== LOAD SKILLCORNER JSON FOR PLAYER DATA ==========
    meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_match.json"
    
    # Read the JSON data
    response = requests.get(meta_data_github_url)
    raw_match_data = response.json()
    
    # Process nested JSON elements
    raw_match_df = pd.json_normalize(raw_match_data, max_level=2)
    raw_match_df["home_team_side"] = raw_match_df["home_team_side"].astype(str)
    
    # Extract players data
    players_df = pd.json_normalize(
        raw_match_df.to_dict("records"),
        record_path="players",
        meta=[
            "home_team.name",
            "home_team.id",
            "away_team.name",
            "away_team.id",
        ],
    )
    
    # Create GK flag - RELIABLE!
    players_df["is_gk"] = players_df["player_role.acronym"] == "GK"
    # ===========================================================
    
    # Separate home and away players
    home_team_id = raw_match_data["home_team"]["id"]
    away_team_id = raw_match_data["away_team"]["id"]
    
    home_players_df = players_df[players_df["team_id"] == home_team_id]
    away_players_df = players_df[players_df["team_id"] == away_team_id]
    
    # Helper function to convert player row to dict
    def player_to_dict(row):
        return {
            'id': int(row['id']),
            'name': row['short_name'],
            'full_name': f"{row['first_name']} {row['last_name']}",
            'jersey_no': int(row['number']),
            'position': row['player_role.name'] if pd.notna(row.get('player_role.name')) else None,
            'is_gk': bool(row['is_gk'])  # ← RELIABLE from SkillCorner JSON!
        }
    
    return {
        'home_team': {
            'id': home_team_id,
            'name': raw_match_data["home_team"]["name"],
            'players': [player_to_dict(row) for _, row in home_players_df.iterrows()]
        },
        'away_team': {
            'id': away_team_id,
            'name': raw_match_data["away_team"]["name"],
            'players': [player_to_dict(row) for _, row in away_players_df.iterrows()]
        },
        'frame_rate': metadata.frame_rate,  # ← From Kloppy (accurate)
        'total_frames': len(dataset.records),  # ← From Kloppy (accurate)
        'duration_minutes': len(dataset.records) / metadata.frame_rate / 60  # ← Calculated from Kloppy
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
        
        COORDINATES: Automatically scales from normalized (0-1) to meters.
        
        GK & POSITION: Uses SkillCorner JSON for reliable data (same as get_metadata).
    """
    wide_df = dataset.to_df()
    
    # Check if coordinates are normalized and get pitch dimensions
    pitch_length = 105  # Default SkillCorner
    pitch_width = 68
    coordinates_are_normalized = False
    
    if hasattr(dataset.metadata, 'coordinate_system'):
        coord_sys = dataset.metadata.coordinate_system
        if hasattr(coord_sys, 'pitch_dimensions'):
            dims = coord_sys.pitch_dimensions
            if hasattr(dims, 'pitch_length'):
                pitch_length = dims.pitch_length
            if hasattr(dims, 'pitch_width'):
                pitch_width = dims.pitch_width
            if hasattr(dims, 'unit') and 'NORMED' in str(dims.unit):
                coordinates_are_normalized = True
    
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
    
    # Get metadata from kloppy
    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]
    
    # ========== GET RELIABLE is_gk AND position FROM SKILLCORNER JSON ==========
    skillcorner_player_data = {}  # player_id -> {is_gk, position}
    
    try:
        meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_match.json"
        response = requests.get(meta_data_github_url)
        raw_match_data = response.json()
        
        # Extract players
        raw_match_df = pd.json_normalize(raw_match_data, max_level=2)
        players_df = pd.json_normalize(
            raw_match_df.to_dict("records"),
            record_path="players"
        )
        
        # Create lookup with is_gk AND position - RELIABLE!
        players_df["is_gk"] = players_df["player_role.acronym"] == "GK"
        
        for _, row in players_df.iterrows():
            skillcorner_player_data[str(row["id"])] = {
                'is_gk': bool(row["is_gk"]),
                'position': row["player_role.name"] if pd.notna(row.get("player_role.name")) else None
            }
        
    except Exception as e:
        print(f"⚠ Warning: Could not load SkillCorner JSON: {e}")
        print("  Falling back to Kloppy metadata (may be unreliable)")
    # ===========================================================================
    
    # Build player metadata lookup
    player_meta = {}
    for team in dataset.metadata.teams:
        for player in team.players:
            player_id_str = str(player.player_id)
            
            # Use SkillCorner JSON if available, fallback to Kloppy
            if player_id_str in skillcorner_player_data:
                # Use SkillCorner data - RELIABLE!
                is_gk = skillcorner_player_data[player_id_str]['is_gk']
                position = skillcorner_player_data[player_id_str]['position']
            else:
                # Fallback to Kloppy metadata (unreliable)
                is_gk = player.starting_position and 'Goalkeeper' in str(player.starting_position)
                position = str(player.starting_position) if player.starting_position else None
            
            player_meta[player_id_str] = {
                'player_id': player.player_id,
                'short_name': player.name,
                'number': player.jersey_no,
                'team_id': team.team_id,
                'team_name': team.name,
                'is_gk': is_gk,        # ← NOW RELIABLE!
                'position': position,   # ← NOW RELIABLE!
            }
    
    # Convert to long format
    long_rows = []
    
    for idx, row in wide_df.iterrows():
        # Scale ball coordinates if normalized
        ball_x = row['ball_x']
        ball_y = row['ball_y']
        
        if coordinates_are_normalized and pd.notna(ball_x) and pd.notna(ball_y):
            ball_x = (ball_x * pitch_length) - (pitch_length / 2)
            ball_y = (ball_y * pitch_width) - (pitch_width / 2)
        
        frame_data = {
            'frame': row['frame_id'],
            'timestamp': row['timestamp'],
            'period': row['period_id'],
            'ball_x': ball_x,
            'ball_y': ball_y,
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
                player_x = row[cols['x']]
                player_y = row[cols['y']]
                
                # Scale player coordinates if normalized
                if coordinates_are_normalized:
                    player_x = (player_x * pitch_length) - (pitch_length / 2)
                    player_y = (player_y * pitch_width) - (pitch_width / 2)
                
                player_row = frame_data.copy()
                player_row.update({
                    'player_id': player_id,
                    'x': player_x,
                    'y': player_y,
                    'is_detected': True,
                })
                
                # Add player metadata (now with reliable is_gk and position!)
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