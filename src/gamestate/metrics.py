"""
metrics.py - Tactical Metrics Calculation

Computes tactical metrics on match segments:
1. Average player positions
2. Team width & depth (compactness)
3. Defensive line height
4. Channel progression patterns

Accepts kloppy TrackingDataset objects only.
Use to_long_dataframe() if you need DataFrame format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy.spatial import ConvexHull
from kloppy.domain import Orientation

from .load_data import (
    load_tracking_dataset,
    to_long_dataframe,
    to_wide_dataframe
)


# ----------------------------------------------------------
# 1. Average Player Positions
# ----------------------------------------------------------

def average_positions(
    segment: 'TrackingDataset',
    team: str = 'home',
    normalize_orientation: bool = True,
    match_id: int = None,
    possession: str = 'all'
) -> pd.DataFrame:
    """
    Calculate average positions for all players in a segment.
    
    Args:
        segment: Kloppy TrackingDataset
        team: 'home' or 'away'
        normalize_orientation: If True, transform so team attacks left→right
        match_id: Match identifier (required for to_long_dataframe)
        possession: 'all', 'ip', or 'oop'
            - 'all': All frames (default)
            - 'ip': In Possession - when team has the ball (attacking shape)
            - 'oop': Out Of Possession - when opponent has ball (defending shape)
    
    Returns:
        DataFrame with columns:
            - player_id: Player identifier
            - name: Player name
            - number: Jersey number
            - position: Playing position
            - is_gk: Boolean, True if goalkeeper
            - sub_status: 'full90', 'subbed_out', 'subbed_in', or 'unused_sub'
            - avg_x: Average x-coordinate (in meters)
            - avg_y: Average y-coordinate (in meters)
            - std_x: Standard deviation of x
            - std_y: Standard deviation of y
            - frames_visible: Number of frames player appeared in
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> drawing = gs.segment_by_game_state(1886347, state='drawing', team='home')
        >>> 
        >>> # All frames
        >>> all_pos = gs.average_positions(drawing, team='home', match_id=1886347)
        >>> 
        >>> # Check substitution status
        >>> print(all_pos[['name', 'number', 'sub_status', 'avg_x', 'avg_y']])
        >>> 
        >>> # Only when attacking (In Possession)
        >>> ip_pos = gs.average_positions(drawing, team='home', match_id=1886347, possession='ip')
        >>> 
        >>> # Only when defending (Out Of Possession)
        >>> oop_pos = gs.average_positions(drawing, team='home', match_id=1886347, possession='oop')
        >>> 
        >>> # Compare
        >>> gs.compare_positions([ip_pos, oop_pos], 
        ...                      titles=["In Possession", "Out Of Possession"])
    """
    if match_id is None:
        raise ValueError("match_id is required")
    
    # Normalize orientation
    dataset = _normalize_for_team(segment, team, normalize_orientation)
    
    # Convert to DataFrame
    df = to_long_dataframe(dataset, match_id)
    
    # Filter to specified team
    if 'team_name' in df.columns:
        # Get team name from first row
        if team == 'home':
            team_name = df['home_team.name'].iloc[0] if 'home_team.name' in df.columns else None
        else:
            team_name = df['away_team.name'].iloc[0] if 'away_team.name' in df.columns else None
        
        if team_name:
            df = df[df['team_name'] == team_name]
    
    # Filter by possession
    df = _filter_by_possession(df, team, possession)
    
    if df.empty:
        print(f"Warning: No data after filtering (team={team}, possession={possession})")
        return pd.DataFrame(columns=[
            'player_id', 'name', 'number', 'position', 'is_gk', 'sub_status',
            'avg_x', 'avg_y', 'std_x', 'std_y', 'frames_visible'
        ])
    
    # Group by player and calculate statistics
    player_stats = df.groupby('player_id').agg({
        'x': ['mean', 'std'],
        'y': ['mean', 'std'],
        'frame': 'count'
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['player_id', 'avg_x', 'std_x', 'avg_y', 'std_y', 'frames_visible']
    
    # Add player metadata (including sub_status)
    player_meta = df.groupby('player_id').agg({
        'short_name': 'first',
        'number': 'first',
        'position': 'first',
        'is_gk': 'first',
        'sub_status': 'first'
    }).reset_index()
    
    # Merge
    result = player_stats.merge(player_meta, on='player_id', how='left')
    
    # Rename and reorder columns
    result = result.rename(columns={'short_name': 'name'})
    result = result[[
        'player_id', 'name', 'number', 'position', 'is_gk', 'sub_status',
        'avg_x', 'avg_y', 'std_x', 'std_y', 'frames_visible'
    ]]
    
    # Round for readability
    result['avg_x'] = result['avg_x'].round(2)
    result['avg_y'] = result['avg_y'].round(2)
    result['std_x'] = result['std_x'].round(2)
    result['std_y'] = result['std_y'].round(2)
    
    # Sort by average x (defensive to attacking)
    result = result.sort_values('avg_x').reset_index(drop=True)
    
    return result


# ----------------------------------------------------------
# 2. Team Width & Depth (Compactness)
# ----------------------------------------------------------

def team_compactness(
    segment: 'TrackingDataset',
    team: str = 'home',
    normalize_orientation: bool = True,
    match_id: int = None,
    possession: str = 'all'
) -> Dict[str, float]:
    """
    Calculate team compactness metrics.
    
    Measures how spread out or compact the team is in the segment.
    
    Args:
        segment: Kloppy TrackingDataset
        team: 'home' or 'away'
        normalize_orientation: If True, transform so team attacks left→right
        match_id: Match identifier (required)
        possession: 'all', 'ip', or 'oop'
    
    Returns:
        Dictionary with:
            - width: Team width (max_y - min_y)
            - depth: Team depth (max_x - min_x)
            - area: Convex hull area of team shape
            - centroid_x: Team center x-coordinate
            - centroid_y: Team center y-coordinate
            - compactness: Average distance from centroid
            - width_to_depth_ratio: Width/depth ratio
            - players_analyzed: Number of players included
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> drawing = gs.segment_by_game_state(1886347, state='drawing', team='home')
        >>> 
        >>> # Compactness when attacking (IP)
        >>> ip_comp = gs.team_compactness(drawing, team='home', match_id=1886347, possession='ip')
        >>> print(f"Width when attacking: {ip_comp['width']}m")
        >>> 
        >>> # Compactness when defending (OOP)
        >>> oop_comp = gs.team_compactness(drawing, team='home', match_id=1886347, possession='oop')
        >>> print(f"Width when defending: {oop_comp['width']}m")
    """
    if match_id is None:
        raise ValueError("match_id is required")
    
    # Get average positions with possession filter
    positions = average_positions(segment, team, normalize_orientation, match_id, possession)
    
    # Remove goalkeepers for field player analysis
    field_players = positions[~positions['is_gk']]
    
    if len(field_players) < 3:
        return {
            'width': 0, 'depth': 0, 'area': 0,
            'centroid_x': 0, 'centroid_y': 0,
            'compactness': 0, 'width_to_depth_ratio': 0,
            'players_analyzed': len(field_players)
        }
    
    # Extract coordinates
    x_coords = field_players['avg_x'].values
    y_coords = field_players['avg_y'].values
    
    # Width and depth
    width = float(y_coords.max() - y_coords.min())
    depth = float(x_coords.max() - x_coords.min())
    
    # Centroid
    centroid_x = float(x_coords.mean())
    centroid_y = float(y_coords.mean())
    
    # Compactness (average distance from centroid)
    distances = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)
    compactness = float(distances.mean())
    
    # Convex hull area
    try:
        points = np.column_stack([x_coords, y_coords])
        hull = ConvexHull(points)
        area = float(hull.volume)  # In 2D, volume is area
    except:
        area = 0.0
    
    # Width to depth ratio
    width_to_depth_ratio = float(width / depth) if depth > 0 else 0.0
    
    return {
        'width': round(width, 2),
        'depth': round(depth, 2),
        'area': round(area, 2),
        'centroid_x': round(centroid_x, 2),
        'centroid_y': round(centroid_y, 2),
        'compactness': round(compactness, 2),
        'width_to_depth_ratio': round(width_to_depth_ratio, 2),
        'players_analyzed': len(field_players)
    }


# ----------------------------------------------------------
# 3. Defensive Line Height
# ----------------------------------------------------------

def defensive_line_height(
    segment: 'TrackingDataset',
    team: str = 'home',
    normalize_orientation: bool = True,
    match_id: int = None,
    possession: str = 'all'
) -> Dict[str, float]:
    """
    Calculate defensive line height by tracking the last man's distance from own goal.
    
    The defensive line is defined by the deepest defender (last man) at each moment.
    We calculate the DISTANCE from own goal to understand how high/low the line is.
    
    Uses conditional orientation transformation so the analyzed team always attacks
    left→right, making calculations simpler and visualizations more intuitive.
    
    Args:
        segment: Kloppy TrackingDataset
        team: 'home' or 'away'
        normalize_orientation: If True, transform so team attacks left→right
        match_id: Match identifier (required)
        possession: 'all', 'ip', or 'oop'
            - 'all': All frames (default)
            - 'ip': In Possession - when team has the ball
            - 'oop': Out Of Possession - when opponent has ball
    
    Returns:
        Dictionary with:
            - deepest_defender_x: Minimum distance (dropped deepest, closest to goal)
            - median_defensive_line_x: Median distance (typical position, most representative)
            - avg_defensive_line_x: Mean distance (overall average, affected by extremes)
            - defensive_line_spread: Std dev (variability of line movement)
    
    Examples:
        >>> import gamestate as gs
        >>> 
        >>> # Full match - home team
        >>> segment = gs.segment_by_time(1886347, 0, 90)
        >>> home_line = gs.defensive_line_height(segment, team='home', match_id=1886347)
        >>> print(home_line)
        {
            'deepest_defender_x': 12.5,      # Dropped to 12.5m from goal (deepest)
            'median_defensive_line_x': 26.0, # Typically at 26m (most representative)
            'avg_defensive_line_x': 28.5,    # Average 28.5m (pulled by high pressing)
            'defensive_line_spread': 8.2     # Moves ±8.2m typically
        }
        >>> 
        >>> # Full match - away team
        >>> away_line = gs.defensive_line_height(segment, team='away', match_id=1886347)
        >>> print(away_line)
        {
            'deepest_defender_x': 15.2,
            'median_defensive_line_x': 30.5,
            'avg_defensive_line_x': 31.8,
            'defensive_line_spread': 6.5
        }
        >>> # Away plays higher and more stable line than home
        >>> 
        >>> # When defending (Out Of Possession)
        >>> oop_line = gs.defensive_line_height(segment, team='home', match_id=1886347, possession='oop')
        >>> print(f"Typical line when defending: {oop_line['median_defensive_line_x']:.1f}m from goal")
        Typical line when defending: 23.5m from goal
    
    Notes:
        Values are in meters from own goal line (0-105m possible range):
        - Low/Deep line: 10-20m from own goal (conservative)
        - Medium line: 20-35m from own goal (balanced)
        - High line: 35-50m from own goal (aggressive pressing)
        
        Median vs Mean:
        - Median shows typical position (not affected by outliers)
        - Mean shows overall average (affected by extreme pressing/dropping)
        - If median < mean: Team occasionally pushes very high
        - If median > mean: Team occasionally drops very deep
        
        Spread indicates line dynamism:
        - Small (2-5m): Static, disciplined line
        - Medium (5-10m): Normal defensive movement
        - Large (10-15m): Very dynamic, situational line
    """
    if match_id is None:
        raise ValueError("match_id is required")
    
    # Normalize orientation
    dataset = _normalize_for_team(segment, team, normalize_orientation)
    
    # Convert to DataFrame
    df = to_long_dataframe(dataset, match_id)
    
    # Filter to specified team
    if 'team_name' in df.columns:
        # Get team name from first row
        if team == 'home':
            team_name = df['home_team.name'].iloc[0] if 'home_team.name' in df.columns else None
        else:
            team_name = df['away_team.name'].iloc[0] if 'away_team.name' in df.columns else None
        
        if team_name:
            df = df[df['team_name'] == team_name]
    
    # Filter by possession
    df = _filter_by_possession(df, team, possession)
    
    if df.empty:
        return {
            'deepest_defender_x': 0.0,
            'median_defensive_line_x': 0.0,
            'avg_defensive_line_x': 0.0,
            'defensive_line_spread': 0.0
        }
    
    # Track last man DISTANCE from own goal for each frame
    last_man_distances = []
    
    # After conditional orientation transformation:
    # - Team ALWAYS attacks left→right
    # - Team ALWAYS defends at X = -52.5 (left goal)
    # - Deepest defender = min(X) = closest to -52.5
    
    own_goal_x = -52.5  # Always defends left goal after transformation
    
    for frame_id in df['frame'].unique():
        frame_data = df[df['frame'] == frame_id]
        
        # Get outfield players (exclude GK)
        outfield = frame_data[frame_data['is_gk'] == False]
        
        if len(outfield) > 0:
            # Deepest defender = minimum X (closest to left goal at -52.5)
            deepest_x = outfield['x'].min()
            
            # Calculate distance from own goal
            # distance = deepest_x - (-52.5) = deepest_x + 52.5
            distance_from_goal = deepest_x - own_goal_x
            
            last_man_distances.append(distance_from_goal)
    
    if not last_man_distances:
        return {
            'deepest_defender_x': 0.0,
            'median_defensive_line_x': 0.0,
            'avg_defensive_line_x': 0.0,
            'defensive_line_spread': 0.0
        }
    
    # Calculate statistics on distances from own goal
    distance_array = np.array(last_man_distances)
    
    return {
        'deepest_defender_x': round(float(distance_array.min()), 2),        # Minimum (dropped deepest)
        'median_defensive_line_x': round(float(np.median(distance_array)), 2), # Median (typical position)
        'avg_defensive_line_x': round(float(distance_array.mean()), 2),     # Mean (overall average)
        'defensive_line_spread': round(float(distance_array.std()), 2) if len(distance_array) > 1 else 0.0
    }


# ----------------------------------------------------------
# 4. Channel Progression Patterns
# ----------------------------------------------------------

def channel_progression(
    segment: 'TrackingDataset',
    team: str = 'home',
    normalize_orientation: bool = True,
    match_id: int = None,
    pitch_width: float = 68.0,
    possession: str = 'ip'
) -> Dict[str, float]:
    """
    Analyze ball progression through vertical pitch channels.
    
    Divides pitch into left/center/right channels and tracks forward ball movement.
    
    Args:
        segment: Kloppy TrackingDataset
        team: 'home' or 'away'
        normalize_orientation: If True, transform so team attacks left→right
        match_id: Match identifier (required)
        pitch_width: Pitch width in meters (default: 68)
        possession: 'all', 'ip', or 'oop' (default: 'ip')
            Note: This metric typically only makes sense for 'ip' (when attacking)
    
    Returns:
        Dictionary with:
            - left_count: Forward progressions in left channel
            - left_pct: Percentage in left channel
            - center_count: Forward progressions in center channel
            - center_pct: Percentage in center channel
            - right_count: Forward progressions in right channel
            - right_pct: Percentage in right channel
            - total_progressions: Total forward movements detected
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> drawing = gs.segment_by_game_state(1886347, state='drawing', team='home')
        >>> 
        >>> # Attack patterns when in possession (default)
        >>> channels = gs.channel_progression(drawing, team='home', match_id=1886347)
        >>> print(f"Left: {channels['left_pct']:.0f}%")
        >>> print(f"Center: {channels['center_pct']:.0f}%")
        >>> print(f"Right: {channels['right_pct']:.0f}%")
    """
    if match_id is None:
        raise ValueError("match_id is required")
    
    # Normalize orientation
    dataset = _normalize_for_team(segment, team, normalize_orientation)
    
    # Convert to DataFrame
    df = to_long_dataframe(dataset, match_id)
    
    # Filter by possession
    df = _filter_by_possession(df, team, possession)
    
    if df.empty:
        return {
            'left_count': 0, 'left_pct': 0.0,
            'center_count': 0, 'center_pct': 0.0,
            'right_count': 0, 'right_pct': 0.0,
            'total_progressions': 0
        }
    
    # Get ball positions by frame
    ball_positions = df.groupby('frame').agg({
        'ball_x': 'first',
        'ball_y': 'first'
    }).sort_values('frame')
    
    # Define channels (thirds of pitch width)
    left_threshold = -pitch_width / 6
    right_threshold = pitch_width / 6
    
    # Track forward progressions by channel
    left_count = 0
    center_count = 0
    right_count = 0
    
    # Analyze consecutive frames
    prev_x = None
    for idx, row in ball_positions.iterrows():
        ball_x = row['ball_x']
        ball_y = row['ball_y']
        
        if prev_x is not None and pd.notna(ball_x) and pd.notna(ball_y):
            # Check if ball moved forward
            if ball_x > prev_x + 1:  # Forward progression threshold (1m)
                # Determine channel
                if ball_y < left_threshold:
                    left_count += 1
                elif ball_y > right_threshold:
                    right_count += 1
                else:
                    center_count += 1
        
        prev_x = ball_x
    
    # Calculate totals and percentages
    total = left_count + center_count + right_count
    
    if total > 0:
        left_pct = 100 * left_count / total
        center_pct = 100 * center_count / total
        right_pct = 100 * right_count / total
    else:
        left_pct = center_pct = right_pct = 0.0
    
    return {
        'left_count': left_count,
        'left_pct': round(left_pct, 1),
        'center_count': center_count,
        'center_pct': round(center_pct, 1),
        'right_count': right_count,
        'right_pct': round(right_pct, 1),
        'total_progressions': total
    }


# ----------------------------------------------------------
# Helper: Compare Metrics Across Segments
# ----------------------------------------------------------

def compare_metrics(
    segments_dict: Dict[str, 'TrackingDataset'],
    metric_func: callable,
    **kwargs
) -> pd.DataFrame:
    """
    Compare a metric across multiple segments.
    
    Args:
        segments_dict: Dict mapping labels to TrackingDataset segments
        metric_func: Metric function to apply (e.g., team_compactness)
        **kwargs: Additional arguments to pass to metric_func
    
    Returns:
        DataFrame with one row per segment
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> segments = gs.get_all_game_states(1886347, team='home')
        >>> comparison = gs.compare_metrics(segments, gs.team_compactness, 
        ...                                  team='home', match_id=1886347)
        >>> print(comparison[['segment', 'width', 'depth', 'area']])
    """
    results = []
    
    for label, segment in segments_dict.items():
        metrics = metric_func(segment, **kwargs)
        
        # Add segment label
        if isinstance(metrics, dict):
            metrics['segment'] = label
            results.append(metrics)
        elif isinstance(metrics, pd.DataFrame):
            metrics['segment'] = label
            results.append(metrics)
    
    # Combine results
    if results:
        if isinstance(results[0], dict):
            return pd.DataFrame(results)
        else:
            return pd.concat(results, ignore_index=True)
    
    return pd.DataFrame()


def metric_summary(
    segments_dict: Dict[str, 'TrackingDataset'],
    team: str = 'home',
    match_id: int = None,
    possession: str = 'all'
) -> pd.DataFrame:
    """
    Generate comprehensive metric summary for all segments.
    
    Calculates all metrics for each segment and returns combined table.
    
    Args:
        segments_dict: Dict mapping labels to TrackingDataset segments
        team: 'home' or 'away'
        match_id: Match identifier (required)
        possession: 'all', 'ip', or 'oop'
    
    Returns:
        DataFrame with all metrics for each segment
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> segments = gs.get_all_game_states(1886347, team='home')
        >>> summary = gs.metric_summary(segments, team='home', match_id=1886347, possession='ip')
        >>> print(summary)
    """
    if match_id is None:
        raise ValueError("match_id is required")
    
    results = []
    
    for label, segment in segments_dict.items():
        # Calculate all metrics
        comp = team_compactness(segment, team, match_id=match_id, possession=possession)
        def_line = defensive_line_height(segment, team, match_id=match_id, possession=possession)
        channels = channel_progression(segment, team, match_id=match_id, possession=possession)
        
        # Combine
        row = {
            'segment': label,
            **comp,
            **def_line,
            **channels
        }
        
        results.append(row)
    
    return pd.DataFrame(results)


# ----------------------------------------------------------
# Helper: Orientation Normalization
# ----------------------------------------------------------

def _normalize_for_team(dataset: 'TrackingDataset', team: str, normalize_orientation: bool) -> 'TrackingDataset':
    """
    Normalize dataset orientation so specified team attacks left→right.
    
    Args:
        dataset: Kloppy TrackingDataset
        team: 'home' or 'away'
        normalize_orientation: If True, apply transformation
    
    Returns:
        Transformed dataset (or original if normalize_orientation=False)
    
    Notes:
        After normalization:
        - Team always defends at X = -52.5 (left goal)
        - Team always attacks towards X = +52.5 (right goal)
        - This makes all plots consistent regardless of actual coin toss
    """
    if not normalize_orientation:
        return dataset
    
    if team.lower() == 'home':
        # Home attacks left→right in both periods
        return dataset.transform(to_orientation=Orientation.STATIC_HOME_AWAY)
    else:  # away
        # Away attacks left→right in both periods
        return dataset.transform(to_orientation=Orientation.STATIC_AWAY_HOME)


# ----------------------------------------------------------
# Helper: Possession Filtering
# ----------------------------------------------------------

def _filter_by_possession(df: pd.DataFrame, team: str, possession: str) -> pd.DataFrame:
    """
    Filter DataFrame by possession state.
    
    Args:
        df: DataFrame with 'possession_group' column
        team: 'home' or 'away'
        possession: 'all', 'ip', or 'oop'
            - 'all': No filtering (returns all frames)
            - 'ip': In Possession (team has the ball - attacking)
            - 'oop': Out Of Possession (opponent has ball - defending)
    
    Returns:
        Filtered DataFrame
    """
    # Normalize possession parameter
    possession = possession.lower()
    
    # Map old values to new (backward compatibility)
    if possession in ['in', 'attack', 'attacking']:
        possession = 'ip'
    elif possession in ['out', 'defend', 'defending', 'defense']:
        possession = 'oop'
    
    # No filtering for 'all'
    if possession == 'all':
        return df
    
    # Check if possession_group column exists
    if 'possession_group' not in df.columns:
        print("Warning: No possession data available, returning all frames")
        return df
    
    # Determine which possession group to keep
    if possession == 'ip':
        # In Possession (IP) - team has the ball
        target_group = 'home team' if team == 'home' else 'away team'
    elif possession == 'oop':
        # Out Of Possession (OOP) - opponent has the ball
        target_group = 'away team' if team == 'home' else 'home team'
    else:
        # Unknown value, return all
        print(f"Warning: Unknown possession value '{possession}'. Use 'all', 'ip', or 'oop'")
        return df
    
    # Filter
    filtered_df = df[df['possession_group'] == target_group].copy()
    
    # Warn if no frames found
    if len(filtered_df) == 0:
        print(f"Warning: No frames found for possession='{possession}' (team={team})")
    
    return filtered_df
