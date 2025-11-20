"""
metrics.py v2 - Tactical Metrics Calculation

Computes tactical metrics on match segments:
1. Average player positions
2. Team width & depth (compactness)
3. Defensive line height
4. Channel progression patterns

Accepts kloppy datasets OR DataFrames (hybrid approach).
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Tuple
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
    segment: Union['TrackingDataset', pd.DataFrame],
    team: str = 'home',
    normalize_orientation: bool = True,
    match_id: Union[int, str] = None
) -> pd.DataFrame:
    """
    Calculate average positions for all players in a segment.
    
    Args:
        segment: Kloppy TrackingDataset or DataFrame (long format)
        team: 'home' or 'away'
        normalize_orientation: If True, transform so team attacks left→right
        match_id: Required if segment is DataFrame
    
    Returns:
        DataFrame with columns:
            - player_id: Player identifier
            - name: Player name
            - number: Jersey number
            - position: Playing position
            - avg_x: Average x-coordinate
            - avg_y: Average y-coordinate
            - std_x: Standard deviation of x
            - std_y: Standard deviation of y
            - frames_visible: Number of frames player appeared in
    
    Example:
        >>> drawing_segment = segments.by_game_state(1886347, state='drawing', team='home')
        >>> positions = average_positions(drawing_segment, team='home')
        >>> print(positions[['name', 'number', 'avg_x', 'avg_y']])
    """
    # Handle both dataset and DataFrame inputs
    if hasattr(segment, 'records'):  # It's a kloppy dataset
        dataset = segment
        
        # Normalize orientation if requested
        if normalize_orientation:
            dataset = dataset.transform(to_orientation=Orientation.HOME_AWAY)
        
        # Convert to DataFrame for easier calculation
        df = to_long_dataframe(dataset, match_id or 'temp')
        
    else:  # It's already a DataFrame
        df = segment.copy()
        
        # For DataFrames, we need to manually normalize if requested
        if normalize_orientation:
            # This is a simplified version - assumes direction columns exist
            if 'period' in df.columns:
                df['x'] = np.where(
                    df['period'] == 2,  # Second half
                    -df['x'],  # Flip x
                    df['x']
                )
                df['y'] = np.where(
                    df['period'] == 2,
                    -df['y'],
                    df['y']
                )
    
    # Filter to specified team
    if 'team_name' in df.columns:
        # Get team name from first row
        if team == 'home':
            team_name = df['home_team.name'].iloc[0] if 'home_team.name' in df.columns else None
        else:
            team_name = df['away_team.name'].iloc[0] if 'away_team.name' in df.columns else None
        
        if team_name:
            df = df[df['team_name'] == team_name]
    
    # Group by player and calculate statistics
    player_stats = df.groupby('player_id').agg({
        'x': ['mean', 'std'],
        'y': ['mean', 'std'],
        'frame': 'count'
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['player_id', 'avg_x', 'std_x', 'avg_y', 'std_y', 'frames_visible']
    
    # Add player metadata
    player_meta = df.groupby('player_id').agg({
        'short_name': 'first',
        'number': 'first',
        'position': 'first',
        'is_gk': 'first'
    }).reset_index()
    
    # Merge
    result = player_stats.merge(player_meta, on='player_id', how='left')
    
    # Rename and reorder columns
    result = result.rename(columns={'short_name': 'name'})
    result = result[[
        'player_id', 'name', 'number', 'position', 'is_gk',
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
    segment: Union['TrackingDataset', pd.DataFrame],
    team: str = 'home',
    normalize_orientation: bool = True,
    match_id: Union[int, str] = None
) -> Dict[str, float]:
    """
    Calculate team compactness metrics.
    
    Measures how spread out or compact the team is in the segment.
    
    Args:
        segment: Kloppy TrackingDataset or DataFrame
        team: 'home' or 'away'
        normalize_orientation: If True, normalize coordinates
        match_id: Required if segment is DataFrame
    
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
        >>> drawing_comp = team_compactness(drawing_segment, team='home')
        >>> winning_comp = team_compactness(winning_segment, team='home')
        >>> print(f"Width change: {winning_comp['width'] - drawing_comp['width']:.1f}m")
    """
    # Get average positions
    positions = average_positions(segment, team, normalize_orientation, match_id)
    
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
    segment: Union['TrackingDataset', pd.DataFrame],
    team: str = 'home',
    normalize_orientation: bool = True,
    match_id: Union[int, str] = None
) -> Dict[str, float]:
    """
    Calculate defensive line height metrics.
    
    Measures how high up the pitch the defensive line is positioned.
    
    Args:
        segment: Kloppy TrackingDataset or DataFrame
        team: 'home' or 'away'
        normalize_orientation: If True, normalize coordinates
        match_id: Required if segment is DataFrame
    
    Returns:
        Dictionary with:
            - deepest_defender_x: Minimum x-coordinate of defenders
            - avg_defensive_line_x: Average x of back line defenders
            - defensive_line_spread: Spread of defensive line (std dev)
            - num_defenders: Number of defenders analyzed
    
    Example:
        >>> drawing_line = defensive_line_height(drawing_segment, team='home')
        >>> winning_line = defensive_line_height(winning_segment, team='home')
        >>> push_up = winning_line['avg_defensive_line_x'] - drawing_line['avg_defensive_line_x']
        >>> print(f"Defensive line pushed {push_up:.1f}m higher when winning")
    """
    # Get average positions
    positions = average_positions(segment, team, normalize_orientation, match_id)
    
    # Filter to defenders (exclude GK)
    defenders = positions[
        (~positions['is_gk']) & 
        (positions['position'].notna()) &
        (positions['position'].str.contains('Back|Defender', case=False, na=False))
    ]
    
    if len(defenders) == 0:
        return {
            'deepest_defender_x': 0,
            'avg_defensive_line_x': 0,
            'defensive_line_spread': 0,
            'num_defenders': 0
        }
    
    defender_x = defenders['avg_x'].values
    
    # Deepest defender (minimum x)
    deepest_x = float(defender_x.min())
    
    # Average defensive line
    avg_line_x = float(defender_x.mean())
    
    # Spread (how compressed the line is)
    spread = float(defender_x.std()) if len(defender_x) > 1 else 0.0
    
    return {
        'deepest_defender_x': round(deepest_x, 2),
        'avg_defensive_line_x': round(avg_line_x, 2),
        'defensive_line_spread': round(spread, 2),
        'num_defenders': len(defenders)
    }


# ----------------------------------------------------------
# 4. Channel Progression Patterns
# ----------------------------------------------------------

def channel_progression(
    segment: Union['TrackingDataset', pd.DataFrame],
    team: str = 'home',
    normalize_orientation: bool = True,
    match_id: Union[int, str] = None,
    pitch_width: float = 68.0
) -> Dict[str, Union[float, int]]:
    """
    Analyze ball progression through vertical pitch channels.
    
    Divides pitch into left/center/right channels and tracks forward ball movement.
    
    Args:
        segment: Kloppy TrackingDataset or DataFrame
        team: 'home' or 'away'
        normalize_orientation: If True, normalize coordinates
        match_id: Required if segment is DataFrame
        pitch_width: Pitch width in meters (default: 68)
    
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
        >>> channels = channel_progression(losing_segment, team='away')
        >>> print(f"When losing, attacked {channels['left_pct']:.0f}% through left flank")
    """
    # Convert to DataFrame if needed
    if hasattr(segment, 'records'):  # It's a kloppy dataset
        dataset = segment
        
        if normalize_orientation:
            dataset = dataset.transform(to_orientation=Orientation.HOME_AWAY)
        
        df = to_long_dataframe(dataset, match_id or 'temp')
    else:
        df = segment.copy()
    
    # Get frames where team has possession
    if team == 'home':
        team_name = df['home_team.name'].iloc[0] if 'home_team.name' in df.columns else None
    else:
        team_name = df['away_team.name'].iloc[0] if 'away_team.name' in df.columns else None
    
    # Filter to when this team has possession
    if 'possession_group' in df.columns:
        possession_filter = (
            (df['possession_group'] == 'home team') if team == 'home'
            else (df['possession_group'] == 'away team')
        )
        df = df[possession_filter]
    
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
    segments_dict: Dict[str, Union['TrackingDataset', pd.DataFrame]],
    metric_func: callable,
    **kwargs
) -> pd.DataFrame:
    """
    Compare a metric across multiple segments.
    
    Args:
        segments_dict: Dict mapping labels to segments
        metric_func: Metric function to apply (e.g., team_compactness)
        **kwargs: Additional arguments to pass to metric_func
    
    Returns:
        DataFrame with one row per segment
    
    Example:
        >>> segments = segments.get_all_game_states(1886347, team='home')
        >>> comparison = compare_metrics(segments, team_compactness, team='home')
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
    segments_dict: Dict[str, Union['TrackingDataset', pd.DataFrame]],
    team: str = 'home',
    match_id: Union[int, str] = None
) -> pd.DataFrame:
    """
    Generate comprehensive metric summary for all segments.
    
    Calculates all metrics for each segment and returns combined table.
    
    Args:
        segments_dict: Dict mapping labels to segments
        team: 'home' or 'away'
        match_id: Match identifier
    
    Returns:
        DataFrame with all metrics for each segment
    
    Example:
        >>> segments = segments.get_all_game_states(1886347, team='home')
        >>> summary = metric_summary(segments, team='home', match_id=1886347)
        >>> print(summary)
    """
    results = []
    
    for label, segment in segments_dict.items():
        # Calculate all metrics
        comp = team_compactness(segment, team, match_id=match_id)
        def_line = defensive_line_height(segment, team, match_id=match_id)
        channels = channel_progression(segment, team, match_id=match_id)
        
        # Combine
        row = {
            'segment': label,
            **comp,
            **def_line,
            **channels
        }
        
        results.append(row)
    
    return pd.DataFrame(results)