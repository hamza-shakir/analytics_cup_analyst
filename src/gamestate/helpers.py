"""
helpers.py - Utility functions for segment validation, dict operations, and diagnostics

This module provides helper functions to work with segments and filters:
- Segment validation (has_frames, check_segment, get_frame_count)
- Dict operations (filter_all_segments, analyze_all_segments)
- Diagnostics (diagnose_filters)
"""

from typing import Dict, Callable, Any, Optional

try:
    from kloppy.domain import TrackingDataset
except ImportError:
    # For testing without kloppy
    TrackingDataset = Any


# ============================================================================
# SEGMENT VALIDATION
# ============================================================================

def has_frames(segment: TrackingDataset) -> bool:
    """
    Check if segment has any frames.
    
    Args:
        segment: TrackingDataset to check
    
    Returns:
        bool: True if segment has at least one frame, False otherwise
    
    Example:
        >>> segment = segment_by_time_window(1886347, 0, 15)
        >>> if has_frames(segment):
        ...     positions = average_positions(segment, 'home', 1886347)
    """
    return len(segment.records) > 0


def get_frame_count(segment: TrackingDataset) -> int:
    """
    Get number of frames in segment.
    
    Args:
        segment: TrackingDataset to count
    
    Returns:
        int: Number of frames in the segment
    
    Example:
        >>> segment = segment_by_time_window(1886347, 0, 45)
        >>> print(f"First half has {get_frame_count(segment)} frames")
    """
    return len(segment.records)


def check_segment(segment: TrackingDataset, min_frames: int = 100, verbose: bool = True) -> bool:
    """
    Check if segment has sufficient frames for analysis.
    
    Args:
        segment: TrackingDataset to check
        min_frames: Minimum recommended frames (default: 100)
        verbose: Print diagnostic messages (default: True)
    
    Returns:
        bool: True if segment has frames (even if below min_frames), False if empty
    
    Example:
        >>> segment = filter_by_phase_type(seg, 'build_up', 'home', 1886347)
        >>> if check_segment(segment, min_frames=200):
        ...     # Proceed with analysis
        ...     compactness = team_compactness(segment, 'home', 1886347)
        
    Notes:
        - Returns False for empty segments (0 frames)
        - Returns True with warning for small segments (< min_frames)
        - Returns True with success message for adequate segments (>= min_frames)
    """
    frame_count = len(segment.records)
    
    if frame_count == 0:
        if verbose:
            print(f"❌ Segment is empty (0 frames)")
        return False
    elif frame_count < min_frames:
        if verbose:
            print(f"⚠️  Segment has only {frame_count} frames (recommended: {min_frames}+)")
            print(f"   Results may not be statistically representative")
        return True
    else:
        if verbose:
            print(f"✅ Segment has {frame_count} frames")
        return True


# ============================================================================
# DICT OPERATIONS
# ============================================================================

def filter_all_segments(
    segments_dict: Dict[str, TrackingDataset],
    filter_func: Callable,
    *args,
    **kwargs
) -> Dict[str, TrackingDataset]:
    """
    Apply a filter function to all segments in a dictionary.
    
    Useful when working with functions that return Dict[str, TrackingDataset],
    such as segment_by_time_windows() or segment_around_goal().
    
    Args:
        segments_dict: Dictionary of label -> TrackingDataset
        filter_func: Filter function to apply (e.g., filter_by_phase_type)
        *args: Positional arguments for filter_func
        **kwargs: Keyword arguments for filter_func
    
    Returns:
        Dict[str, TrackingDataset]: Dictionary with same keys, filtered segments
    
    Example:
        >>> # Get 15-minute windows
        >>> windows = segment_by_time_windows(1886347, 15)
        >>> 
        >>> # Filter all windows to build-up phases
        >>> filtered = filter_all_segments(
        ...     windows,
        ...     filter_by_phase_type,
        ...     'build_up',
        ...     team='home',
        ...     match_id=1886347
        ... )
        >>> 
        >>> # Now analyze each window
        >>> for window_label, segment in filtered.items():
        ...     if has_frames(segment):
        ...         pos = average_positions(segment, 'home', 1886347)
        ...         print(f"{window_label}: {len(pos)} players")
    
    Example 2 - Multiple filters:
        >>> windows = segment_by_time_windows(1886347, 15)
        >>> 
        >>> # Apply multiple filters
        >>> windows = filter_all_segments(
        ...     windows, filter_by_game_state, 'winning', 'home', 1886347
        ... )
        >>> windows = filter_all_segments(
        ...     windows, filter_by_phase_type, 'create', 'home', 1886347
        ... )
    """
    filtered_dict = {}
    
    for label, segment in segments_dict.items():
        filtered_segment = filter_func(segment, *args, **kwargs)
        filtered_dict[label] = filtered_segment
    
    return filtered_dict


def analyze_all_segments(
    segments_dict: Dict[str, TrackingDataset],
    metric_func: Callable,
    team: str,
    match_id: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply a metric function to all segments in a dictionary.
    
    Useful for analyzing all time windows or goal contexts at once.
    
    Args:
        segments_dict: Dictionary of label -> TrackingDataset
        metric_func: Metric function to apply (e.g., team_compactness)
        team: 'home' or 'away'
        match_id: Match identifier
        **kwargs: Additional keyword arguments for metric_func
    
    Returns:
        Dict[str, Any]: Dictionary mapping segment labels to metric results
    
    Example:
        >>> # Get 15-minute windows
        >>> windows = segment_by_time_windows(1886347, 15)
        >>> 
        >>> # Analyze compactness in all windows
        >>> results = analyze_all_segments(
        ...     windows,
        ...     team_compactness,
        ...     team='home',
        ...     match_id=1886347
        ... )
        >>> 
        >>> # Print results
        >>> for window, compactness in results.items():
        ...     if compactness is not None:
        ...         print(f"{window}: width={compactness['width']:.1f}m")
    
    Example 2 - With filtered segments:
        >>> windows = segment_by_time_windows(1886347, 15)
        >>> windows = filter_all_segments(
        ...     windows, filter_by_phase_type, 'build_up', 'home', 1886347
        ... )
        >>> 
        >>> # Analyze defensive line in build-up for each window
        >>> results = analyze_all_segments(
        ...     windows,
        ...     defensive_line_height,
        ...     team='home',
        ...     match_id=1886347
        ... )
    """
    results = {}
    
    for label, segment in segments_dict.items():
        if not has_frames(segment):
            results[label] = None
        else:
            result = metric_func(segment, team=team, match_id=match_id, **kwargs)
            results[label] = result
    
    return results


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def diagnose_filters(
    match_id: int,
    team: str,
    filters: list,
    start_minute: Optional[float] = None,
    end_minute: Optional[float] = None
) -> None:
    """
    Diagnose filter pipeline to show where frames are lost.
    
    Helps identify which filters are most restrictive and whether
    the combination of filters results in useful data.
    
    Args:
        match_id: Match identifier
        team: 'home' or 'away'
        filters: List of (filter_func, args, kwargs) tuples
        start_minute: Optional start time for base segment
        end_minute: Optional end time for base segment
    
    Example:
        >>> diagnose_filters(
        ...     match_id=1886347,
        ...     team='home',
        ...     filters=[
        ...         (filter_by_game_state, ('winning',), {}),
        ...         (filter_by_phase_type, ('build_up',), {}),
        ...         (filter_by_third, ('defensive',), {})
        ...     ],
        ...     start_minute=0,
        ...     end_minute=45
        ... )
        
        Output:
        ╔═══════════════════════════════════════════════════════════╗
        ║                   FILTER DIAGNOSTICS                       ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Match: 1886347 | Team: home | Period: 0-45 mins          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Base segment:                          54,000 frames      ║
        ║ After filter_by_game_state('winning'): 23,400 frames (-57%)║
        ║ After filter_by_phase_type('build_up'): 8,200 frames (-65%)║
        ║ After filter_by_third('defensive'):     3,100 frames (-62%)║
        ╠═══════════════════════════════════════════════════════════╣
        ║ FINAL RESULT: 3,100 frames (6% of original)              ║
        ╚═══════════════════════════════════════════════════════════╝
    """
    from load_data import load_tracking_data
    from segments import segment_by_time_window, get_full_match
    
    # Load base segment
    if start_minute is not None and end_minute is not None:
        segment = segment_by_time_window(match_id, start_minute, end_minute)
        period_desc = f"{start_minute}-{end_minute} mins"
    else:
        segment = get_full_match(match_id)
        period_desc = "full match"
    
    initial_frames = get_frame_count(segment)
    
    # Print header
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║                   FILTER DIAGNOSTICS                       ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║ Match: {match_id} | Team: {team} | Period: {period_desc:<20}║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║ Base segment: {initial_frames:>35,} frames      ║")
    
    # Apply filters sequentially
    current_frames = initial_frames
    
    for filter_func, args, kwargs in filters:
        # Apply filter
        segment = filter_func(segment, *args, team=team, match_id=match_id, **kwargs)
        new_frames = get_frame_count(segment)
        
        # Calculate loss
        frames_lost = current_frames - new_frames
        percent_lost = (frames_lost / current_frames * 100) if current_frames > 0 else 0
        
        # Format filter name and args
        filter_name = filter_func.__name__
        args_str = ', '.join([f"'{a}'" if isinstance(a, str) else str(a) for a in args])
        
        # Print result
        print(f"║ After {filter_name}({args_str}): {new_frames:>15,} frames (-{percent_lost:.0f}%)║")
        
        current_frames = new_frames
    
    # Print footer
    final_percent = (current_frames / initial_frames * 100) if initial_frames > 0 else 0
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║ FINAL RESULT: {current_frames:>30,} frames ({final_percent:.1f}% of original)              ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    # Recommendations
    if current_frames == 0:
        print("❌ WARNING: No frames remaining after filtering!")
        print("   → Try relaxing some filter criteria")
        print("   → Check if data exists for this combination")
    elif current_frames < 100:
        print("⚠️  WARNING: Very few frames remaining!")
        print(f"   → Only {current_frames} frames may not be statistically meaningful")
        print("   → Consider using broader time windows or fewer filters")
    elif current_frames < 500:
        print("⚠️  Caution: Limited data remaining")
        print(f"   → {current_frames} frames may be sufficient but results could be noisy")
    else:
        print(f"✅ Good: {current_frames:,} frames should provide reliable results")