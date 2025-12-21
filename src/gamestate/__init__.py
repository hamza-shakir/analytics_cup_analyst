"""
Phase-Based Tactical Analytics Toolkit
SkillCorner × PySport Analytics Cup Submission

Context-aware segmentation for football tracking data analysis.
"""

__version__ = "1.0.0"

# ============================================================================
# DATA LOADING
# ============================================================================
from .load_data import (
    load_tracking_dataset,
    load_event_data,
    load_phases_data,
    get_metadata,
    list_players,
    print_match_info,
)

# ============================================================================
# MATCH CONTEXT
# ============================================================================
from .context import (
    summary,
    score_progression,
)

# ============================================================================
# SEGMENTATION
# ============================================================================
from .segments import (
    get_full_match,
    segment_by_game_state,
    get_all_game_states,
    get_game_state_summary,
    segment_by_time_window,
    segment_by_time_windows,
    segment_around_goal,
    all_goals_context,
)

# ============================================================================
# SEGMENTATION
# ============================================================================
from .filters import (
    filter_by_ip_phase,
    filter_by_oop_phase,
    filter_by_pitch_third,
    filter_by_game_state,
)

# ============================================================================
# HELPERS (for SEGMENTATION AND FILTERING)
# ============================================================================
from .helpers import (
    has_frames,
    get_frame_count,
    check_segment,
    filter_all_segments,
    analyze_all_segments,
    diagnose_filters,
)

# ============================================================================
# METRICS
# ============================================================================
from .metrics import (
    average_positions,
    team_compactness,
    defensive_line_height,
    channel_progression,
    compare_metrics,
    metric_summary,
)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
from .plots import (
    plot_average_positions,
    plot_phase_comparison,
    plot_defensive_blocks,
    plot_game_state_evolution,
    plot_team_compactness,
    plot_defensive_line,
    plot_channel_progression,
)

# Submodules (for modular imports)
from . import context
from . import segments
from . import metrics
from . import plots

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Data loading (grouped namespace)
    "load_tracking_dataset",
    "load_event_data",
    "load_phases_data",
    "get_metadata",
    "list_players",
    "to_wide_dataframe",
    "to_long_dataframe",
    "print_match_info",
    
    # Context (flat)
    "summary",
    "score_progression",
    
    # Segmentation (flat)
    "get_full_match",
    "segment_by_game_state",
    "get_all_game_states",
    "get_game_state_summary",
    "segment_by_time_window",
    "segment_by_time_windows",
    "segment_around_goal",
    "all_goals_context",

    # Filters (flat)
    "filter_by_ip_phase",
    "filter_by_oop_phase",
    "filter_by_pitch_third",
    "filter_by_game_state",

    # Helpers (flat)
    "has_frames",
    "get_frame_count",
    "check_segment",
    "filter_all_segments",
    "analyze_all_segments",
    "diagnose_filters",

    # Metrics (flat)
    "average_positions",
    "team_compactness",
    "defensive_line_height",
    "channel_progression",
    "compare_metrics",
    "metric_summary",
    
    # Plots (flat)
    "plot_average_positions",
    "plot_phase_comparison",
    "plot_defensive_blocks",
    "plot_game_state_evolution",
    "plot_team_compactness",
    "plot_defensive_line",
    "plot_channel_progression",
    
    # Submodules
    "context",
    "segments",
    "metrics",
    "plots",
]