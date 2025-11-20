"""
gamestate — Game State Football Analysis Package

Provides tools for:
- Loading SkillCorner tracking and event data
- Computing team and player analytics
- Match context and segmentation
"""

# Core data loading (keep grouped)
from .load_data import (
    load_tracking_dataset,
    load_event_data,
    get_metadata,
    list_players,
    to_wide_dataframe,
    to_long_dataframe,
    print_match_info,
    load_enriched_tracking_data,  # Backward compatibility
)

# Match context (flat)
from .context import (
    summary,
    score_progression,
    substitutions,
)

# Segmentation (flat)
from .segments import (
    segment_by_game_state,
    get_all_game_states,
    get_game_state_summary,
    segment_by_time_window,
    segment_by_time_windows,
    segment_around_goal,
    all_goals_context,
)

# Metrics (flat)
from .metrics import (
    average_positions,
    team_compactness,
    defensive_line_height,
    channel_progression,
    compare_metrics,
    metric_summary,
)

# Plots (flat)
from .plots import (
    plot_average_positions,
    compare_positions,
    plot_shape_with_metrics,
    plot_defensive_line,
    plot_channel_progression,
    create_comparison_dashboard,
    save_plot,
)

# Submodules (for modular imports)
from . import context
from . import segments
from . import metrics
from . import plots

__all__ = [
    # Data loading (grouped namespace)
    "load_tracking_dataset",
    "load_event_data",
    "get_metadata",
    "list_players",
    "to_wide_dataframe",
    "to_long_dataframe",
    "print_match_info",
    "load_enriched_tracking_data",
    
    # Context (flat)
    "summary",
    "score_progression",
    "substitutions",
    
    # Segmentation (flat)
    "segment_by_game_state",
    "get_all_game_states",
    "get_game_state_summary",
    "segment_by_time_window",
    "segment_by_time_windows",
    "segment_around_goal",
    "all_goals_context",
    
    # Metrics (flat)
    "average_positions",
    "team_compactness",
    "defensive_line_height",
    "channel_progression",
    "compare_metrics",
    "metric_summary",
    
    # Plots (flat)
    "plot_average_positions",
    "compare_positions",
    "plot_shape_with_metrics",
    "plot_defensive_line",
    "plot_channel_progression",
    "create_comparison_dashboard",
    "save_plot",
    
    # Submodules
    "context",
    "segments",
    "metrics",
    "plots",
]
