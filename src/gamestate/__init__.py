"""
gamestate — Game State Football Analysis Package

Provides tools for:
- Segmenting matches into game-state windows
- Computing team shape & passing network metrics
- Visualising shape and pass networks
"""

#
from .context import (
    summary,
    score_progression,
    substitutions
)

#
# from .segmentation import (
#     segment_by_game_state,
#     segment_by_goal,
#     segment_by_time_window,
#     segment_by_substitution
# )

#
# from .metrics import (
#     compute_average_positions,
#     compute_pass_network
# )

# #
# from .plots import (
#     plot_shape,
#     plot_pass_network
# )

# #
# from .quick import (
#     quick_shape,
#     quick_pass_network
# )


__all__ = [
    "summary",
    "score_progression",
    "substitutions",
#     "segment_by_game_state",
#     "segment_by_goal",
#     "segment_by_time_window",
#     "segment_by_substitution",
#     "compute_average_positions",
#     "compute_pass_network",
#     "plot_shape",
#     "plot_pass_network",
#     "quick_shape",
#     "quick_pass_network",
]
