"""
plots.py - Visualization Functions

Creates publication-quality visualizations for tactical analysis:
1. Average player positions
2. Phase comparisons
3. Defensive block analysis
4. Game state evolution

Uses consistent design system with substitution status indicators.
"""

import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict

from .load_data import get_metadata
from .metrics import average_positions, team_compactness, defensive_line_height


# ============================================================================
# DESIGN SYSTEM
# ============================================================================

COLORS = {
    # Teams
    'home': '#FFFFFF',           # White
    'away': '#4DB2FF',           # Mustard yellow
    
    # Substitution status (outlines)
    'starter': '#FFFFFF',        # White outline - full 90
    'subbed_in': "#4CFF42",      # Green outline - substituted in
    'subbed_out': '#E63946',     # Red outline - substituted out
    
    # Pitch elements
    'pitch_green': '#001400',    # Very dark green
    'pitch_lines': '#FFFFFF',    # White lines
}

PLAYER_SPECS = {
    'size': 1000,                # Marker size (increased for pitch.draw())
    'edge_width_starter': 2.5,   # Edge width
    'edge_width_sub': 2.5,       # Edge width for substitutes
    'alpha': 0.95,               # Marker transparency
    'number_size': 16,           # Jersey number font size (increased for pitch.draw())
    'number_color_home': 'black',  # Black text
    'number_color_away': 'black',  # Black text
    'number_weight': 'bold',
}

PITCH_SPECS = {
    'pitch_type': 'skillcorner',
    'line_alpha': 0.75,
    'pitch_length': 105,
    'pitch_width': 68,
    'linewidth': 1.5,
}

FIGURE_SIZES = {
    'single': (14, 10),
    'comparison_2': (18, 8),
    'comparison_3': (20, 7),
    'timeline': (16, 6),
}


# ============================================================================
# HELPER: CREATE PITCH
# ============================================================================

def create_pitch(show_thirds=False):
    """
    Create mplsoccer pitch with consistent styling.
    
    Args:
        show_thirds: If True, add dotted lines dividing pitch into thirds
    
    Returns:
        Pitch object
    
    Example:
        >>> pitch = create_pitch(show_thirds=True)
        >>> fig, ax = pitch.draw(figsize=(14, 10))
    """
    pitch = Pitch(
        pitch_type=PITCH_SPECS['pitch_type'],
        line_alpha=PITCH_SPECS['line_alpha'],
        pitch_length=PITCH_SPECS['pitch_length'],
        pitch_width=PITCH_SPECS['pitch_width'],
        pitch_color=COLORS['pitch_green'],
        line_color=COLORS['pitch_lines'],
        linewidth=PITCH_SPECS['linewidth'],
    )
    
    return pitch


def add_pitch_thirds(ax):
    """
    Add dotted vertical lines dividing pitch into thirds.
    
    Args:
        ax: Matplotlib axes object
    
    Example:
        >>> pitch = create_pitch()
        >>> fig, ax = pitch.draw()
        >>> add_pitch_thirds(ax)
    """
    # Pitch thirds boundaries (SkillCorner coordinates: -52.5 to 52.5)
    third_positions = [-17.5, 17.5]
    
    for x_pos in third_positions:
        ax.axvline(
            x=x_pos,
            linestyle=':',
            linewidth=2,
            color=COLORS['pitch_lines'],
            alpha=0.6,
            zorder=1
        )


# ============================================================================
# CORE FUNCTION: PLOT AVERAGE POSITIONS
# ============================================================================

def plot_average_positions(
    positions_df: pd.DataFrame,
    title: str = "Average Positions",
    show_thirds: bool = False,
    pitch_color: str = None,
    line_color: str = None,
    team_color: str = None,
    marker_size: int = 1000,
    number_size: int = 16,
    show_numbers: bool = True,
    show_legend: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot average player positions from a positions DataFrame.
    
    Takes the output from average_positions() metric function and creates
    a visualization with player markers showing substitution status.
    
    Player markers show substitution status via edge colors:
    - White outline: Started and played full game
    - Green outline: Substituted in
    - Red outline: Substituted out
    
    Args:
        positions_df: DataFrame from average_positions() with columns:
            - player_id, name, number, position, is_gk, sub_status, team,
            - avg_x, avg_y, std_x, std_y, frames_visible
        title: Plot title
        show_thirds: If True, show pitch thirds dividers
        pitch_color: Pitch background color (if None, uses design default)
        line_color: Pitch line color (if None, uses design default)
        team_color: Team marker color (if None, auto-detects from DataFrame)
        marker_size: Size of player markers in points
        number_size: Font size for jersey numbers
        show_numbers: If True, display jersey numbers on markers
        show_legend: If True, show legend explaining marker colors
        figsize: Figure size as (width, height) in inches
        ax: Matplotlib axes to plot on (if None, creates new figure)
    
    Returns:
        Tuple of (fig, ax): Matplotlib figure and axes objects
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Step 1: Calculate positions (team specified here)
        >>> segment = gs.get_full_match(1886347)
        >>> segment = gs.filter_by_ip_phase(segment, 'build_up', 'home', 1886347)
        >>> positions = gs.average_positions(segment, 'home', 1886347)
        >>> 
        >>> # Step 2: Plot (team auto-detected!)
        >>> fig, ax = gs.plot_average_positions(
        ...     positions,
        ...     title='Build-up Phase',
        ...     show_thirds=True
        ... )
        >>> plt.show()
        >>> 
        >>> # Customize appearance
        >>> fig, ax = gs.plot_average_positions(
        ...     positions,
        ...     title='Custom Title',
        ...     marker_size=1200,
        ...     team_color='#FF5733'  # Override auto-detected color
        ... )
    """
    # Apply defaults from design system
    if pitch_color is None:
        pitch_color = COLORS['pitch_green']
    if line_color is None:
        line_color = COLORS['pitch_lines']
    
    # Validate input
    if positions_df.empty:
        print("⚠️  No player positions to plot")
        return None, None
    
    # Auto-detect team from DataFrame
    if team_color is None:
        if 'team' in positions_df.columns:
            detected_team = positions_df['team'].iloc[0]
            team_color = COLORS['home'] if detected_team == 'home' else COLORS['away']
        else:
            # Fallback to home color if team column missing
            team_color = COLORS['home']
            print("⚠️  'team' column not found in DataFrame, using home color")
    
    # Create pitch with custom colors
    pitch = Pitch(
        pitch_type=PITCH_SPECS['pitch_type'],
        line_alpha=PITCH_SPECS['line_alpha'],
        pitch_length=PITCH_SPECS['pitch_length'],
        pitch_width=PITCH_SPECS['pitch_width'],
        pitch_color=pitch_color,
        line_color=line_color,
        linewidth=PITCH_SPECS['linewidth'],
    )
    
    # Create or use provided axes
    if ax is None:
        fig, ax = pitch.draw(figsize=figsize)
    else:
        pitch.draw(ax=ax)
        fig = ax.figure
    
    # Add pitch thirds if requested
    if show_thirds:
        add_pitch_thirds(ax)
    
    # Determine number color (always black for both teams)
    number_color = 'black'
    
    # Plot each player
    for _, player in positions_df.iterrows():
        # Skip if no position data
        if pd.isna(player['avg_x']) or pd.isna(player['avg_y']):
            continue
        
        # Determine edge color based on substitution status
        sub_status = player.get('sub_status', 'full90')
        
        if sub_status == 'full90':
            edge_color = COLORS['starter']
            edge_width = PLAYER_SPECS['edge_width_starter']
        elif sub_status == 'subbed_in':
            edge_color = COLORS['subbed_in']
            edge_width = PLAYER_SPECS['edge_width_sub']
        elif sub_status == 'subbed_out':
            edge_color = COLORS['subbed_out']
            edge_width = PLAYER_SPECS['edge_width_sub']
        else:
            # Default to starter style if unknown
            edge_color = COLORS['starter']
            edge_width = PLAYER_SPECS['edge_width_starter']
        
        # Plot player marker
        ax.scatter(
            player['avg_x'],
            player['avg_y'],
            c=team_color,
            s=marker_size,
            marker='o',
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=PLAYER_SPECS['alpha'],
            zorder=10  # Match old code
        )
        
        # Add jersey number (if enabled)
        if show_numbers:
            ax.text(
                player['avg_x'],
                player['avg_y'],
                str(int(player['number'])),
                fontsize=number_size,
                color='black',  # Always black
                weight=PLAYER_SPECS['number_weight'],
                ha='center',
                va='center',
                zorder=16  # Match old code
            )
    
    # Add legend
    if show_legend:
        # Detect team for legend label
        team_label = "Team"
        if 'team' in positions_df.columns:
            detected_team = positions_df['team'].iloc[0]
            team_label = f"{'Home' if detected_team == 'home' else 'Away'} Team"
        
        legend_elements = [
            Patch(
                facecolor=team_color,
                edgecolor=COLORS['starter'],
                linewidth=2,
                label=team_label
            ),
            Patch(
                facecolor='white',
                edgecolor=COLORS['starter'],
                linewidth=2,
                label='○ Started & played full'
            ),
            Patch(
                facecolor='white',
                edgecolor=COLORS['subbed_in'],
                linewidth=2.5,
                label='○ Substituted IN'
            ),
            Patch(
                facecolor='white',
                edgecolor=COLORS['subbed_out'],
                linewidth=2.5,
                label='○ Substituted OUT'
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=10,
            framealpha=0.9
        )
    
    # Add title
    if title is None:
        # Auto-generate title
        possession_str = {
            'all': 'All Frames',
            'ip': 'In Possession',
            'oop': 'Out of Possession'
        }.get(possession, possession)
        
        title = f"Average Positions - {possession_str}"
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    return fig, ax


# ============================================================================
# COMPARISON: PHASE COMPARISON
# ============================================================================

def plot_phase_comparison(
    match_id: int,
    team: str,
    phases: List[str] = None,
    possession: str = 'ip',
    show_thirds: bool = False,
    figsize: Tuple[int, int] = None
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Compare average positions across multiple phases side-by-side.
    
    Args:
        match_id: Match identifier
        team: 'home' or 'away'
        phases: List of phase types to compare (default: ['build_up', 'create', 'finish'])
        possession: 'all', 'ip', or 'oop'
        show_thirds: If True, show pitch thirds dividers
        figsize: Figure size (if None, auto-calculated)
    
    Returns:
        Tuple of (fig, axes): Matplotlib figure and list of axes
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Compare attacking phases
        >>> fig, axes = gs.plot_phase_comparison(
        ...     1886347,
        ...     'home',
        ...     phases=['build_up', 'create', 'finish']
        ... )
        >>> plt.show()
    """
    from .segments import get_full_match
    from .filters import filter_by_ip_phase
    from .metrics import average_positions
    
    # Default phases
    if phases is None:
        phases = ['build_up', 'create', 'finish']
    
    # Determine figure size
    if figsize is None:
        if len(phases) == 2:
            figsize = FIGURE_SIZES['comparison_2']
        else:
            figsize = FIGURE_SIZES['comparison_3']
    
    # Create subplots
    fig, axes = plt.subplots(1, len(phases), figsize=figsize, facecolor='white')
    
    # Handle single phase case (axes won't be array)
    if len(phases) == 1:
        axes = [axes]
    
    # Plot each phase
    for ax, phase in zip(axes, phases):
        # Get segment
        segment = get_full_match(match_id=match_id)
        segment = filter_by_ip_phase(
            segment=segment,
            phase_type=phase,
            team=team,
            match_id=match_id
        )
        
        # Calculate positions
        positions = average_positions(
            segment=segment,
            team=team,
            match_id=match_id,
            possession=possession
        )
        
        # Create phase title
        phase_title = phase.replace('_', ' ').title()
        
        # Plot with scaled-down sizes for multi-plot layout
        plot_average_positions(
            positions_df=positions,
            title=phase_title,
            show_thirds=show_thirds,
            show_legend=(ax == axes[0]),  # Only show legend on first plot
            marker_size=500,  # Smaller than default 1000
            number_size=8,   # Smaller than default 16
            ax=ax
        )
    
    # Add overall title
    fig.suptitle(
        f"Phase Comparison - {team.title()} Team",
        fontsize=18,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    
    return fig, axes


# ============================================================================
# COMPARISON: DEFENSIVE BLOCKS
# ============================================================================

def plot_defensive_blocks(
    match_id: int,
    team: str,
    blocks: List[str] = None,
    figsize: Tuple[int, int] = None
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Compare defensive block positions side-by-side.
    
    Args:
        match_id: Match identifier
        team: 'home' or 'away'
        blocks: List of defensive block types (default: ['high_block', 'medium_block', 'low_block'])
        figsize: Figure size (if None, auto-calculated)
    
    Returns:
        Tuple of (fig, axes): Matplotlib figure and list of axes
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Compare defensive structures
        >>> fig, axes = gs.plot_defensive_blocks(
        ...     1886347,
        ...     'home'
        ... )
        >>> plt.show()
    """
    from .segments import get_full_match
    from .filters import filter_by_oop_phase
    from .metrics import average_positions
    
    # Default blocks
    if blocks is None:
        blocks = ['high_block', 'medium_block', 'low_block']
    
    # Determine figure size
    if figsize is None:
        figsize = FIGURE_SIZES['comparison_3']
    
    # Create subplots
    fig, axes = plt.subplots(1, len(blocks), figsize=figsize, facecolor='white')
    
    # Handle single block case
    if len(blocks) == 1:
        axes = [axes]
    
    # Plot each block
    for ax, block in zip(axes, blocks):
        # Get segment
        segment = get_full_match(match_id=match_id)
        segment = filter_by_oop_phase(
            segment=segment,
            phase_type=block,
            team=team,
            match_id=match_id
        )
        
        # Calculate positions (OOP = defending)
        positions = average_positions(
            segment=segment,
            team=team,
            match_id=match_id,
            possession='oop'
        )
        
        # Create block title
        block_title = block.replace('_', ' ').title()
        
        # Plot with scaled-down sizes for multi-plot layout
        plot_average_positions(
            positions_df=positions,
            title=block_title,
            show_thirds=False,
            show_legend=(ax == axes[0]),
            marker_size=500,  # Smaller than default 1000
            number_size=8,   # Smaller than default 16
            ax=ax
        )
    
    # Add overall title
    fig.suptitle(
        f"Defensive Block Comparison - {team.title()} Team",
        fontsize=18,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    
    return fig, axes


# ============================================================================
# TIMELINE: GAME STATE EVOLUTION
# ============================================================================

def plot_game_state_evolution(
    match_id: int,
    team: str,
    metric: str = 'width',
    window_minutes: int = 15,
    possession: str = 'all',
    figsize: Tuple[int, int] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot how a tactical metric evolves over time windows.
    
    Args:
        match_id: Match identifier
        team: 'home' or 'away'
        metric: Metric to track
            - 'width': Team width (compactness)
            - 'depth': Team depth (compactness)
            - 'compactness': Average distance from centroid
            - 'defensive_line': Defensive line height (avg)
        window_minutes: Size of time windows in minutes
        possession: 'all', 'ip', or 'oop' (filter possession state)
        figsize: Figure size (if None, uses default)
    
    Returns:
        Tuple of (fig, ax): Matplotlib figure and axes
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Track team width evolution
        >>> fig, ax = gs.plot_game_state_evolution(
        ...     match_id=1886347,
        ...     team='home',
        ...     metric='width',
        ...     window_minutes=15
        ... )
        >>> plt.show()
        >>> 
        >>> # Track defensive line when defending
        >>> fig, ax = gs.plot_game_state_evolution(
        ...     match_id=1886347,
        ...     team='home',
        ...     metric='defensive_line',
        ...     possession='oop'
        ... )
    """
    from .segments import segment_by_time_windows
    
    # Get time windows
    windows = segment_by_time_windows(match_id=match_id, window_minutes=window_minutes)
    
    # Calculate metrics for each window
    periods = []
    values = []
    
    for period_label, segment in windows.items():
        # Calculate metric based on type
        if metric in ['width', 'depth', 'compactness']:
            result = team_compactness(
                segment=segment,
                team=team,
                match_id=match_id,
                possession=possession
            )
            metric_value = result.get(metric)
        elif metric == 'defensive_line':
            result = defensive_line_height(
                segment=segment,
                team=team,
                match_id=match_id,
                possession=possession
            )
            metric_value = result.get('avg_defensive_line_x')
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'width', 'depth', 'compactness', or 'defensive_line'")
        
        periods.append(period_label)
        values.append(metric_value)
    
    # Create figure
    if figsize is None:
        figsize = FIGURE_SIZES['timeline']
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Determine team color (use darker versions for visibility on white background)
    if team == 'home':
        line_color = '#2E86AB'  # Blue (visible on white)
    else:
        line_color = '#F4A300'  # Mustard (visible on white)
    
    # Plot line
    x_positions = list(range(len(periods)))
    ax.plot(
        x_positions,
        values,
        marker='o',
        markersize=10,
        linewidth=3,
        color=line_color,
        markerfacecolor=line_color,
        markeredgecolor='white',
        markeredgewidth=2,
        zorder=3
    )
    
    # Fill between
    ax.fill_between(
        x_positions,
        values,
        alpha=0.2,
        color=line_color,
        zorder=1
    )
    
    # Styling
    ax.set_xlabel('Match Period', fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(periods, rotation=45 if len(periods) > 6 else 0, ha='right' if len(periods) > 6 else 'center')
    
    # Y-label based on metric
    metric_labels = {
        'width': 'Team Width (m)',
        'depth': 'Team Depth (m)',
        'compactness': 'Compactness (m)',
        'defensive_line': 'Defensive Line Height (m from goal)'
    }
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
    
    # Title
    possession_str = {
        'all': 'All Phases',
        'ip': 'In Possession',
        'oop': 'Out of Possession'
    }.get(possession, possession)
    
    ax.set_title(
        f"{metric_labels.get(metric, metric)} Evolution - {team.title()} Team ({possession_str})",
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    
    # Add value annotations on points
    for i, (x, y) in enumerate(zip(x_positions, values)):
        ax.annotate(
            f'{y:.1f}',
            xy=(x, y),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            color=line_color,
            weight='bold'
        )
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================
# AGGREGATE METRICS: TEAM COMPACTNESS (ON PITCH)
# ============================================================================

def plot_team_compactness(
    positions_df: pd.DataFrame,
    compactness_metrics: Dict[str, float],
    title: str = "Team Shape and Compactness",
    show_thirds: bool = False,
    pitch_color: str = None,
    line_color: str = None,
    figsize: Tuple[int, int] = (14, 10)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot average positions with lines connecting players to show team shape and compactness.
    
    Shows:
    - Player positions
    - Lines connecting all players (convex hull outline)
    - Team centroid marker
    - Compactness metrics in annotation
    
    Args:
        positions_df: DataFrame from average_positions()
        compactness_metrics: Dict from team_compactness() containing:
            - 'centroid_x', 'centroid_y': Team center point
            - 'width', 'depth': Team dimensions
            - 'area': Convex hull area
            - 'compactness': Average distance from centroid
        title: Plot title
        show_thirds: If True, show pitch thirds dividers
        pitch_color: Pitch background color (if None, uses default)
        line_color: Pitch line color (if None, uses default)
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Tuple of (fig, ax): Matplotlib figure and axes
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Calculate metrics
        >>> segment = gs.get_full_match(1886347)
        >>> positions = gs.average_positions(segment, 'home', 1886347, possession='oop')
        >>> compactness = gs.team_compactness(segment, 'home', 1886347, possession='oop')
        >>> 
        >>> # Plot
        >>> fig, ax = gs.plot_team_compactness(
        ...     positions_df=positions,
        ...     compactness_metrics=compactness,
        ...     title='Team Compactness - Defending'
        ... )
        >>> plt.show()
    """
    from scipy.spatial import ConvexHull
    
    # Create base plot with positions
    fig, ax = plot_average_positions(
        positions_df=positions_df,
        title=title,
        show_thirds=show_thirds,
        pitch_color=pitch_color,
        line_color=line_color,
        show_legend=True,  # Keep legend for sub status
        figsize=figsize
    )
    
    # Get field players (exclude GK)
    field_players = positions_df[~positions_df['is_gk']].copy()
    
    if len(field_players) >= 3:
        # Get player coordinates
        x_coords = field_players['avg_x'].values
        y_coords = field_players['avg_y'].values
        points = np.column_stack([x_coords, y_coords])
        
        # Draw convex hull (connecting lines around all players)
        try:
            hull = ConvexHull(points)
            
            # Draw lines connecting the hull vertices
            for simplex in hull.simplices:
                ax.plot(
                    points[simplex, 0],
                    points[simplex, 1],
                    color='#F4A300',  # Yellow/Gold
                    linewidth=2.5,
                    linestyle='--',
                    alpha=0.7,
                    zorder=2
                )
        except Exception as e:
            print(f"Could not draw convex hull: {e}")
    
    # Extract centroid
    centroid_x = compactness_metrics.get('centroid_x')
    centroid_y = compactness_metrics.get('centroid_y')
    
    if centroid_x is not None and centroid_y is not None:
        # Draw centroid marker
        ax.scatter(
            centroid_x,
            centroid_y,
            marker='X',
            s=600,
            c='#F4A300',
            edgecolors='black',
            linewidths=3,
            label='Team Centroid',
            zorder=15,
            alpha=0.9
        )
    
    # Add metrics annotation
    width = compactness_metrics.get('width')
    depth = compactness_metrics.get('depth')
    area = compactness_metrics.get('area')
    compactness = compactness_metrics.get('compactness')
    
    if area is not None and compactness is not None:
        metrics_text = (
            f"Width: {width:.1f}m\n"
            f"Depth: {depth:.1f}m\n"
            f"Area: {area:.0f} m²\n"
            f"Compactness: {compactness:.1f}m"
        )
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.9,
                edgecolor='#F4A300',
                linewidth=2
            ),
            weight='bold',
            zorder=20
        )
    
    return fig, ax


# ============================================================================
# AGGREGATE METRICS: DEFENSIVE LINE HEIGHT (ON PITCH)
# ============================================================================

def plot_defensive_line(
    positions_df: pd.DataFrame,
    defensive_line_metrics: Dict[str, float],
    title: str = "Defensive Line Height",
    show_thirds: bool = False,
    pitch_color: str = None,
    line_color: str = None,
    figsize: Tuple[int, int] = (14, 10)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot average positions with defensive line metrics overlaid as vertical lines.
    
    Shows:
    - Player positions
    - Average defensive line (dotted red line)
    - Median defensive line (dotted orange line)
    - Deepest defender position (dotted yellow line)
    
    Args:
        positions_df: DataFrame from average_positions()
        defensive_line_metrics: Dict from defensive_line_height() containing:
            - 'avg_defensive_line_x': Average line position
            - 'median_defensive_line_x': Median line position
            - 'deepest_defender_x': Deepest defender position
        title: Plot title
        show_thirds: If True, show pitch thirds dividers
        pitch_color: Pitch background color (if None, uses default)
        line_color: Pitch line color (if None, uses default)
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Tuple of (fig, ax): Matplotlib figure and axes
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Calculate metrics
        >>> segment = gs.get_full_match(1886347)
        >>> segment = gs.filter_by_oop_phase(segment, 'low_block', 'home', 1886347)
        >>> positions = gs.average_positions(segment, 'home', 1886347, possession='oop')
        >>> def_line = gs.defensive_line_height(segment, 'home', 1886347, possession='oop')
        >>> 
        >>> # Plot
        >>> fig, ax = gs.plot_defensive_line(
        ...     positions_df=positions,
        ...     defensive_line_metrics=def_line,
        ...     title='Defensive Line - Low Block'
        ... )
        >>> plt.show()
    """
    # Create base plot with positions
    fig, ax = plot_average_positions(
        positions_df=positions_df,
        title=title,
        show_thirds=show_thirds,
        pitch_color=pitch_color,
        line_color=line_color,
        show_legend=False,  # We'll add custom legend
        figsize=figsize
    )
    
    # Extract metrics
    avg_line = defensive_line_metrics.get('avg_defensive_line_x')
    median_line = defensive_line_metrics.get('median_defensive_line_x')
    deepest = defensive_line_metrics.get('deepest_defender_x')
    
    # Draw vertical lines for each metric
    if avg_line is not None:
        ax.axvline(
            avg_line,
            color='#F4A300',  # Orange
            linestyle=':',
            linewidth=3,
            alpha=0.8,
            label=f'Avg: {avg_line:.1f}m',
            zorder=2
        )
    
    if median_line is not None:
        ax.axvline(
            median_line,
            color="#00E8F4",  # Cyan
            linestyle=':',
            linewidth=3,
            alpha=0.8,
            label=f'Median: {median_line:.1f}m',
            zorder=2
        )
    
    if deepest is not None:
        ax.axvline(
            deepest,
            color="#E63946",  # Red
            linestyle=':',
            linewidth=3,
            alpha=0.8,
            label=f'Deepest: {deepest:.1f}m',
            zorder=2
        )
    
    # Add legend
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    
    return fig, ax


# ============================================================================
# AGGREGATE METRICS: CHANNEL PROGRESSION (BAR CHART)
# ============================================================================

def plot_channel_progression(
    channel_metrics: Dict[str, float],
    title: str = "Channel Progression Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 3-way bar chart showing progression counts through Left/Center/Right channels.
    
    Args:
        channel_metrics: Dict from channel_progression() containing:
            - 'left_count': Number of progressions in left channel
            - 'center_count': Number of progressions in center channel
            - 'right_count': Number of progressions in right channel
            - 'left_pct', 'center_pct', 'right_pct': Percentages (optional)
        title: Plot title
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Tuple of (fig, ax): Matplotlib figure and axes
    
    Example:
        >>> import gamestate as gs
        >>> 
        >>> # Calculate metrics
        >>> segment = gs.get_full_match(1886347)
        >>> channels = gs.channel_progression(segment, 'home', 1886347, possession='ip')
        >>> 
        >>> # Plot
        >>> fig, ax = gs.plot_channel_progression(
        ...     channel_metrics=channels,
        ...     title='Attack Patterns by Channel'
        ... )
        >>> plt.show()
    """
    # Extract data
    left_count = channel_metrics.get('left_count', 0)
    center_count = channel_metrics.get('center_count', 0)
    right_count = channel_metrics.get('right_count', 0)
    
    left_pct = channel_metrics.get('left_pct', 0)
    center_pct = channel_metrics.get('center_pct', 0)
    right_pct = channel_metrics.get('right_pct', 0)
    
    # Prepare data
    channels = ['Left', 'Center', 'Right']
    counts = [left_count, center_count, right_count]
    percentages = [left_pct, center_pct, right_pct]
    colors = ['#E63946', '#F4A300', '#06A77D']  # Red, Yellow, Green
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create bar chart
    bars = ax.bar(
        channels,
        counts,
        color=colors,
        alpha=0.8,
        edgecolor='white',
        linewidth=2,
        width=0.6
    )
    
    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(counts) * 0.02,
            f'{int(count)}\n({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=12,
            weight='bold'
        )
    
    # Styling
    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Progressions', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, max(counts) * 1.15)  # Add space for labels
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================
# UTILITY: SETUP PLOT STYLE
# ============================================================================

def setup_plot_style():
    """
    Apply consistent matplotlib style settings.
    
    Call this once at the start of your analysis to ensure
    all plots have consistent styling.
    
    Example:
        >>> import gamestate as gs
        >>> gs.setup_plot_style()
        >>> # Now all plots will use consistent styling
    """
    import matplotlib.pyplot as plt
    
    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    
    # Figure settings
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Axes settings
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    
    # Tick settings
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Legend settings
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.framealpha'] = 0.9
    
    # Grid settings
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    print("✅ Plot style configured!")