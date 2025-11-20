"""
plots.py v2 - Tactical Visualization Functions

Creates professional pitch visualizations using mplsoccer:
1. Average position plots (shape maps)
2. Comparison plots (side-by-side, matrix)
3. Metric overlays (width/depth indicators)

Uses mplsoccer's Pitch class for SkillCorner coordinates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from typing import Union, List, Dict, Tuple, Optional

try:
    from mplsoccer import Pitch
    MPLSOCCER_AVAILABLE = True
except ImportError:
    MPLSOCCER_AVAILABLE = False
    print("Warning: mplsoccer not installed. Install with: pip install mplsoccer")


# ----------------------------------------------------------
# 1. Single Shape Plot
# ----------------------------------------------------------

def plot_average_positions(
    positions_df: pd.DataFrame,
    title: str = "Average Positions",
    pitch_color: str = "#001400",
    line_color: str = "white",
    player_color: str = "#4DB2FF",
    figsize: Tuple[int, int] = (10, 7),
    show_numbers: bool = True,
    show_names: bool = False,
    marker_size: int = 600,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot average player positions on a pitch.
    
    Args:
        positions_df: DataFrame from metrics.average_positions()
                     Must have: avg_x, avg_y, number, name
        title: Plot title
        pitch_color: Background color
        line_color: Line color
        player_color: Player marker color
        figsize: Figure size (width, height)
        show_numbers: Show jersey numbers on markers
        show_names: Show player names below markers
        marker_size: Size of player markers
        ax: Existing axes (if None, creates new figure)
    
    Returns:
        (fig, ax) tuple
    
    Example:
        >>> positions = metrics.average_positions(segment, team='home')
        >>> fig, ax = plot_average_positions(positions, title="Drawing 0-0")
        >>> plt.show()
    """
    if not MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required. Install with: pip install mplsoccer")
    
    # Create pitch (SkillCorner uses 105m x 68m)
    pitch = Pitch(
        pitch_type='skillcorner',
        pitch_length=105,
        pitch_width=68,
        pitch_color=pitch_color,
        line_color=line_color,
        linewidth=1.5,
        line_alpha=0.75
    )
    
    # Create figure if not provided
    if ax is None:
        fig, ax = pitch.draw(figsize=figsize)
    else:
        fig = ax.figure
        pitch.draw(ax=ax)
    
    # Plot players
    scatter = ax.scatter(
        positions_df['avg_x'],
        positions_df['avg_y'],
        c=player_color,
        s=marker_size,
        edgecolors='white',
        linewidths=2.5,
        zorder=10,
        alpha=0.95
    )
    
    # Add jersey numbers
    if show_numbers:
        for idx, row in positions_df.iterrows():
            if pd.notna(row['number']):
                ax.text(
                    row['avg_x'],
                    row['avg_y'],
                    str(int(row['number'])),
                    color='black',
                    fontweight='bold',
                    fontsize=10,
                    ha='center',
                    va='center',
                    zorder=15
                )
    
    # Add player names
    if show_names:
        for idx, row in positions_df.iterrows():
            if pd.notna(row['name']):
                ax.text(
                    row['avg_x'],
                    row['avg_y'] - 3,
                    row['name'],
                    color=line_color,
                    fontsize=7,
                    ha='center',
                    va='top',
                    zorder=15
                )
    
    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
    
    return fig, ax


# ----------------------------------------------------------
# 2. Comparison Plots
# ----------------------------------------------------------

def compare_positions(
    positions_list: List[pd.DataFrame],
    titles: List[str],
    layout: Tuple[int, int] = None,
    figsize: Tuple[int, int] = None,
    pitch_color: str = "#001400",
    player_colors: List[str] = None,
    suptitle: str = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Compare average positions across multiple segments (side-by-side or grid).
    
    Args:
        positions_list: List of position DataFrames
        titles: List of titles (one per segment)
        layout: (rows, cols) for subplot layout. If None, auto-determines
        figsize: Figure size. If None, auto-sizes based on layout
        pitch_color: Background color
        player_colors: List of colors (one per segment). If None, uses default
        suptitle: Main title for entire figure
    
    Returns:
        (fig, axes) tuple
    
    Example:
        >>> drawing_pos = metrics.average_positions(drawing_segment, team='home')
        >>> winning_pos = metrics.average_positions(winning_segment, team='home')
        >>> fig, axes = compare_positions(
        ...     [drawing_pos, winning_pos],
        ...     titles=["Drawing 0-0", "Winning 1-0"],
        ...     layout=(1, 2)
        ... )
    """
    if not MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required. Install with: pip install mplsoccer")
    
    n_plots = len(positions_list)
    
    # Auto-determine layout if not provided
    if layout is None:
        if n_plots <= 3:
            layout = (1, n_plots)
        elif n_plots == 4:
            layout = (2, 2)
        elif n_plots <= 6:
            layout = (2, 3)
        else:
            layout = (3, 3)
    
    rows, cols = layout
    
    # Auto-size figure
    if figsize is None:
        figsize = (cols * 6, rows * 5)
    
    # Default colors
    if player_colors is None:
        player_colors = ['#4DB2FF'] * n_plots
    
    # Create pitch
    pitch = Pitch(
        pitch_type='skillcorner',
        pitch_color=pitch_color,
        line_color='white',
        linewidth=1.5,
        line_alpha=0.75
    )
    
    # Create subplots
    fig, axes = pitch.draw(nrows=rows, ncols=cols, figsize=figsize)
    
    # Flatten axes for easier iteration
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # Plot each segment
    for i, (positions, title, color) in enumerate(zip(positions_list, titles, player_colors)):
        if i < len(axes_flat):
            ax = axes_flat[i]
            
            # Plot positions
            plot_average_positions(
                positions,
                title=title,
                pitch_color=pitch_color,
                player_color=color,
                ax=ax,
                show_numbers=True
            )
    
    # Hide unused subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].remove()
    
    # Add super title
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout()
    
    return fig, axes


# ----------------------------------------------------------
# 3. Shape with Metrics Overlay
# ----------------------------------------------------------

def plot_shape_with_metrics(
    positions_df: pd.DataFrame,
    compactness_metrics: Dict,
    title: str = "Team Shape with Metrics",
    pitch_color: str = "#001400",
    show_centroid: bool = True,
    show_hull: bool = True,
    show_width_depth: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot average positions with compactness metrics overlaid.
    
    Shows:
    - Player positions
    - Team centroid
    - Convex hull (team area)
    - Width and depth lines
    
    Args:
        positions_df: DataFrame from metrics.average_positions()
        compactness_metrics: Dict from metrics.team_compactness()
        title: Plot title
        pitch_color: Background color
        show_centroid: Show team center point
        show_hull: Show convex hull outline
        show_width_depth: Show width/depth indicators
    
    Returns:
        (fig, ax) tuple
    
    Example:
        >>> positions = metrics.average_positions(segment, team='home')
        >>> compactness = metrics.team_compactness(segment, team='home')
        >>> fig, ax = plot_shape_with_metrics(positions, compactness)
    """
    # Base plot
    fig, ax = plot_average_positions(positions_df, title=title, pitch_color=pitch_color)
    
    # Exclude goalkeepers for metrics overlay
    field_players = positions_df[~positions_df['is_gk']]
    
    if len(field_players) < 3:
        return fig, ax
    
    x_coords = field_players['avg_x'].values
    y_coords = field_players['avg_y'].values
    
    # Show centroid
    if show_centroid:
        centroid_x = compactness_metrics['centroid_x']
        centroid_y = compactness_metrics['centroid_y']
        
        ax.scatter(
            centroid_x, centroid_y,
            c='yellow', s=200, marker='x',
            linewidths=3, zorder=20,
            label='Team Centroid'
        )
    
    # Show convex hull
    if show_hull:
        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack([x_coords, y_coords])
            hull = ConvexHull(points)
            
            # Draw hull
            for simplex in hull.simplices:
                ax.plot(
                    points[simplex, 0], points[simplex, 1],
                    'yellow', linestyle='--', linewidth=1.5,
                    alpha=0.6, zorder=5
                )
        except:
            pass
    
    # Show width and depth lines
    if show_width_depth:
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        
        # Depth line (vertical)
        ax.plot([min_x, max_x], [0, 0], 
                color='red', linestyle=':', linewidth=2, 
                alpha=0.7, label=f"Depth: {compactness_metrics['depth']:.1f}m")
        
        # Width line (horizontal)
        ax.plot([0, 0], [min_y, max_y],
                color='cyan', linestyle=':', linewidth=2,
                alpha=0.7, label=f"Width: {compactness_metrics['width']:.1f}m")
    
    # Add legend
    ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    
    # Add metrics text
    metrics_text = f"Area: {compactness_metrics['area']:.0f}m²\n"
    metrics_text += f"Compactness: {compactness_metrics['compactness']:.1f}m"
    
    ax.text(
        0.02, 0.02, metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    return fig, ax


# ----------------------------------------------------------
# 4. Defensive Line Visualization
# ----------------------------------------------------------

def plot_defensive_line(
    positions_df: pd.DataFrame,
    defensive_line_metrics: Dict,
    title: str = "Defensive Line Height",
    pitch_color: str = "#001400"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot average positions with defensive line highlighted.
    
    Args:
        positions_df: DataFrame from metrics.average_positions()
        defensive_line_metrics: Dict from metrics.defensive_line_height()
        title: Plot title
        pitch_color: Background color
    
    Returns:
        (fig, ax) tuple
    
    Example:
        >>> positions = metrics.average_positions(segment, team='home')
        >>> def_line = metrics.defensive_line_height(segment, team='home')
        >>> fig, ax = plot_defensive_line(positions, def_line)
    """
    # Base plot
    fig, ax = plot_average_positions(positions_df, title=title, pitch_color=pitch_color)
    
    # Get defensive line metrics
    avg_line_x = defensive_line_metrics.get('avg_defensive_line_x')
    deepest_x = defensive_line_metrics.get('deepest_defender_x')
    
    if avg_line_x is not None and deepest_x is not None:
        # Draw average defensive line
        ax.axvline(
            avg_line_x, 
            color='red', linestyle='--', linewidth=2.5,
            alpha=0.8, label=f'Avg Line: {avg_line_x:.1f}m', zorder=2
        )
        
        # Draw deepest defender line
        ax.axvline(
            deepest_x,
            color='orange', linestyle=':', linewidth=2,
            alpha=0.6, label=f'Deepest: {deepest_x:.1f}m', zorder=2
        )
        
        # Highlight defenders
        defenders = positions_df[
            (~positions_df['is_gk']) &
            (positions_df['position'].notna()) &
            (positions_df['position'].str.contains('Back|Defender', case=False, na=False))
        ]
        
        if not defenders.empty:
            ax.scatter(
                defenders['avg_x'],
                defenders['avg_y'],
                s=800, facecolors='none',
                edgecolors='red', linewidths=3,
                zorder=12, label='Defenders'
            )
        
        # Legend
        ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    
    return fig, ax


# ----------------------------------------------------------
# 5. Channel Analysis Visualization
# ----------------------------------------------------------

def plot_channel_progression(
    channel_metrics: Dict,
    title: str = "Attack Patterns by Channel",
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize attack patterns through pitch channels.
    
    Args:
        channel_metrics: Dict from metrics.channel_progression()
        title: Plot title
        figsize: Figure size
    
    Returns:
        (fig, ax) tuple
    
    Example:
        >>> channels = metrics.channel_progression(segment, team='home')
        >>> fig, ax = plot_channel_progression(channels)
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Data
    channels = ['Left', 'Center', 'Right']
    percentages = [
        channel_metrics['left_pct'],
        channel_metrics['center_pct'],
        channel_metrics['right_pct']
    ]
    counts = [
        channel_metrics['left_count'],
        channel_metrics['center_count'],
        channel_metrics['right_count']
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Bar chart
    bars = ax.bar(channels, percentages, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add percentage labels on bars
    for bar, pct, count in zip(bars, percentages, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, height,
            f'{pct:.1f}%\n({count})',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            color='white'
        )
    
    # Styling
    ax.set_ylabel('Percentage of Forward Progressions', fontsize=12, color='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
    ax.set_ylim(0, max(percentages) * 1.2)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='white')
    ax.set_axisbelow(True)
    
    # Style ticks
    ax.tick_params(colors='white', labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    return fig, ax


# ----------------------------------------------------------
# 6. Complete Comparison Dashboard
# ----------------------------------------------------------

def create_comparison_dashboard(
    segment1_data: Dict,
    segment2_data: Dict,
    labels: Tuple[str, str] = ("Segment 1", "Segment 2"),
    figsize: Tuple[int, int] = (16, 10)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create comprehensive comparison dashboard.
    
    Shows side-by-side:
    - Average positions
    - Team compactness metrics
    - Defensive line comparison
    - Channel progression patterns
    
    Args:
        segment1_data: Dict with keys: 'positions', 'compactness', 'def_line', 'channels'
        segment2_data: Dict with same keys
        labels: Labels for the two segments
        figsize: Figure size
    
    Returns:
        (fig, axes) tuple
    
    Example:
        >>> # Calculate all metrics for both segments
        >>> seg1 = {
        ...     'positions': metrics.average_positions(drawing, team='home'),
        ...     'compactness': metrics.team_compactness(drawing, team='home'),
        ...     'def_line': metrics.defensive_line_height(drawing, team='home'),
        ...     'channels': metrics.channel_progression(drawing, team='home')
        ... }
        >>> seg2 = { ... }  # Same for winning segment
        >>> fig, axes = create_comparison_dashboard(seg1, seg2, 
        ...                                          labels=("Drawing", "Winning"))
    """
    fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
    
    # Create grid: 2 rows x 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top row: Average positions
    ax1 = fig.add_subplot(gs[0, 0])
    plot_average_positions(
        segment1_data['positions'],
        title=f"{labels[0]} - Average Positions",
        ax=ax1
    )
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_average_positions(
        segment2_data['positions'],
        title=f"{labels[1]} - Average Positions",
        ax=ax2,
        player_color='#FF6B6B'
    )
    
    # Bottom left: Compactness comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a1a')
    
    metrics_to_compare = ['width', 'depth', 'area', 'compactness']
    seg1_values = [segment1_data['compactness'][m] for m in metrics_to_compare]
    seg2_values = [segment2_data['compactness'][m] for m in metrics_to_compare]
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, seg1_values, width, label=labels[0], color='#4DB2FF', alpha=0.8)
    bars2 = ax3.bar(x + width/2, seg2_values, width, label=labels[1], color='#FF6B6B', alpha=0.8)
    
    ax3.set_ylabel('Value', fontsize=11, color='white')
    ax3.set_title('Team Compactness Comparison', fontsize=12, fontweight='bold', color='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.title() for m in metrics_to_compare], rotation=45, ha='right')
    ax3.legend(fontsize=9)
    ax3.tick_params(colors='white')
    ax3.grid(axis='y', alpha=0.3)
    
    for spine in ax3.spines.values():
        spine.set_edgecolor('white')
    
    # Bottom right: Channel comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1a1a1a')
    
    channels = ['Left', 'Center', 'Right']
    seg1_channel_pcts = [
        segment1_data['channels']['left_pct'],
        segment1_data['channels']['center_pct'],
        segment1_data['channels']['right_pct']
    ]
    seg2_channel_pcts = [
        segment2_data['channels']['left_pct'],
        segment2_data['channels']['center_pct'],
        segment2_data['channels']['right_pct']
    ]
    
    x_chan = np.arange(len(channels))
    bars1_chan = ax4.bar(x_chan - width/2, seg1_channel_pcts, width, 
                         label=labels[0], color='#4DB2FF', alpha=0.8)
    bars2_chan = ax4.bar(x_chan + width/2, seg2_channel_pcts, width,
                         label=labels[1], color='#FF6B6B', alpha=0.8)
    
    ax4.set_ylabel('Percentage', fontsize=11, color='white')
    ax4.set_title('Attack Patterns Comparison', fontsize=12, fontweight='bold', color='white')
    ax4.set_xticks(x_chan)
    ax4.set_xticklabels(channels)
    ax4.legend(fontsize=9)
    ax4.tick_params(colors='white')
    ax4.grid(axis='y', alpha=0.3)
    
    for spine in ax4.spines.values():
        spine.set_edgecolor('white')
    
    # Main title
    fig.suptitle('Tactical Comparison Dashboard', 
                 fontsize=16, fontweight='bold', color='white', y=0.98)
    
    return fig, np.array([[ax1, ax2], [ax3, ax4]])


# ----------------------------------------------------------
# Helper: Save Figure
# ----------------------------------------------------------

def save_plot(fig: plt.Figure, filename: str, dpi: int = 300):
    """
    Save plot to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename (with extension)
        dpi: Resolution
    
    Example:
        >>> fig, ax = plot_average_positions(positions)
        >>> save_plot(fig, 'shape_drawing.png')
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✓ Saved: {filename}")