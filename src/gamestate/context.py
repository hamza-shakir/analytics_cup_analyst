import pandas as pd
import numpy as np

from src.gamestate.load_data import (
    load_enriched_tracking_data,
    load_event_data
)


#
def summary(match_id):
    # load data
    enriched_tracking_data = load_enriched_tracking_data(match_id)
    goal_event_df = load_event_data(match_id, enriched_tracking_data)

    pass


#
def score_progression(match_id):
    # load data
    enriched_tracking_data = load_enriched_tracking_data(match_id)
    goal_event_df = load_event_data(match_id, enriched_tracking_data)

    sp_df = goal_event_df[[
        "home_team.name",
        "home_team_score",
        "away_team_score",
        "away_team.name",
        "minute_start",
        "frame_start"
    ]]

    return sp_df


#
def substitutions(match_id):
    # Load and filter in one step
    enriched_tracking_data = load_enriched_tracking_data(match_id)
    
    sub_df = enriched_tracking_data[
        enriched_tracking_data['appearance_type'].isin(['sub_in', 'sub_out'])
    ].copy()
    
    if sub_df.empty:
        return pd.DataFrame(columns=["team_name", "sub_minute", "sub_out_id", "sub_out_name", 
                                      "sub_out_playtime", "sub_in_id", "sub_in_name",
                                      "sub_in_playtime"])
    
    # Split and prepare in one pass
    sub_out_df = (
        sub_df[sub_df['appearance_type'] == 'sub_out']
        [['team_name', 'player_id', 'short_name', 'end_time_minute', 'playtime_minutes']]
        .rename(columns={
            "player_id": "sub_out_id",
            "short_name": "sub_out_name",
            "end_time_minute": "sub_out_minute",
            "playtime_minutes": "sub_out_playtime"
        })
        .assign(sub_out_minute=lambda x: pd.to_numeric(x['sub_out_minute'], errors='coerce'))
    )
    
    sub_in_df = (
        sub_df[sub_df['appearance_type'] == 'sub_in']
        [['team_name', 'player_id', 'short_name', 'start_time_minute', 'playtime_minutes']]
        .rename(columns={
            "player_id": "sub_in_id",
            "short_name": "sub_in_name",
            "start_time_minute": "sub_in_minute",
            "playtime_minutes": "sub_in_playtime"
        })
        .assign(sub_in_minute=lambda x: pd.to_numeric(x['sub_in_minute'], errors='coerce'))
    )
    
    # Pair per team using nearest-minute matching
    tolerance_minutes = 2
    paired_list = []
    teams = sorted(set(sub_out_df['team_name'].unique()).union(sub_in_df['team_name'].unique()))
    
    for team in teams:
        o = sub_out_df[sub_out_df['team_name'] == team].sort_values('sub_out_minute')
        i = sub_in_df[sub_in_df['team_name'] == team].sort_values('sub_in_minute')

        if o.empty and i.empty:
            continue
        if o.empty:
            merged = i.copy()
            merged['sub_out_id'] = pd.NA
            merged['sub_out_name'] = pd.NA
            merged['sub_out_minute'] = pd.NA
            merged['sub_out_playtime'] = pd.NA
        elif i.empty:
            merged = o.copy()
            merged['sub_in_id'] = pd.NA
            merged['sub_in_name'] = pd.NA
            merged['sub_in_minute'] = pd.NA
            merged['sub_in_playtime'] = pd.NA
        else:
            merged = pd.merge_asof(
                o,
                i,
                left_on='sub_out_minute',
                right_on='sub_in_minute',
                by='team_name',
                direction='nearest',
                tolerance=tolerance_minutes,
            )

        paired_list.append(merged)

    # Combine results
    if paired_list:
        subs_paired = pd.concat(paired_list, ignore_index=True, sort=False)
    else:
        subs_paired = pd.DataFrame()
    
    # Select and order columns
    result = subs_paired[[
        "team_name", "sub_out_id", "sub_out_name", "sub_out_minute", 
        "sub_out_playtime", "sub_in_name", "sub_in_id", "sub_in_minute", 
        "sub_in_playtime"
    ]].copy()
    
    # Combine sub minutes into single column (they should be the same after merge_asof)
    result['sub_minute'] = result['sub_out_minute'].fillna(result['sub_in_minute'])
    
    # Remove individual minute columns and reorder
    result = result[[
        "team_name", "sub_minute", "sub_out_id", "sub_out_name", "sub_out_playtime",
        "sub_in_id", "sub_in_name", "sub_in_playtime"
    ]]
    
    # Remove duplicates
    result = result.drop_duplicates(subset=['team_name', 'sub_out_id', 'sub_in_id'])
    
    # Sort and reset index
    result = result.sort_values(
        ["team_name", "sub_minute"], 
        na_position='last'
    ).reset_index(drop=True)
    
    return result