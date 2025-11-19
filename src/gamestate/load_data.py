import pandas as pd
import numpy as np
import requests


# ----------------------------------------------------------
# internal caches (module-level)
# persists as long as Python process lives
# ----------------------------------------------------------

_tracking_cache = {}
_meta_cache = {}
_enriched_cache = {}
_events_cache = {}


#
def time_to_seconds(time_str):
    if time_str is None:
        return 90 * 60  # 120 minutes = 7200 seconds
    h, m, s = map(int, time_str.split(":"))

    return h * 3600 + m * 60 + s


#
def time_to_minutes(time_str):
    if time_str is None:
        return 90  # 90 minutes
    h, m, s = map(int, time_str.split(":"))
    return h * 60 + m + round(s / 60)


#
def load_tracking_data(match_id):
    # Check cache first, in case it was already loaded in previous calls
    if match_id in _tracking_cache:
        return _tracking_cache[match_id]
    
    # Load tracking data from GitHub URL
    tracking_data_github_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"  # Data is stored using GitLFS
    raw_data = pd.read_json(tracking_data_github_url, lines=True)


    # This is common
    raw_df = pd.json_normalize(
        raw_data.to_dict("records"),
        "player_data",
        ["frame", "timestamp", "period", "possession", "ball_data"],
    )

    # Extract 'player_id' and 'group from the 'possession' dictionary
    raw_df["possession_player_id"] = raw_df["possession"].apply(
        lambda x: x.get("player_id")
    )
    raw_df["possession_group"] = raw_df["possession"].apply(lambda x: x.get("group"))


    # (Optional) Expand the ball_data with json_normalize
    raw_df[["ball_x", "ball_y", "ball_z", "is_detected_ball"]] = pd.json_normalize(
        raw_df.ball_data
    )


    # (Optional) Drop the original 'possession' column if you no longer need it
    raw_df = raw_df.drop(columns=["possession", "ball_data"])

    # Add the match_id identifier to your dataframe
    raw_df["match_id"] = match_id
    tracking_df = raw_df.copy()

    # Store in cache, if this is the first session-run
    _tracking_cache[match_id] = tracking_df
    
    return tracking_df


#
def load_meta_data(match_id):
    # Check cache first, in case it was already loaded in previous calls
    if match_id in _meta_cache:
        return _meta_cache[match_id]

    # Load meta data from GitHub URL
    meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_match.json"
    # Read the JSON data as a JSON object
    response = requests.get(meta_data_github_url)
    raw_match_data = response.json()


    # The output has nested json elements. We process them
    raw_match_df = pd.json_normalize(raw_match_data, max_level=2)
    raw_match_df["home_team_side"] = raw_match_df["home_team_side"].astype(str)

    players_df = pd.json_normalize(
        raw_match_df.to_dict("records"),
        record_path="players",
        meta=[
            "home_team_score",
            "away_team_score",
            "date_time",
            "home_team_side",
            "home_team.name",
            "home_team.id",
            "away_team.name",
            "away_team.id",
        ],  # data we keep
    )


    # Take only players who played and create their total time
    players_df = players_df[
        ~((players_df.start_time.isna()) & (players_df.end_time.isna()))
    ]
    players_df["total_time"] = players_df["end_time"].apply(time_to_seconds) - players_df[
        "start_time"
    ].apply(time_to_seconds)
    players_df["total_time_minutes"] = players_df["end_time"].apply(time_to_minutes) - players_df[
        "start_time"
    ].apply(time_to_minutes)

    players_df['start_time_minute'] = players_df['start_time'].apply(time_to_minutes)
    players_df['end_time_minute'] = players_df['end_time'].apply(time_to_minutes)

    # Create a flag for GK
    players_df["is_gk"] = players_df["player_role.acronym"] == "GK"

    # Add a flag if the given player is home or away
    players_df["match_name"] = (
        players_df["home_team.name"] + " vs " + players_df["away_team.name"]
    )


    # Add a flag if the given player is home or away
    players_df["home_away_player"] = np.where(
        players_df.team_id == players_df["home_team.id"], "Home", "Away"
    )

    # Create flag from player
    players_df["team_name"] = np.where(
        players_df.team_id == players_df["home_team.id"],
        players_df["home_team.name"],
        players_df["away_team.name"],
    )

    # Figure out sides
    players_df[["home_team_side_1st_half", "home_team_side_2nd_half"]] = (
        players_df["home_team_side"]
        .astype(str)
        .str.strip("[]")
        .str.replace("'", "")
        .str.split(", ", expand=True)
    )
    # Clean up sides
    players_df["direction_player_1st_half"] = np.where(
        players_df.home_away_player == "Home",
        players_df.home_team_side_1st_half,
        players_df.home_team_side_2nd_half,
    )
    players_df["direction_player_2nd_half"] = np.where(
        players_df.home_away_player == "Home",
        players_df.home_team_side_2nd_half,
        players_df.home_team_side_1st_half,
    )


    # Renaming columns for consistent use
    players_df = players_df.rename(columns={
        "id": "player_id",
        # "playing_time.total.minutes_played": "playtime_minutes",
        "total_time_minutes": "playtime_minutes"
        })

    # Clean up and keep the columns that we want to keep about

    columns_to_keep = [
        "start_time",
        "end_time",
        "match_name",
        "date_time",
        "home_team.name",
        "away_team.name",
        "home_team.id",
        "away_team.id",
        "player_id",
        "short_name",
        "number",
        "team_id",
        "team_name",
        "player_role.position_group",
        "total_time",
        "playtime_minutes",
        "start_time_minute",
        "end_time_minute",
        "player_role.name",
        "player_role.acronym",
        "is_gk",
        "direction_player_1st_half",
        "direction_player_2nd_half",
    ]
    players_df = players_df[columns_to_keep]

    # Store in cache, if this is the first session-run
    _meta_cache[match_id] = players_df

    return players_df


#
def load_enriched_tracking_data(match_id):
    # Check cache first, in case it was already loaded in previous calls
    if match_id in _enriched_cache:
        return _enriched_cache[match_id]
    
    # loading relevant data
    tracking_df = load_tracking_data(match_id)
    players_df = load_meta_data(match_id)

    # Merging datasets
    enriched_tracking_data = tracking_df.merge(
        players_df, on=["player_id"]
    )

    # Labelling players' appearance type for identication
    enriched_tracking_data['appearance_type'] = np.where(
        enriched_tracking_data['playtime_minutes'] == enriched_tracking_data['playtime_minutes'].max(),
        'full',
        np.where(
            enriched_tracking_data['start_time'] == "00:00:00",
            'sub_out',
            'sub_in'
        )
    )

    # Store in cache, if this is the first session-run
    _enriched_cache[match_id] = enriched_tracking_data

    return enriched_tracking_data


#
def load_event_data(match_id, enriched_tracking_data):
    # Check cache first, in case it was already loaded in previous calls
    if match_id in _events_cache:
        return _events_cache[match_id]
    
    # Load event data from GitHub URL
    event_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"
    raw_event_df = pd.read_csv(event_data_github_url)

    # Extract home_team.id and away_team.id from enriched_tracking_data (they're constant for the match)
    home_team_id = enriched_tracking_data['home_team.id'].iloc[0]
    away_team_id = enriched_tracking_data['away_team.id'].iloc[0]

    # Add these IDs to raw_event_df
    raw_event_df['home_team.id'] = home_team_id
    raw_event_df['away_team.id'] = away_team_id

    # Extract home_team.name and away_team.name from enriched_tracking_data (they're constant for the match)
    home_team_name = enriched_tracking_data['home_team.name'].iloc[0]
    away_team_name = enriched_tracking_data['away_team.name'].iloc[0]

    # Add these team names to raw_event_df
    raw_event_df['home_team.name'] = home_team_name
    raw_event_df['away_team.name'] = away_team_name

    # print(f"Home Team ID: {home_team_id}")
    # print(f"Away Team ID: {away_team_id}")
    # print(f"raw_event_df shape: {raw_event_df.shape}")

    # Create home_team_score and away_team_score based on team_id matching
    # If the event team_id is home_team.id, then team_score is home_team_score
    raw_event_df['home_team_score'] = raw_event_df.apply(
        lambda row: row['team_score'] if row['team_id'] == row['home_team.id'] else row['opponent_team_score'],
        axis=1
    )

    # If the event team_id is away_team.id, then team_score is away_team_score
    raw_event_df['away_team_score'] = raw_event_df.apply(
        lambda row: row['opponent_team_score'] if row['team_id'] == row['home_team.id'] else row['team_score'],
        axis=1
    )

    # game_state value ('drawing', winning', losing') is to be interpreted from the home team's point of view

    # Quick verification
    # print("Sample of home_team_score and away_team_score:")
    # print(raw_event_df[['team_id', 'home_team.id', 'away_team.id', 'team_score', 'opponent_team_score', 'home_team_score', 'away_team_score']].head(10))

    # Define the columns to keep (store names in a list and use it)
    cols = [
        "event_id",
        "match_id",
        "period",
        "game_state",
        "home_team_score",
        "away_team_score",
        "defensive_structure",
        "home_team.id",
        "away_team.id",
        "home_team.name",
        "away_team.name",
        "frame_start",
        "frame_end",
        "frame_physical_start",
        "time_start",
        "time_end",
        "minute_start",
        "second_start",
        "duration",
    ]

    # Find the first occurrence of a 0-0 score pair; if not present start from the first row
    zero_zero_idx = raw_event_df.loc[(raw_event_df["home_team_score"] == 0) & (raw_event_df["away_team_score"] == 0)].index
    if len(zero_zero_idx) > 0:
        start_idx = zero_zero_idx[0]
    else:
        start_idx = raw_event_df.index[0]

    # Work from that start row onward and keep the first row and every first row where the score pair changes
    df_from_start = raw_event_df.loc[start_idx:].copy()
    score_pairs = df_from_start[["home_team_score", "away_team_score"]].apply(tuple, axis=1)
    mask = score_pairs.ne(score_pairs.shift()).fillna(True)

    # Build the goal_event_df using the stored column list and reset index for convenience
    goal_event_df = df_from_start[mask][cols].reset_index(drop=True)

    # Store in cache, if this is the first session-run
    _events_cache[match_id] = goal_event_df

    return goal_event_df