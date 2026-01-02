import pandas as pd
import numpy as np
import glob
import os

def load_data(data_dir):
    """
    Loads and concatenates all match data files from 2015 to 2024.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "atp_matches_201[5-9].csv"))) + \
            sorted(glob.glob(os.path.join(data_dir, "atp_matches_202[0-4].csv")))
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        raise ValueError("No data files found.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Convert date to datetime
    full_df['tourney_date'] = pd.to_datetime(full_df['tourney_date'], format='%Y%m%d')
    full_df = full_df.sort_values('tourney_date').reset_index(drop=True)
    
    return full_df

def calculate_elo(df, k_factor=32):
    """
    Calculates dynamic Elo ratings for players based on match results.
    """
    # Initialize Elo ratings
    player_elo = {}
    
    # elo1 and elo2 columns
    elo1_list = []
    elo2_list = []
    
    def get_elo(player_id):
        return player_elo.get(player_id, 1500)
    
    for idx, row in df.iterrows():
        p1_id = row['winner_id'] if 'winner_id' in row else row['player_id'] # Check column names later, standard is usually winner_id in ATP data but file showed player_id/opponent_id for winner/loser context?
        # Examining the head output from Step 15:
        # player_id is the winner (Grigor Dimitrov won)
        # opponent_id is the loser
        
        # Let's verify standard naming. "player_id" is usually winner in these files if described as "winner_id" is not present. 
        # The head output shows: player_id, opponent_id. And score "7-6(5) 6-4" implies player (winner) first usually.
        # Wait, Step 15 output:
        # 2024-0339... player_name="Grigor Dimitrov", opponent_name="Holger Rune", score="7-6(5) 6-4"
        # Dimitrov won. So player_id = winner, opponent_id = loser.
        
        p1_id = row['player_id']
        p2_id = row['opponent_id']
        
        elo1 = get_elo(p1_id)
        elo2 = get_elo(p2_id)
        
        elo1_list.append(elo1)
        elo2_list.append(elo2)
        
        # Calculate expected score
        e1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        e2 = 1 / (1 + 10 ** ((elo1 - elo2) / 400))
        
        # Actual score (Player 1 is always the winner in raw rows)
        s1 = 1
        s2 = 0
        
        # Update ratings
        new_elo1 = elo1 + k_factor * (s1 - e1)
        new_elo2 = elo2 + k_factor * (s2 - e2)
        
        player_elo[p1_id] = new_elo1
        player_elo[p2_id] = new_elo2
        
    df['player_elo'] = elo1_list
    df['opponent_elo'] = elo2_list
    
    return df

def compute_rolling_stats(df, window=10):
    """
    Computes rolling averages for aces, double faults, and break points saved.
    """
    # We need to construct a player-centric view first to easily calculate rolling stats
    
    # Columns of interest for stats (from winner perspective in raw df)
    # w_ace, w_df, w_bpSaved
    # l_ace, l_df, l_bpSaved
    
    # We create a long-form dataframe where each row is a player in a match
    
    # Winner entries
    w_df_cols = ['tourney_date', 'match_num', 'player_id', 'w_ace', 'w_df', 'w_bpSaved']
    winners = df[w_df_cols].copy()
    winners.columns = ['tourney_date', 'match_num', 'id', 'ace', 'df', 'bpSaved']
    
    # Loser entries
    l_df_cols = ['tourney_date', 'match_num', 'opponent_id', 'l_ace', 'l_df', 'l_bpSaved']
    losers = df[l_df_cols].copy()
    losers.columns = ['tourney_date', 'match_num', 'id', 'ace', 'df', 'bpSaved']
    
    # Concatenate and sort
    player_stats = pd.concat([winners, losers]).sort_values(['id', 'tourney_date', 'match_num'])
    
    # Calculate rolling means
    player_stats['ace_mean'] = player_stats.groupby('id')['ace'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    player_stats['df_mean'] = player_stats.groupby('id')['df'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    player_stats['bpSaved_mean'] = player_stats.groupby('id')['bpSaved'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    
    # Fill NAs with 0 or global mean (let's use 0 for now as simpler start)
    player_stats = player_stats.fillna(0)
    
    # Map back to main df
    # Join for player (winner)
    df = df.merge(player_stats[['tourney_date', 'match_num', 'id', 'ace_mean', 'df_mean', 'bpSaved_mean']], 
                  left_on=['tourney_date', 'match_num', 'player_id'], 
                  right_on=['tourney_date', 'match_num', 'id'], 
                  how='left')
    df.rename(columns={'ace_mean': 'player_ace_mean', 'df_mean': 'player_df_mean', 'bpSaved_mean': 'player_bpSaved_mean'}, inplace=True)
    df.drop('id', axis=1, inplace=True)
    
    # Join for opponent (loser)
    df = df.merge(player_stats[['tourney_date', 'match_num', 'id', 'ace_mean', 'df_mean', 'bpSaved_mean']], 
                  left_on=['tourney_date', 'match_num', 'opponent_id'], 
                  right_on=['tourney_date', 'match_num', 'id'], 
                  how='left', suffixes=('', '_opp'))
    df.rename(columns={'ace_mean': 'opponent_ace_mean', 'df_mean': 'opponent_df_mean', 'bpSaved_mean': 'opponent_bpSaved_mean'}, inplace=True)
    df.drop('id', axis=1, inplace=True)
    
    return df

def calculated_h2h(df):
    """
    Calculates Head-to-Head win difference (Player Wins - Opponent Wins) prior to the match.
    """
    h2h_map = {} # Key: tuple(sorted(id1, id2)), Value: {id1: wins, id2: wins}
    
    h2h_diffs = []
    
    for idx, row in df.iterrows():
        p_id = row['player_id']
        o_id = row['opponent_id']
        
        pair_key = tuple(sorted((p_id, o_id)))
        
        if pair_key not in h2h_map:
            h2h_map[pair_key] = {p_id: 0, o_id: 0}
            
        stats = h2h_map[pair_key]
        
        # Current diff BEFORE this match
        diff = stats[p_id] - stats[o_id]
        h2h_diffs.append(diff)
        
        # Update for next time (Player always wins in this dataframe row)
        stats[p_id] += 1
        
    df['h2h_diff'] = h2h_diffs
    return df

def prepare_features(df):
    """
    Creates the final feature set including differences and balancing the classes.
    """
    # Balance classes: Swap 50% of matches to make the opponent the "features" perspective and label=0
    
    np.random.seed(42)
    swap_mask = np.random.rand(len(df)) < 0.5
    
    # Create new columns for the model
    # Features relative to "Player 1" (which might be the winner or loser after swap)
    
    # Initialize with default winner orientation
    model_df = pd.DataFrame()
    model_df['label'] = np.where(swap_mask, 0, 1) # If swapped, Winner is P2, so P1 lost (0). If not swapped, Winner is P1 (1).
    
    # If not swapped (Winner is P1):
    # P1 attributes = player_*, P2 attributes = opponent_*
    # If swapped (Winner is P2):
    # P1 attributes = opponent_*, P2 attributes = player_*
    
    # Helper to select based on mask
    def select(col_p, col_o):
        return np.where(swap_mask, df[col_o], df[col_p])
    
    model_df['p1_ace_mean'] = select('player_ace_mean', 'opponent_ace_mean')
    model_df['p2_ace_mean'] = select('opponent_ace_mean', 'player_ace_mean')
    
    model_df['p1_df_mean'] = select('player_df_mean', 'opponent_df_mean')
    model_df['p2_df_mean'] = select('opponent_df_mean', 'player_df_mean')
    
    model_df['p1_bpSaved_mean'] = select('player_bpSaved_mean', 'opponent_bpSaved_mean')
    model_df['p2_bpSaved_mean'] = select('opponent_bpSaved_mean', 'player_bpSaved_mean')
    
    model_df['p1_elo'] = select('player_elo', 'opponent_elo')
    model_df['p2_elo'] = select('opponent_elo', 'player_elo')
    
    # H2H is (Player - Opponent).
    # If not swapped: (P_wins - O_wins).
    # If swapped: We want (O_wins - P_wins) = -(P_wins - O_wins).
    raw_h2h = df['h2h_diff']
    model_df['h2h_diff'] = np.where(swap_mask, -raw_h2h, raw_h2h)

    # Derived Diff Features
    model_df['ace_diff'] = model_df['p1_ace_mean'] - model_df['p2_ace_mean']
    model_df['df_diff'] = model_df['p1_df_mean'] - model_df['p2_df_mean']
    model_df['bp_diff'] = model_df['p1_bpSaved_mean'] - model_df['p2_bpSaved_mean']
    model_df['elo_diff'] = model_df['p1_elo'] - model_df['p2_elo']
    
    # Extra static features?
    # Surface
    model_df['surface'] = df['surface']
    
    # IDs for reference if needed
    model_df['p1_id'] = select('player_id', 'opponent_id')
    model_df['p2_id'] = select('opponent_id', 'player_id')
    
    # Date for splitting
    model_df['tourney_date'] = df['tourney_date']
    
    return model_df

if __name__ == "__main__":
    # For quick testing
    import sys
    data_path = "/Users/salimhabbal/Desktop/Tennis XGBoost/data/raw"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
    print("Loading data...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} matches.")
    
    print("Calculating Elo...")
    df = calculate_elo(df)
    
    print("Calculating Rolling Stats...")
    df = compute_rolling_stats(df)
    
    print("Calculating H2H...")
    df = calculated_h2h(df)
    
    print("Preparing final features...")
    final_df = prepare_features(df)
    
    print("Data processing complete.")
    print(final_df.head())
    print(final_df.columns)
