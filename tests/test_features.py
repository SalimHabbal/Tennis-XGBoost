import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import calculate_elo, compute_rolling_stats, calculated_h2h

class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        # Create a tiny dummy dataset
        data = {
            'tourney_date': pd.to_datetime(['20230101', '20230102', '20230103', '20230104']),
            'match_num': [1, 2, 3, 4],
            'player_id': [1, 1, 2, 1], # Winner
            'opponent_id': [2, 3, 3, 2], # Loser
            'w_ace': [10, 5, 2, 8],
            'w_df': [1, 0, 0, 1],
            'w_bpSaved': [5, 2, 1, 3],
            'l_ace': [5, 2, 1, 4],
            'l_df': [2, 3, 1, 0],
            'l_bpSaved': [1, 0, 0, 1],
            'surface': ['Hard', 'Hard', 'Hard', 'Hard'],
            'winner_id': [1, 1, 2, 1] # Redundant but sometimes used
        }
        self.df = pd.DataFrame(data)
        
    def test_elo_calculation(self):
        df = calculate_elo(self.df.copy())
        
        # Initial Elo is 1500
        # Match 1: P1(1500) vs P2(1500). P1 wins.
        # E1 = 0.5. Delta = 32 * (1 - 0.5) = 16.
        # New P1 = 1516, New P2 = 1484.
        
        # In our dataframe, the 'player_elo' col stores rating BEFORE the match? 
        # Or current? The implementation in `data_processing.py` stores the rating used FOR calculation
        # (which is current rating before update).
        
        # Row 0: P1 Elo should be 1500.
        self.assertEqual(df.loc[0, 'player_elo'], 1500)
        self.assertEqual(df.loc[0, 'opponent_elo'], 1500)
        
        # Row 1: P1(1516) vs P3(1500). P1 wins.
        # Check if row 1 player_elo is 1516.
        # Match 1 happened at index 0. Match 2 at index 1.
        self.assertEqual(df.loc[1, 'player_elo'], 1516)
        
    def test_rolling_stats(self):
        # Match 1: P1 wins (10 aces). P2 loses (5 aces).
        # Match 2: P1 wins (5 aces). P3 loses (2 aces).
        # Match 3: P2 wins (2 aces). P3 loses (1 ace).
        # Match 4: P1 wins (8 aces). P2 loses (4 aces).
        
        # By Match 4 (index 3):
        # P1 has matches 1 (10) and 2 (5). Mean = 7.5.
        # P2 has matches 1 (5) and 3 (2 - winner). Mean = 3.5.
        
        df = compute_rolling_stats(self.df.copy(), window=10)
        
        # Row 3 (Match 4): P1 vs P2
        p1_mean = df.loc[3, 'player_ace_mean']
        p2_mean = df.loc[3, 'opponent_ace_mean']
        
        self.assertEqual(p1_mean, 7.5)
        self.assertEqual(p2_mean, 3.5)
        
    def test_h2h(self):
        # Match 1: P1 vs P2. P1 wins. Diff before: 0.
        # Match 4: P1 vs P2. P1 wins. Diff before: P1 has 1 win, P2 has 0. Diff = 1.
        
        df = calculated_h2h(self.df.copy())
        
        self.assertEqual(df.loc[0, 'h2h_diff'], 0)
        self.assertEqual(df.loc[3, 'h2h_diff'], 1)

if __name__ == '__main__':
    unittest.main()
