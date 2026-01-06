# Tennis Match Prediction with XGBoost

## Project Overview
This project implements a machine learning system to predict the outcomes of professional ATP tennis matches. By analyzing historical match data from 2015 to 2024, the system constructs a predictive model using **XGBoost** (Extreme Gradient Boosting).

The core of the solution involves advanced feature engineering, specifically **Dynamic Elo Ratings**, **Rolling Statistics** (Aces, Double Faults, Break Points), and **Head-to-Head** history to capture player form and matchup dynamics. The model is trained on nine years of data and validated on the complete 2024 season to simulate real-world forecasting performance.

## XGBoost: Theory & Implementation

### What is XGBoost?
**XGBoost** stands for **Extreme Gradient Boosting**. It is a powerful machine learning algorithm based on decision trees. Unlike a single decision tree which can be prone to overfitting (learning noise) or underfitting (missing patterns), XGBoost builds a "forest" of trees sequentially.

1.  **Gradient Boosting**: The model learns in stages. It builds a first tree to predict the target. Then, it calculates the *residuals* (errors) of that tree. The next tree is trained specifically to predict those errors, correcting the mistakes of the previous one. This process optimizes a differentiable loss function (Gradient Descent).
2.  **Regularization**: XGBoost includes L1 (Lasso) and L2 (Ridge) regularization terms in its objective function. This penalizes overly complex models, preventing the trees from growing too deep or too specific, which is crucial for tackling overfitting.
3.  **Tree Pruning**: It uses a "max_depth" parameter and prunes trees backwards, removing splits that don't provide a positive gain in predictive power.

### Implementation in this Project
The `XGBClassifier` was used from the `xgboost` Python library.
-   **Objective**: `binary:logistic` (Outputting the probability of a specific player winning).
-   **Evaluation**: Logarithmic Loss (`logloss`), ensuring the model is penalized confidently for wrong predictions.

## Data Processing Pipeline

The data pipeline transforms raw match records into a machine learning ready dataset.

### 1. Data Source
-   **Input**: CSV files `atp_matches_2015.csv` through `atp_matches_2024.csv`.
-   **Volume**: Approximately 27,000 matches.

### 2. Processing Steps
-   **Chronological Sorting**: Matches are ordered by date to ensure that future information is not used in past predictions.
-   **Class Balancing**: In the raw data, the "Winner" is always listed first. If fed directly, the model would simply learn that "Player 1 always wins." Randomly swapped the Player/Opponent perspective for 50% of the rows to create a balanced target variable (`label`: 0 or 1).

### 3. Training & Testing Split
A strict **Time-Series Split** was used to mimic real-world usage:
-   **Training Set**: Years 2015–2023 (Learning historical patterns).
-   **Testing Set**: Year 2024 (Evaluating on "future" unseen data).

## Feature Engineering & Overfitting Control

### Selected Features
The following features were engineered to capture player skill and form. The model selected the following features (ranked by importance):

1.  **`elo_diff` (51.1%)**: The difference in dynamic Elo rating between the two players. This is overwhelmingly the strongest predictor.
2.  **`p2_elo` (12.0%)** & **`p1_elo` (10.8%)**: Absolute Elo ratings.
3.  **`bp_diff` (5.8%)**: Difference in "Break Points Saved" consistency over the last 10 matches.
4.  **`ace_diff` (5.5%)**: Difference in serving dominance (Aces) over the last 10 matches.
5.  **`surface` (5.4%)**: The court surface (Hard, Clay, Grass), as players often specialize.
6.  **`h2h_diff` (5.0%)**: Historical Head-to-Head win differential.
7.  **`df_diff` (4.5%)**: Difference in unforced error proneness (Double Faults).

### Feature Details
-   **Dynamic Elo**: A custom zero-sum rating system (Starting 1500, K=32). Unlike official ATP rankings, this updates instantly after every match.
-   **Rolling Averages**: We compute the mean stats (Aces, etc.) for a player's previous 10 matches to represent current form.
-   **Head-to-Head (H2H)**: Calculates `(Player Wins - Opponent Wins)` for the specific pair *before* the match starts.

### Controlling Overfitting
To prevent the model from memorizing the training data, specific hyperparameters were tuned:
-   **`max_depth=4`**: Limits tree depth to prevent modeling specific noise.
-   **`subsample=0.8`**: Uses only 80% of rows for each tree (adding randomness).
-   **`colsample_bytree=0.8`**: Uses only 80% of features for each tree.
-   **`learning_rate=0.05`**: Slow learning rate implies more trees (`n_estimators=1000`) are needed, which generally generalizes better.
-   **`early_stopping_rounds=50`**: Stops training if validation accuracy doesn't improve for 50 rounds.

## Model Performance

On the **2024 Test Set** (3,076 matches), the model achieved:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **64.37%** |
| **Log Loss** | **0.6185** |

*Note: Professional tennis prediction benchmarks typically range from 65% to 70%. Achieving ~64.4% with a purely statistical model (without odds data or injury reports) is a strong baseline results.*

## Setup and Execution

### Prerequisites
-   Python 3.8+
-   `libomp` (Required for XGBoost on macOS)
    ```bash
    brew install libomp
    ```

### Installation
1.  **Clone/Open** the project folder.
2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Model
Execute the main script to load data, train the model, and view results:

```bash
python3 main.py
```

### Expected Output
```text
Welcome to Tennis XGBoost Match Predictor
Using data from: .../data/raw
Loading and processing data...
Train matches: 24598
Test matches: 3076
Training model...
...
Results for Test Year 2024:
Accuracy: 0.6437
Log Loss: 0.6185
...
Feature Importances:
elo_diff: 0.5110
...
```
