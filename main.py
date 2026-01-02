import sys
import os
from src.model_training import train_and_eval

def main():
    print("Welcome to Tennis XGBoost Match Predictor")
    
    # Define data directory
    # Assume we are running from project root
    data_dir = os.path.join(os.getcwd(), "data", "raw")
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
        
    print(f"Using data from: {data_dir}")
    
    # Train and Evaluate
    try:
        model, accuracy = train_and_eval(data_dir)
        print(f"\nTraining completed successfully.")
        print(f"Final Test Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
