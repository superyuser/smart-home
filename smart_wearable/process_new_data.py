import pandas as pd
import numpy as np
from lstm import predict_state
import os
from datetime import datetime
import traceback

def process_and_classify_data(input_file_path, output_directory='smart_wearable/processed_data'):
    """
    Process new health data and add state classifications.
    
    Args:
        input_file_path (str): Path to the new data CSV file
        output_directory (str): Directory where processed data will be saved
    
    Returns:
        str: Path to the saved processed data file
    """
    try:
        # Create output directory if it doesn't exist
        print(f"Creating output directory: {output_directory}")
        os.makedirs(output_directory, exist_ok=True)
        
        # Check if input file exists
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        
        # Load the new data
        print(f"Loading data from {input_file_path}...")
        data = pd.read_csv(input_file_path)
        print(f"Loaded {len(data)} rows of data")
        
        # Verify required columns
        required_columns = ['timestamp', 'heart_rate', 'hrv', 'steps', 'sleep_hours']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("\nData preview:")
        print(data.head())
        print("\nColumn names:", data.columns.tolist())
        
        # Make predictions
        print("\nMaking predictions...")
        states, probabilities = predict_state(data)
        print(f"Generated predictions for {len(states)} sequences")
        
        # Create a DataFrame with predictions
        print("\nAdding predictions to the dataset...")
        # First SEQUENCE_LENGTH-1 rows will be Unknown (need SEQUENCE_LENGTH timesteps for first prediction)
        state_column = ['Unknown'] * (len(data) - len(states)) + states
        
        # Create probability columns with the same length as the original data
        probability_columns = {}
        for i, state in enumerate(['Neutral', 'Focus', 'Fatigue', 'Stress', 'Emergency']):
            probs = [np.nan] * (len(data) - len(states))  # Fill start with NaN
            probs.extend([prob[i] for prob in probabilities])  # Add actual probabilities
            probability_columns[f'{state.lower()}_probability'] = probs
        
        # Add predictions to the original data
        data_with_predictions = data.copy()
        data_with_predictions['state'] = state_column
        for col_name, probs in probability_columns.items():
            data_with_predictions[col_name] = probs
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'processed_data_{timestamp}.csv'
        output_path = os.path.join(output_directory, output_filename)
        
        # Save to CSV
        print(f"\nSaving processed data to: {output_path}")
        data_with_predictions.to_csv(output_path, index=False)
        
        # Print summary
        print("\nClassification Summary:")
        print("----------------------")
        state_counts = data_with_predictions['state'].value_counts()
        for state, count in state_counts.items():
            print(f"{state}: {count} instances ({count/len(data_with_predictions):.1%})")
        
        return output_path
        
    except Exception as e:
        print(f"\nError processing data:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTraceback:")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Process new health data and add state classifications.')
    parser.add_argument('input_path', help='Path to the input CSV file')
    parser.add_argument('--output-dir', default='smart_wearable/processed_data',
                      help='Directory where processed data will be saved')
    
    args = parser.parse_args()
    
    print("\nStarting data processing...")
    print(f"Input file: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    
    result_path = process_and_classify_data(args.input_path, args.output_dir)
    
    if result_path:
        print("\nProcessing completed successfully!")
        print(f"Results saved to: {result_path}")
    else:
        print("\nProcessing failed. Please check the errors above.") 