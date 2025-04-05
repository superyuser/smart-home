import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os

# Constants
SEQUENCE_LENGTH = 10  # Number of time steps to look back
NUM_CLASSES = 5      # Number of classification states
RANDOM_SEED = 42     # For reproducibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directory for saved models if it doesn't exist
os.makedirs('smart_wearable/saved_models', exist_ok=True)

class HealthDataset(Dataset):
    """Custom Dataset for loading health data sequences"""
    def __init__(self, sequences, labels=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = None if labels is None else torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.sequences[idx]
        return self.sequences[idx], self.labels[idx]

class LSTMClassifier(nn.Module):
    """LSTM model for health state classification"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def load_and_preprocess_data():
    """
    Load and preprocess the raw health data.
    Returns preprocessed features and generated labels.
    """
    # Load the raw data
    df = pd.read_csv('smart_wearable/raw_data.csv')
    
    # Convert timestamp to hour of day (0-23) to capture daily patterns
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    # Normalize features using MinMaxScaler (range: 0-1)
    scaler = MinMaxScaler()
    features = ['heart_rate', 'hrv', 'steps', 'sleep_hours', 'hour']
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    
    # Generate labels based on the health metrics
    def generate_label(row):
        if row['heart_rate'] > 0.8 and row['hrv'] < 0.3:  # High HR, Low HRV
            return 4  # Emergency
        elif row['heart_rate'] > 0.7 and row['hrv'] < 0.4:  # Elevated HR, Low HRV
            return 3  # Stress
        elif row['sleep_hours'] < 0.3 or (row['heart_rate'] > 0.6 and row['steps'] < 0.3):
            return 2  # Fatigue
        elif 0.4 <= row['heart_rate'] <= 0.6 and row['hrv'] >= 0.5:  # Normal HR, Good HRV
            return 1  # Focus
        else:
            return 0  # Neutral

    labels = df_scaled.apply(generate_label, axis=1)
    return df_scaled.values, labels.values

def create_sequences(data, labels):
    """
    Create sequences for LSTM input from the time series data.
    """
    X, y = [], []
    for i in range(len(data) - SEQUENCE_LENGTH):
        X.append(data[i:i + SEQUENCE_LENGTH])
        y.append(labels[i + SEQUENCE_LENGTH])
    return np.array(X), np.array(y)

def train_model():
    """
    Main function to train the LSTM model.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    features, labels = load_and_preprocess_data()
    
    # Save the scaler for future use
    scaler = MinMaxScaler()
    features_list = ['heart_rate', 'hrv', 'steps', 'sleep_hours', 'hour']
    scaler.fit(pd.DataFrame(features, columns=features_list))
    joblib.dump(scaler, 'smart_wearable/saved_models/scaler.save')
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(features, labels)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Create datasets and dataloaders
    train_dataset = HealthDataset(X_train, y_train)
    test_dataset = HealthDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model
    input_size = features.shape[1]  # Number of features
    model = LSTMClassifier(input_size=input_size).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    print("Training the model...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluate the model
    print("\nEvaluating the model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'smart_wearable/saved_models/model.pth')
    print("Model saved to smart_wearable/saved_models/model.pth")
    
    return model

def predict_state(new_data, model=None):
    """
    Make predictions on new health data.
    
    Args:
        new_data (pd.DataFrame): DataFrame with columns ['timestamp', 'heart_rate', 'hrv', 'steps', 'sleep_hours']
        model (LSTMClassifier, optional): Pre-trained model. If None, loads from saved file.
    
    Returns:
        list: Predicted states for each sequence in the data
        list: Probability scores for each prediction
    """
    # Load the saved model if not provided
    if model is None:
        model_path = 'smart_wearable/saved_models/model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError("No saved model found. Please train the model first.")
        
        input_size = 5  # Number of features
        model = LSTMClassifier(input_size=input_size).to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    
    # Load the saved scaler
    scaler = joblib.load('smart_wearable/saved_models/scaler.save')
    
    # Preprocess the new data
    new_data['hour'] = pd.to_datetime(new_data['timestamp']).dt.hour
    features = ['heart_rate', 'hrv', 'steps', 'sleep_hours', 'hour']
    scaled_data = scaler.transform(new_data[features])
    
    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        sequences.append(scaled_data[i:i + SEQUENCE_LENGTH])
    
    if not sequences:
        raise ValueError("Not enough data points to create a sequence. Need at least 10 consecutive readings.")
    
    # Create dataset and dataloader
    predict_dataset = HealthDataset(np.array(sequences))
    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)
    
    # Make predictions
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch_X in predict_loader:
            if isinstance(batch_X, torch.Tensor):
                batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Convert numeric predictions to state names
    state_names = ['Neutral', 'Focus', 'Fatigue', 'Stress', 'Emergency']
    predicted_states = [state_names[pred] for pred in predictions]
    
    return predicted_states, probabilities

def example_prediction():
    """
    Example of how to use the prediction function with new data.
    """
    # Example new data (should be replaced with real data)
    example_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=15, freq='10min'),
        'heart_rate': np.random.uniform(60, 100, 15),
        'hrv': np.random.uniform(50, 80, 15),
        'steps': np.random.uniform(0, 1000, 15),
        'sleep_hours': np.random.uniform(5, 8, 15)
    })
    
    try:
        states, probs = predict_state(example_data)
        print("\nPredictions for new data:")
        print("-------------------------")
        for i, (state, prob) in enumerate(zip(states, probs)):
            print(f"Sequence {i+1}:")
            print(f"Predicted State: {state}")
            print(f"Confidence Scores:")
            for state_name, confidence in zip(['Neutral', 'Focus', 'Fatigue', 'Stress', 'Emergency'], prob):
                print(f"  {state_name}: {confidence:.2%}")
            print()
    except FileNotFoundError:
        print("Please train the model first using train_model()")
    except Exception as e:
        print(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    # Show example prediction
    print("\nDemonstrating predictions on example data...")
    example_prediction()
