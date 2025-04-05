import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define state classes
STATE_MAPPING = {
    0: "neutral",
    1: "focus",
    2: "fatigue",
    3: "stress",
    4: "emergency"
}

def preprocess_data(data_path):
    """
    Preprocess the raw health data for LSTM model
    """
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['steps'] = df['steps'].clip(lower=0)

    df['hr_change'] = df['heart_rate'].diff().fillna(0)
    df['hrv_change'] = df['hrv'].diff().fillna(0)
    df['hr_hrv_ratio'] = df['heart_rate'] / df['hrv'].replace(0, 0.1)

    def label_state(row):
        hr, hrv, steps, sleep = row['heart_rate'], row['hrv'], row['steps'], row['sleep_hours']
        hr_change = row['hr_change']
        baseline_hr, baseline_hrv = 70, 60
        focus_score = fatigue_score = stress_score = emergency_score = 0

        if hrv > baseline_hrv: focus_score += 1
        if 60 < hr < 85: focus_score += 1
        if steps > 1500: focus_score += 1

        if hrv < baseline_hrv * 0.8: fatigue_score += 1
        if sleep < 6: fatigue_score += 1
        if steps < 500: fatigue_score += 1

        if hrv < baseline_hrv * 0.6: stress_score += 1
        if hr > baseline_hr + 20 and steps < 100: stress_score += 1

        if hr > 120 or hr < 40: emergency_score += 1
        if hrv < baseline_hrv * 0.3: emergency_score += 1
        if abs(hr_change) > 20: emergency_score += 1

        if emergency_score >= 1: return 4
        elif stress_score >= 2: return 3
        elif fatigue_score >= 2: return 2
        elif focus_score >= 2: return 1
        else: return 0

    df['state'] = df.apply(label_state, axis=1)

    features = [
        'heart_rate', 'hrv', 'steps', 'sleep_hours',
        'hour', 'day_of_week', 'hr_change', 'hrv_change', 'hr_hrv_ratio'
    ]

    X = df[features].values
    y = df['state'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, features, df

def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=5):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size2, hidden_size2)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        # First LSTM layer
        out, _ = self.lstm1(x)
        # Apply batch normalization to the last time step output
        batch_size, seq_len, hidden_size = out.size()
        out = out.reshape(-1, hidden_size)
        out = self.bn1(out)
        out = out.reshape(batch_size, seq_len, hidden_size)
        
        # Second LSTM layer (use only the output of the last time step)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Get the last time step output
        out = self.dropout(out)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.fc2(out)
        
        return out

def build_lstm_model(input_shape, num_classes=5):
    input_size = input_shape[1]  # Number of features
    model = LSTMModel(input_size=input_size, output_size=num_classes)
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Build and prepare model
    model = build_lstm_model(input_shape)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # For early stopping
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    
    # For history tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model
            best_model = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Restore best model
    model.load_state_dict(best_model)
    
    return model, history

def evaluate_model(model, X_test, y_test, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    y_pred_classes = np.array(all_predictions)
    y_test_np = np.array(all_targets)
    
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred_classes, target_names=class_names))
    
    cm = confusion_matrix(y_test_np, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    return y_pred_classes, np.array(all_probabilities)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('training_history.png')

def save_model(model, scaler, features):
    torch.save(model.state_dict(), 'state_prediction_model.pt')
    np.save('scaler.npy', scaler.scale_)
    np.save('feature_names.npy', features)
    print("Model and preprocessing components saved.")

def predict_state(model, scaler, features, new_data, time_steps=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    input_data = pd.DataFrame([new_data], columns=features)
    input_scaled = scaler.transform(input_data)
    sequence = np.array([input_scaled] * time_steps)
    sequence = np.reshape(sequence, (1, time_steps, len(features)))
    
    with torch.no_grad():
        input_tensor = torch.FloatTensor(sequence).to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, STATE_MAPPING[predicted_class], probabilities[0].cpu().numpy()

def main():
    data_path = 'raw_data.csv'
    X_scaled, y, scaler, features, df = preprocess_data(data_path)
    
    time_steps = 5
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model, history = train_model(X_train, y_train, X_val, y_val, input_shape)
    
    class_names = [STATE_MAPPING[i] for i in range(5)]
    y_pred, _ = evaluate_model(model, X_test, y_test, class_names)
    
    plot_training_history(history)
    save_model(model, scaler, features)
    
    print("\nExample prediction:")
    example_data = {
        'heart_rate': 95,
        'hrv': 35,
        'steps': 200,
        'sleep_hours': 5.0,
        'hour': 14,
        'day_of_week': 2,
        'hr_change': 5,
        'hrv_change': -2,
        'hr_hrv_ratio': 2.7
    }
    
    class_idx, state, probs = predict_state(model, scaler, features, example_data)
    print(f"Predicted state: {state} (class {class_idx})")
    print(f"Class probabilities: {probs}")
    
    plt.figure(figsize=(10, 6))
    state_counts = df['state'].value_counts().sort_index()
    sns.barplot(x=state_counts.index.map(lambda x: STATE_MAPPING[x]), y=state_counts.values)
    plt.title('Distribution of States in Dataset')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.savefig('state_distribution.png')
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(x='state', y='heart_rate', data=df)
    plt.xlabel('State')
    plt.ylabel('Heart Rate')
    plt.xticks(ticks=range(5), labels=[STATE_MAPPING[i] for i in range(5)])
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x='state', y='hrv', data=df)
    plt.xlabel('State')
    plt.ylabel('HRV')
    plt.xticks(ticks=range(5), labels=[STATE_MAPPING[i] for i in range(5)])
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='state', y='steps', data=df)
    plt.xlabel('State')
    plt.ylabel('Steps')
    plt.xticks(ticks=range(5), labels=[STATE_MAPPING[i] for i in range(5)])
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x='state', y='sleep_hours', data=df)
    plt.xlabel('State')
    plt.ylabel('Sleep Hours')
    plt.xticks(ticks=range(5), labels=[STATE_MAPPING[i] for i in range(5)])
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png')
    
    print("\nModel development complete. Review the generated plots for model performance.")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()