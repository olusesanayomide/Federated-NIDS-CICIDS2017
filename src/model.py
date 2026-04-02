import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data(filepath='data/cicids2017_cleaned.csv'):
    """
    Loads the CICIDS2017 dataset, performs label encoding, 
    and scales features using MinMaxScaler.
    """
    print(f"🔍 Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Label Encoding
    le = LabelEncoder()
    df['Attack Type'] = le.fit_transform(df['Attack Type'])
    
    # Optional: Print mapping for logs
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("✅ Mapping Key:", mapping)

    # 2. Feature/Target Separation
    X = df.drop('Attack Type', axis=1)
    y = df['Attack Type'].values # Convert to numpy array
    
    # 3. Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Preprocessing complete. Shape: {X_scaled.shape}")
    
    return X_scaled, y

def partition_data(X, y, num_clients=3):
    """
    Shuffles and splits data into chunks for federated clients.
    """
    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=42)
    
    X_clients = np.array_split(X, num_clients)
    y_clients = np.array_split(y, num_clients)
    
    return X_clients, y_clients
