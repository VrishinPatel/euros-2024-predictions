import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

# Load the models
rf_model = joblib.load('rf_model.pkl')
nn_model = tf.keras.models.load_model('nn_model.h5')

# Load and preprocess the future match data
future_matches = pd.read_csv('future_matches.csv')  # Read the future matches data
X_new = future_matches[['goals_home', 'goals_away']]  # Add more features if available

# Scale the features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Make predictions
rf_predictions = rf_model.predict(X_new_scaled)
nn_predictions = nn_model.predict(X_new_scaled)
nn_predictions = (nn_predictions > 0.5).astype(int)
combined_predictions = (rf_predictions + nn_predictions.flatten()) / 2
combined_predictions = (combined_predictions > 0.5).astype(int)

# Add predictions to the future matches DataFrame
future_matches['predictions'] = combined_predictions
print(future_matches[['home_team', 'away_team', 'predictions']])
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

# Load the models
rf_model = joblib.load('rf_model.pkl')
nn_model = tf.keras.models.load_model('nn_model.h5')

# Load and preprocess the future match data
future_matches = pd.read_csv('future_matches.csv')  # Read the future matches data
X_new = future_matches[['goals_home', 'goals_away']]  # Add more features if available

# Scale the features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Make predictions
rf_predictions = rf_model.predict(X_new_scaled)
nn_predictions = nn_model.predict(X_new_scaled)
nn_predictions = (nn_predictions > 0.5).astype(int)
combined_predictions = (rf_predictions + nn_predictions.flatten()) / 2
combined_predictions = (combined_predictions > 0.5).astype(int)

# Add predictions to the future matches DataFrame
future_matches['predictions'] = combined_predictions
print(future_matches[['home_team', 'away_team', 'predictions']])
