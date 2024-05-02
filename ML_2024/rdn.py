import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Chargement des données
data = pd.read_csv('weather_data.csv')

# Sélection des features et de la target
features = data[['Humidity', 'Pressure', 'Wind Speed', 'Latitude', 'Longitude']]
target = data['Temperature']

# Début du traitement
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Construction et entraînement du modèle
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Sauvegarde du modèle et du scaler
model.save('model.keras')
pd.to_pickle(scaler, 'scaler.pkl')

