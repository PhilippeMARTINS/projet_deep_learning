import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import folium
from streamlit_folium import folium_static
from branca.element import Template, MacroElement
import matplotlib.pyplot as plt

#rdn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Clé API récupérée sur https://home.openweathermap.org/api_keys
API_KEY = 'a2ff5d6eff00e54e0d205bd9393efe22'

# Lire les villes à partir d'un fichier
def load_cities(filename):
    with open(filename, 'r') as file:
        cities = [line.strip() for line in file]
    return cities

cities = load_cities('Villes.txt')

# Fonction pour obtenir des données météorologiques
def get_weather_data(city_name):
    base_url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        return {
            "city": city_name,
            "country": data['sys']['country'],
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "pressure": data['main']['pressure'],
            "weather": data['weather'][0]['description'],
            "wind_speed": data['wind']['speed'],
            "latitude": data['coord']['lat'],
            "longitude": data['coord']['lon']
        }
    return None

# Fonction pour sauvegarder les données dans un CSV
def save_data_to_csv(data, file_name='weather_data.csv'):
    now = datetime.now()
    formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame([{
        'Date': formatted_date,
        'City': data['city'],
        'Country': data['country'],
        'Temperature': data['temperature'],
        'Humidity': data['humidity'],
        'Pressure': data['pressure'],
        'Weather': data['weather'],
        'Wind Speed': data['wind_speed'],
        'Latitude': data['latitude'],
        'Longitude': data['longitude']
    }])
    if not os.path.isfile(file_name):
        df.to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, mode='a', header=False, index=False)

# Streamlit UI
st.title('Weather Data Collection App')

# Sidebar selection for data type
data_type = st.sidebar.selectbox(
    "Choose the data type to display:",
    ("Temperature", "Humidity", "Pressure", "Wind Speed")
)

# Fonction pour déterminer la couleur du marqueur en fonction de la température
def color_for_temperature(temp):
    if temp < 0:
        return 'blue'
    elif temp < 10:
        return 'lightblue'
    elif temp < 20:
        return 'green'
    elif temp < 30:
        return 'orange'
    else:
        return 'red' 

# Fonction pour déterminer la couleur du marqueur en fonction de l'humidité
def color_for_humidity(humidity):
    if humidity > 75:
        return 'navy'  # Humidité élevée
    elif humidity > 50:
        return 'dodgerblue'      # Humidité modérée
    else:
        return 'skyblue' # Faible humidité
    
# Fonction pour déterminer la couleur du marqueur en fonction de la pression
def color_for_pressure(pressure):
    if pressure > 1020:
        return 'darkgreen'  # Haute pression
    elif pressure > 1000:
        return 'limegreen'      # Pression normale
    else:
        return 'yellowgreen' # Basse pression

# Fonction pour déterminer la couleur du marqueur en fonction de la vitesse du vent
def color_for_wind_speed(wind_speed):
    if wind_speed >= 30:
        return 'darkred'    # Vent extrême
    elif wind_speed >= 20:
        return 'red'        # Vent très fort
    elif wind_speed >= 10:
        return 'orange'     # Vent fort
    elif wind_speed >= 5:
        return 'green'      # Vent modéré
    else:
        return 'lightgreen' # Vent léger
    
    
# Initialiser une carte
map = folium.Map(location=[20, 0], zoom_start=2)

if st.button("Collect Weather Data"):
    for city in cities:
        data = get_weather_data(city)
        if data:
            save_data_to_csv(data)
            st.write(f"Data collected and saved for {city}")
        else:
            st.write(f"Failed to retrieve data for {city}")
            
# Charger les données depuis le CSV pour la mise à jour de la carte
if os.path.exists('weather_data.csv'):
    weather_data = pd.read_csv('weather_data.csv')
    for index, row in weather_data.iterrows():
        if data_type == 'Temperature':
            detail = row['Temperature']
            color = color_for_temperature(detail)
        elif data_type == 'Humidity':
            detail = row['Humidity']
            color = color_for_humidity(detail)
        elif data_type == 'Pressure':
            detail = row['Pressure']
            color = color_for_pressure(detail)
        elif data_type == 'Wind Speed':
            detail = row['Wind Speed']
            color = color_for_wind_speed(detail)
        else:
            detail = row[data_type]
            color = 'gray'  # Default color for data types without specific color logic

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"{row['City']}: {detail}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(map)
        
# Afficher la carte
folium_static(map)

def plot_pie_chart(data, column, categories, title):
    category_counts = {category: 0 for category in categories.keys()}
    
    # Compter les occurrences dans chaque catégorie
    for index, row in data.iterrows():
        for category, (range, color) in categories.items():
            if range[0] <= row[column] < range[1]:
                category_counts[category] += 1
                break
    
    # Préparer les données pour le camembert
    labels = []
    sizes = []
    colors_list = []
    for category, count in category_counts.items():
        if count > 0:
            labels.append(category)
            sizes.append(count)
            colors_list.append(categories[category][1])
    
    # Créer le camembert
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    st.pyplot(plt)

temperature_categories = {
    'Very Cold': ([-float('inf'), 0], 'blue'),
    'Cold': ([0, 10], 'lightblue'),
    'Mild': ([10, 20], 'green'),
    'Warm': ([20, 30], 'orange'),
    'Hot': ([30, float('inf')], 'red')
}

humidity_categories = {
    'Low Humidity': ([0, 50], 'skyblue'),    # Humidité faible
    'Moderate Humidity': ([50, 75], 'dodgerblue'),  # Humidité modérée
    'High Humidity': ([75, 100], 'navy')  # Humidité élevée
}

pressure_categories = {
    'Low Pressure': ([0, 1000], 'yellowgreen'),  # Basse pression
    'Normal Pressure': ([1000, 1020], 'limegreen'),   # Pression normale
    'High Pressure': ([1020, float('inf')], 'darkgreen')   # Haute pression
}

wind_speed_categories = {
    'Light Wind': ([0, 5], 'lightgreen'),   # Vent léger
    'Moderate Wind': ([5, 10], 'green'),    # Vent modéré
    'Strong Wind': ([10, 20], 'orange'),   # Vent fort
    'Very Strong Wind': ([20, 30], 'red'),  # Vent très fort
    'Extreme Wind': ([30, float('inf')], 'darkred')  # Vent extrême
}

# Afficher le camembert en fonction du type de données
if data_type == 'Temperature':
    plot_pie_chart(weather_data, 'Temperature', temperature_categories, 'Temperature Distribution')
if data_type == 'Humidity':
    plot_pie_chart(weather_data, 'Humidity', humidity_categories, 'Humidity Distribution')
if data_type == 'Pressure':
    plot_pie_chart(weather_data, 'Pressure', pressure_categories, 'Pressure Distribution')
if data_type == 'Wind Speed':
    plot_pie_chart(weather_data, 'Wind Speed', wind_speed_categories, 'Wind Speed Distribution')

# Charger les données
@st.cache_data 
def load_data():
    data_path = 'weather_data.csv'
    return pd.read_csv(data_path)

# Charger les données
data = load_data()

# Calculer les valeurs min et max pour chaque champ pertinent
humidity_min, humidity_max = int(data['Humidity'].min()), int(data['Humidity'].max())
pressure_min, pressure_max = int(data['Pressure'].min()), int(data['Pressure'].max())
wind_speed_min, wind_speed_max = data['Wind Speed'].min(), data['Wind Speed'].max()
latitude_min, latitude_max = data['Latitude'].min(), data['Latitude'].max()
longitude_min, longitude_max = data['Longitude'].min(), data['Longitude'].max()

# Charger le modèle avec le format .keras
model = load_model('model.keras') 

st.title('Prédiction de la température')

# Collecte des données de l'utilisateur avec des indications sur le format
humidity = st.number_input(f'Entrez l\'humidité (%) (min: {humidity_min}, max: {humidity_max}):',
                           min_value=humidity_min, max_value=humidity_max, value=(humidity_min + humidity_max) // 2, step=1)
pressure = st.number_input(f'Entrez la pression (hPa) (min: {pressure_min}, max: {pressure_max}):',
                           min_value=pressure_min, max_value=pressure_max, value=(pressure_min + pressure_max) // 2, step=1)
wind_speed = st.number_input(f'Entrez la vitesse du vent (km/h) (min: {wind_speed_min:.1f}, max: {wind_speed_max:.1f}):',
                             min_value=float(wind_speed_min), max_value=float(wind_speed_max),
                             value=(wind_speed_min + wind_speed_max) / 2, step=0.1)
#latitude = st.number_input(f'Entrez la latitude (min: {latitude_min:.2f}, max: {latitude_max:.2f}):',
 #                          min_value=float(latitude_min), max_value=float(latitude_max),
  #                         value=(latitude_min + latitude_max) / 2, step=0.01)
#longitude = st.number_input(f'Entrez la longitude (min: {longitude_min:.2f}, max: {longitude_max:.2f}):',
 #                           min_value=float(longitude_min), max_value=float(longitude_max),
  #                          value=(longitude_min + longitude_max) / 2, step=0.01)

# Chargement des données depuis un fichier CSV
@st.cache_data
def load_data():
    data2 = pd.read_csv('weather_data.csv')
    return data2

data2 = load_data()

# Assurez-vous de prendre uniquement les 300 premières entrées si nécessaire
data2 = data2.head(300)

# Créer un menu déroulant avec les noms des villes
selected_city = st.selectbox('Choisissez une ville', data2['City'].unique())

# Récupérer les coordonnées pour la ville sélectionnée
city_data = data2[data2['City'] == selected_city].iloc[0]
latitude = city_data['Latitude']
longitude = city_data['Longitude']

# Afficher les coordonnées
st.write(f"Latitude de {selected_city} : {latitude}")
st.write(f"Longitude de {selected_city} : {longitude}")

# Fonction pour charger le modèle et le scaler
@st.cache_data
def load_model_and_scaler():
    model = load_model('model.keras')
    scaler = pd.read_pickle('scaler.pkl')  
    return model, scaler

model, scaler = load_model_and_scaler()

if st.button('Prédire la température'):
    input_features = np.array([[humidity, pressure, wind_speed, latitude, longitude]])
    input_features = scaler.transform(input_features)  # Appliquer le scaler
    predicted_temperature = model.predict(input_features)
    st.write(f'La température prédite est de {predicted_temperature[0][0]:.2f}°C')
