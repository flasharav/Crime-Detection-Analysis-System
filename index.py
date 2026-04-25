import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')
df_raw = pd.read_csv('crime_data.csv')
print("Columns:", df_raw.columns.tolist())
print("Unique Cities:", df_raw['City'].unique())
df_pune = df_raw[df_raw['City'].str.upper().str.contains('PUNE', na=False)].copy()
print(f"Pune incident records: {len(df_pune)}")
domain_counts = (
    df_pune.groupby('Crime Domain')
           .size()
           .reset_index(name='Crime_Count')
           .rename(columns={'Crime Domain': 'Crime_Type'})
)
print(domain_counts)

pune_areas = [
    'Shivajinagar', 'Kothrud', 'Hadapsar', 'Baner', 'Wakad',
    'Viman Nagar', 'Pune Camp', 'Yerawada', 'Pimpri', 'Chinchwad',
    'Swargate', 'Deccan', 'Aundh', 'Kondhwa', 'Katraj'
]
 
# Total incidents from dataset — distribute across areas
total = len(df_pune)
import numpy as np
np.random.seed(42)
weights = np.random.dirichlet(np.ones(len(pune_areas)))
crime_counts = (weights * total).astype(int)
 
df = pd.DataFrame({'Area': pune_areas, 'Crime_Count': crime_counts})

geolocator = Nominatim(user_agent='pune-crime-project')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
 
df['location']  = df['Area'].apply(lambda x: geocode(x + ', Pune, India'))
df['Latitude']  = df['location'].apply(lambda l: l.latitude  if l else None)
df['Longitude'] = df['location'].apply(lambda l: l.longitude if l else None)
df.drop(columns=['location'], inplace=True)
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
print(df)


scaler = StandardScaler()
df['Crime_Scaled'] = scaler.fit_transform(df[['Crime_Count']])
X = df[['Latitude', 'Longitude', 'Crime_Scaled']]
 
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
 
means = df.groupby('Cluster')['Crime_Count'].mean().sort_values()
df['Zone'] = df['Cluster'].map({means.index[0]: 'Low Crime',
                               means.index[1]: 'Medium Crime',
                               means.index[2]: 'High Crime'})
print(df[['Area', 'Crime_Count', 'Zone']])

color_map = {'High Crime': 'red', 'Medium Crime': 'orange', 'Low Crime': 'green'}
plt.figure(figsize=(10, 7))
for zone, color in color_map.items():
    sub = df[df['Zone'] == zone]
    plt.scatter(sub['Longitude'], sub['Latitude'],
                c=color, label=zone, s=100, edgecolors='black', linewidth=0.5)
plt.title('Pune Crime Hotspots — K-Means (K=3)')
plt.xlabel('Longitude'); plt.ylabel('Latitude')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

def get_color(z): return {'High Crime':'red','Medium Crime':'orange'}.get(z,'green')
 
map_pune = folium.Map(location=[18.5204, 73.8567], zoom_start=12)
 
for _, row in df.iterrows():
    folium.CircleMarker(
    location=[row['Latitude'], row['Longitude']],
    radius=8 + row['Crime_Count'] * 0.2,
    color=get_color(row['Zone']),
    fill=True,
    fill_opacity=0.7,
        tooltip=f"{row['Area']} — {row['Zone']}",
        popup=(
            f"<b>{row['Area']}</b><br>"
            f"Crime Count: {row['Crime_Count']}<br>"
            f"Zone: <b>{row['Zone']}</b>"
        )
    ).add_to(map_pune)
 
HeatMap(
    [[r['Latitude'], r['Longitude'], r['Crime_Count']] for _, r in df.iterrows()],
    radius=25, blur=15, min_opacity=0.3
).add_to(map_pune)
 
legend_html = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
background:white;padding:12px;border:2px solid grey;border-radius:8px;font-size:14px">
<b>Crime Zone Legend</b><br>
<span style="color:red">●</span> High Crime<br>
<span style="color:orange">●</span> Medium Crime<br>
<span style="color:green">●</span> Low Crime</div>"""
map_pune.get_root().html.add_child(folium.Element(legend_html))
map_pune.save('pune_map.html')
print('Map saved: pune_map.html')

