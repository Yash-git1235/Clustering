import numpy as np
from sklearn.cluster import DBSCAN
import folium
from shapely.geometry import Polygon, Point
import csv

thr_main = 50 #(ft) threshold value
thr = 5 #(ft) increasing threshold value

# Define the bounding box corners
bounding_box_coords = [
    (38.315386, -76.550875),
    (38.315683, -76.552586),
    (38.315895, -76.552519),
    (38.315607, -76.550800)
]

# Create a bounding box polygon using Shapely
bounding_box = Polygon(bounding_box_coords)

# Latitude and longitude data (include some points outside the bounding box to test filtering)
data = np.array([
    [38.3158894, -76.5518986],
    [38.3157856, -76.5522313],
    [38.3157902, -76.5519936],
    [38.315759, -76.552272],
    [38.3157564, -76.5522055],
    [38.3157147, -76.5518794],
    [38.3157567, -76.5518732],
    [38.3157422, -76.5520774]
])

# Filter points to keep only those inside the bounding box
filtered_data = np.array([point for point in data if bounding_box.contains(Point(point[0], point[1]))])
filtered_data_og = filtered_data

# Check if filtered_data is empty
if filtered_data.size == 0:
    print("No points are inside the bounding box.")
else:
    while(thr<=thr_main):
        # Convert 50 feet to degrees (approximation)
        feet_to_degrees_lat = thr / 364000
        feet_to_degrees_lon = thr / 288200
        eps = np.mean([feet_to_degrees_lat, feet_to_degrees_lon])

        # Apply DBSCAN clustering on filtered data
        dbscan = DBSCAN(eps=eps, min_samples=1)
        dbscan.fit(filtered_data)

        # Get clustering results
        labels = dbscan.labels_

        # Find centroids for each cluster and handle standalone points
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            cluster_points = filtered_data[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        filtered_data = np.array(centroids)
        thr+=5

    # Initialize Folium map centered around the bounding box area
    map_center = [filtered_data[:, 0].mean(), filtered_data[:, 1].mean()]
    m = folium.Map(location=map_center, zoom_start=15)

    # Plot bounding box on the map
    folium.PolyLine(bounding_box_coords + [bounding_box_coords[0]], color='blue', weight=2.5, popup="Bounding Box").add_to(m)

    # Plot filtered data points and circles on the map
    for idx, point in enumerate(filtered_data_og):
        folium.CircleMarker(
            location=[point[0], point[1]],
            radius=5,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.6,
            popup=f"Filtered Point {idx+1}"
        ).add_to(m)

        # Add 50ft radius circle around each point
        folium.Circle(
            location=[point[0], point[1]],
            radius=(thr_main / 2) * 0.3048,  # 25 feet in meters
            color='purple',
            fill=True,
            fill_opacity=0.2
        ).add_to(m)

    # Plot centroids of each cluster on the map
    for idx, centroid in enumerate(centroids):
        folium.Marker(
            location=[centroid[0], centroid[1]],
            popup=f"Centroid {idx+1}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    # Save and display the map
    m.save("map_clusters_with_bounding_box.html")
    print("Map saved as 'map_clusters_with_bounding_box.html'")

    # Save centroids to CSV
    with open("clustered_centroids.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Latitude", "Longitude"])
        writer.writerows(centroids)

    print("Centroids saved to 'clustered_centroids.csv'")
