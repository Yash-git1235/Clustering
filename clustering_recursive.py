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
    # 10 points outside the bounding box
    (38.315200, -76.550500),
    (38.315300, -76.553000),
    (38.316000, -76.552800),
    (38.316500, -76.551200),
    (38.315900, -76.550400),
    (38.315000, -76.551000),
    (38.316100, -76.552000),
    (38.315100, -76.552600),
    (38.316200, -76.553000),
    (38.315800, -76.550100),
    # 20 points inside the bounding box
    (38.315500, -76.551500),
    (38.315620, -76.551700),
    (38.315430, -76.551900),
    (38.315780, -76.552300),
    (38.315500, -76.552000),
    (38.315600, -76.551850),
    (38.315700, -76.552100),
    (38.315620, -76.552450),
    (38.315470, -76.551600),
    (38.315550, -76.551800),
    (38.315560, -76.552200),
    (38.315630, -76.552000),
    (38.315750, -76.552250),
    (38.315800, -76.551750),
    (38.315650, -76.551900),
    (38.315540, -76.551990),
    (38.315710, -76.552150),
    (38.315670, -76.552300),
    (38.315450, -76.551800),
    (38.315590, -76.552100)
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
        print(thr)
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
