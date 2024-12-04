import os
import re
import csv
import cv2
import sys
import time
import math
import pickle
import shutil
import signal
import argparse
import threading
import subprocess
import pandas as pd 
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from shapely.geometry import Polygon, Point
import csv
df=pd.read_csv('updated_results1.csv')
latitude=df['lat']
longitude=df['lon']
def cluster(latitude, longitude, threshold, threshold_step):
    # Create an initial CSV file with latitude, longitude
    initial_data = pd.DataFrame({
        "Latitude": latitude,
        "Longitude": longitude,
    })

    # Save this initial data to a CSV file
    initial_csv_file = "initial_data.csv"
    initial_data.to_csv(initial_csv_file, index=False)
    print(f"Initial data saved to '{initial_csv_file}'")

    # Define clustering parameters
    thr_main = threshold  # Maximum threshold value
    thr = threshold_step  # Increment threshold value

    # Define the bounding box corners
    bounding_box_coords = [
        (13.030968, 77.565227),
        (13.031323, 77.565268),
        (13.031202, 77.565494),
        (13.031195, 77.565582),
        (13.031080, 77.565680),
        (13.030892, 77.565660)
    ]
    bounding_box = Polygon(bounding_box_coords)

    # Combine latitude and longitude into a single array
    data = np.column_stack((latitude, longitude))

    # Filter points inside the bounding box
    filtered_data = np.array([point for point in data if bounding_box.contains(Point(point[0], point[1]))])

    if filtered_data.size == 0:
        print("No points are inside the bounding box.")
        return initial_csv_file

    all_centroids = []  # Store centroids for all thresholds
    got4=False
    while thr <= thr_main:
        if(got4==False):
                feet_to_degrees_lat = thr / 364000
                feet_to_degrees_lon = thr / 288200
                eps = np.mean([feet_to_degrees_lat, feet_to_degrees_lon])

                dbscan = DBSCAN(eps=eps, min_samples=1)
                dbscan.fit(filtered_data)

                labels = dbscan.labels_
                unique_labels = np.unique(labels)

                centroids = [filtered_data[labels == label].mean(axis=0) for label in unique_labels]
                centroids = np.array(centroids)
                all_centroids.append(centroids)

                filtered_data = centroids  # Update data for the next iteration
                print(f"Threshold: {thr}, Clusters: {len(centroids)}")
                thr += threshold_step
                if(len(centroids)<5):
                    break
    print(centroids)
    # Visualize on a Folium map
    map_center = [filtered_data[:, 0].mean(), filtered_data[:, 1].mean()]
    m = folium.Map(location=map_center, zoom_start=15)

    # Plot the bounding box
    folium.PolyLine(bounding_box_coords + [bounding_box_coords[0]], color='blue', weight=2.5, popup="Bounding Box").add_to(m)

    # Plot initial data points on the map
    for idx, point in enumerate(data):
        folium.CircleMarker(
            location=[point[0], point[1]],
            radius=5,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.6,
            popup=f"Initial Point {idx+1}"
        ).add_to(m)

    # Plot filtered data points (centroids) on the map
    for point in filtered_data:
        folium.Marker(location=[point[0], point[1]], icon=folium.Icon(color="red")).add_to(m)

    # Save and display the map
    m.save("map_clusters_with_bounding_box.html")
    print("Map saved as 'map_clusters_with_bounding_box.html'")


    # Save centroids
    with open("clustered_results.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Latitude", "Longitude"])
            i = 0
            while i<len(centroids):
                writer.writerow(centroids[i])
                i+=1
    print("Centroids saved to 'clustered_results.csv' ")

cluster(latitude,longitude,50,5)
