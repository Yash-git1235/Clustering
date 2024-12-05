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

# Read the dataset
df = pd.read_csv('/home/yash/Downloads/clustering and other-20241203T135732Z-001/clustering and other/updated_results2.csv')  # path to updated_results.csv
latitude = df['lat']
longitude = df['lon']

def cluster(latitude, longitude, threshold, threshold_step):
    # threshold = maximum threshold value = 2 * drop boundary (in feet)
    # threshold_step = decrease for more accuracy

    # Define clustering parameters
    thr_main = threshold  # Maximum threshold value
    thr = threshold_step  # Increment threshold value

    # Create an initial CSV file with latitude, longitude
    initial_data = pd.DataFrame({
        "Latitude": latitude,
        "Longitude": longitude,
    })

    # Save this initial data to a CSV file
    initial_csv_file = "initial_data.csv"
    initial_data.to_csv(initial_csv_file, index=False)
    print(f"Initial data saved to '{initial_csv_file}'")

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
        return None

    all_centroids = []  # Store centroids for all thresholds
    while thr <= thr_main:
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

    print("Final centroids:", centroids)

    # Save centroids
    with open("clustered_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Latitude", "Longitude"])
        writer.writerows(centroids)

    print("Centroids saved to 'clustered_results.csv'")
    return centroids

# Run the cluster function and get the centroids
final_centroids = cluster(latitude, longitude, 50, 5)

if final_centroids is not None:
    latitude, longitude = final_centroids[:, 0], final_centroids[:, 1]

count=1
for lat,lon in zip(latitude,longitude):
        if count<=4:
            val=[lat,lon]
            print(val)
            print(f'dropping on target {count}')
            count+=1
            #automation(val,count)
print('removing out_filter1 and out_filter2')
