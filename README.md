# 🚦 Smart Traffic Management System: Optimizing Traffic Flow Using Machine Learning

## 📌 Project Description

This project aims to optimize urban traffic flow by dynamically adjusting traffic signal timers based on real-time vehicle detection using **YOLO (You Only Look Once)** object detection. The system processes traffic camera images, detects and counts vehicles in each lane, and adjusts signal durations accordingly, aiming to reduce congestion and idle time at intersections.

---

## 📌 Installation
Step I: Clone the Repository
      $ git clone https://github.com/Rogue-Ghost/Major-Project 
      
Step II: Download the weights file from here and place it in the Adaptive-Traffic-Signal-Timer/Code/YOLO/darkflow/bin directory

Step III: Install the required packages

      # On the terminal, move into Adaptive-Traffic-Signal-Timer/Code/YOLO/darkflow directory
      $ cd Adaptive-Traffic-Signal-Timer/Code/YOLO/darkflow
      $ pip install -r requirements.txt
      $ python setup.py build_ext --inplace
      
Step IV: Run the code
      # To run vehicle detection
      $ python vehicle_detection.py
      
      # To run simulation
      $ python simulation.py

## 📌 Video Reference
Drive link - https://drive.google.com/drive/folders/1r5Ez458UjN8NzmgGDn3BKpz7w-kyxDHp?usp=sharing
