import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 1. Scheduling Algorithms
def FCFS(jobs):
    jobs.sort(key=lambda x: x[0])  # Sort jobs by arrival time (first come first serve)
    start_time = 0
    schedule = []
    for job in jobs:
        start_time = max(start_time, job[0])
        finish_time = start_time + job[1]
        schedule.append((job[0], start_time, finish_time, job[2]))  # (Arrival, Start, Finish, Job ID)
        start_time = finish_time
    return schedule

def SJF(jobs):
    jobs.sort(key=lambda x: x[1])  # Sort by burst time (Shortest Job First)
    start_time = 0
    schedule = []
    for job in jobs:
        start_time = max(start_time, job[0])
        finish_time = start_time + job[1]
        schedule.append((job[0], start_time, finish_time, job[2]))
        start_time = finish_time
    return schedule

def SRTF(jobs):
    jobs = sorted(jobs, key=lambda x: x[0])  # Sort by arrival time
    queue = []
    time = 0
    schedule = []
    job_idx = 0

    while job_idx < len(jobs) or queue:
        # Add all jobs that arrive by current time
        while job_idx < len(jobs) and jobs[job_idx][0] <= time:
            queue.append(jobs[job_idx])
            job_idx += 1

        if queue:
            # Select job with shortest remaining time
            queue.sort(key=lambda x: x[1])  # Sort by burst time
            current_job = queue.pop(0)
            start_time = time
            finish_time = start_time + current_job[1]
            schedule.append((current_job[0], start_time, finish_time, current_job[2]))  # (Arrival, Start, Finish, Job ID)
            time = finish_time
        else:
            time += 1  # No jobs are available, increment time

    return schedule

def RR(jobs, quantum):
    jobs = sorted(jobs, key=lambda x: x[0])  # Sort by arrival time
    queue = []
    time = 0
    schedule = []
    remaining_burst_time = {job[2]: job[1] for job in jobs}  # Job ID -> Remaining time
    job_idx = 0

    while job_idx < len(jobs) or queue:
        # Add jobs that arrive at the current time
        while job_idx < len(jobs) and jobs[job_idx][0] <= time:
            queue.append(jobs[job_idx])
            job_idx += 1

        if queue:
            current_job = queue.pop(0)
            job_id = current_job[2]
            start_time = time
            burst_time = min(quantum, remaining_burst_time[job_id])
            finish_time = start_time + burst_time
            schedule.append((current_job[0], start_time, finish_time, job_id))  # (Arrival, Start, Finish, Job ID)
            time = finish_time
            remaining_burst_time[job_id] -= burst_time

            if remaining_burst_time[job_id] > 0:
                queue.append(current_job)  # Re-queue job if it’s not completed
        else:
            time += 1  # No jobs to schedule, increment time

    return schedule

# 2. Decision Tree to Choose Algorithm (Example Decision Tree)
def decision_tree_schedule(jobs):
    features = []
    labels = []
    
    for job in jobs:
        features.append([job[0], job[1]])  # [Arrival Time, Burst Time]
        labels.append(job[2])  # Use Job ID as the label
    
    clf = DecisionTreeClassifier()
    clf.fit(features, labels)
    return clf.predict([[jobs[0][0], jobs[0][1]]])[0]  # Predict which algorithm (for simplicity)

# 3. Streamlit UI
st.title("Scheduling Algorithms Visualization")
st.sidebar.header("Scheduling Options")

# Input for jobs: (Arrival Time, Burst Time, Job ID)
num_jobs = st.sidebar.number_input("Number of Jobs", min_value=1, max_value=10, value=5)
jobs = []

for i in range(num_jobs):
    arrival_time = st.sidebar.number_input(f"Arrival Time for Job {i+1}", min_value=0, value=i*2)
    burst_time = st.sidebar.number_input(f"Burst Time for Job {i+1}", min_value=1, value=5)
    job_id = i + 1  # Job ID is just 1, 2, 3, ...
    jobs.append((arrival_time, burst_time, job_id))

# Algorithm Selection
algorithm = st.selectbox("Choose Scheduling Algorithm", ["FCFS", "SJF", "SRTF", "RR"])
quantum = st.number_input("Quantum for RR (if selected)", min_value=1, value=4)

if st.button("Run Scheduling"):
    # Decision Tree to dynamically choose an algorithm (as an extension)
    decision = decision_tree_schedule(jobs)
    st.write(f"Decision Tree suggests using Algorithm {decision}")
    
    # Apply selected algorithm
    if algorithm == "FCFS":
        schedule = FCFS(jobs)
    elif algorithm == "SJF":
        schedule = SJF(jobs)
    elif algorithm == "SRTF":
        schedule = SRTF(jobs)
    elif algorithm == "RR":
        schedule = RR(jobs, quantum)
    
    # Visualize Results (Gantt Chart using Plotly)
    gantt_data = []
    for job in schedule:
        gantt_data.append({
            'Task': f"Job {job[3]}",
            'Start': job[1],
            'Finish': job[2],
            'Job ID': job[3],
        })
    
    df = pd.DataFrame(gantt_data)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Job ID", title="Job Scheduling Gantt Chart")
    st.plotly_chart(fig)
    
    # Display Schedule Results
    st.write(f"Schedule for {algorithm} Algorithm:")
    schedule_df = pd.DataFrame(schedule, columns=["Arrival Time", "Start Time", "Finish Time", "Job ID"])
    st.write(schedule_df)
