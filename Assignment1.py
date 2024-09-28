# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:53:29 2024

@author: vidya
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Filter Dataset for 2016 Campaigns
csv_path = 'C:/Users/vidya/data/kickstarter.csv'

@st.cache_data
def load_full_data(file_path):
    return pd.read_csv(file_path)

kickstarter_full = load_full_data(csv_path)
kickstarter_2016 = kickstarter_full[kickstarter_full['Launched'].str.startswith('2016')]

st.title("Kickstarter Campaign Analysis - 2016")

# Display the filtered dataset
st.subheader("Filtered Dataset for 2016 Campaigns")
st.write(kickstarter_2016)

# Step 2: Initial Exploration and Anomaly Detection
st.subheader("Initial Dataset Exploration")
critical_columns = ['Category', 'Goal', 'Pledged', 'Backers', 'State']
st.write("Critical Columns for Analysis:", critical_columns)
st.write(kickstarter_2016[critical_columns].describe())
anomalies = kickstarter_2016[(kickstarter_2016['Backers'] == 0) & (kickstarter_2016['Pledged'] > 0)]
st.write("anomalies: ", anomalies)

#Step 3
# Standardize state labels to lowercase to avoid mismatch issues
kickstarter_2016['State'] = kickstarter_2016['State'].str.lower()

# Chat with Data Section
st.header("Chat with Data: Is the Dataset Balanced for Campaign States?")
st.write("""
Based on a question from Basole & Major (2024), we want to explore whether the dataset is balanced with regard to successful and unsuccessful campaigns.
This chart, as it allows for easy comparison of the distribution between different categories.
""")

# Add a button to display the plot
if st.button("Show Campaign State Distribution"):
    # Count the number of campaigns for each state
    state_counts = kickstarter_2016['State'].value_counts()

    # Create the bar plot
    fig, ax = plt.subplots()
    state_counts.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gray'], ax=ax)
    ax.set_title("Distribution of Campaign States in 2016")
    ax.set_xlabel("Campaign State")
    ax.set_ylabel("Number of Campaigns")
    plt.xticks(rotation=45)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Determine if the dataset is balanced
    successful_count = state_counts.get('successful', 0)
    failed_count = state_counts.get('failed', 0)
    balance_message = (
        f"The dataset is not balanced. There are {successful_count} successful campaigns and {failed_count} failed campaigns."
        if successful_count != failed_count
        else "The dataset is balanced between successful and failed campaigns."
    )
    st.write(balance_message)
else:
    st.write("Click the button above to display the distribution of campaign states.")


# Step 4: Top Categories for Successful and Failed Campaigns
st.subheader("Top 3 Categories for Successful and Failed Campaigns")
category_success_fail = kickstarter_2016.groupby(['Category', 'State']).size().unstack(fill_value=0)
top_successful_categories = category_success_fail['successful'].nlargest(3)
top_failed_categories = category_success_fail['failed'].nlargest(3)
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
top_successful_categories.plot(kind='bar', color='green', ax=ax2)
top_failed_categories.plot(kind='bar', color='red', ax=ax3)
plt.tight_layout()
st.pyplot(fig2)

# Step 5: Distribution of Log-Transformed Funding Goals
st.subheader("Distribution of Log-Transformed Funding Goals")
kickstarter_2016['Log_Goal'] = np.log10(kickstarter_2016['Goal'] + 1)
fig3, ax4 = plt.subplots()
ax4.hist(kickstarter_2016['Log_Goal'], bins=50, color='purple', alpha=0.7)
ax4.set_title("Distribution of Log-Transformed Funding Goals")
ax4.set_xlabel("Log10(Goal + 1)")
ax4.set_ylabel("Number of Campaigns")
st.pyplot(fig3)

