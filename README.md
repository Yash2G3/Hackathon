# Hack-O-Hire

# Website Link: https://hackathon-hack.streamlit.app/

## Problem Statement

In financial markets, detecting anomalies in bank statements is crucial for maintaining security and trust. However, traditional methods often fall short in efficiently identifying irregularities, leading to potential risks and losses for both financial institutions and their clients. Barclays, a leading global financial services provider, aims to enhance anomaly detection in bank statements to safeguard against fraudulent activities and ensure the integrity of financial transactions.

## Objective

The objective of this hackathon challenge is to develop a robust anomaly detection system for financial bank statements using the Isolation Forest algorithm. The system should accurately flag anomalous transactions based on predefined criteria, such as the sum of withdrawals over a period of five days relative to the number of transactions within the same period.

## Proposed Prototype

### Data Preprocessing:

- Gather historical bank transaction data, including withdrawal amounts and timestamps.
- Group transactions into sets of five days.
- Calculate the sum of withdrawals for each five-day period.

### Isolation Forest Algorithm:

- Implement the Isolation Forest algorithm from the PyOD library to identify anomalies within the transaction data.
- Train the model on preprocessed bank statement data, considering withdrawal sums as features.
- Define a threshold for anomaly detection based on the relative withdrawal sums and the number of transactions.

### Visualization:

- Utilize Matplotlib to visualize the bank statement data and highlight detected anomalies.
- Generate plots illustrating withdrawal sums over time and mark anomalies for easy interpretation by users.

### Web Application:

- Develop a simple web application using Flask to provide a user-friendly interface for anomaly detection.
- Allow users to upload bank statement data files or input transaction details directly.
- Display detected anomalies and relevant insights, including timestamps and withdrawal amounts.

## Tech Stack

- **Scikit-Learn:** Utilize Scikit-Learn for data preprocessing and training the Isolation Forest model.
- **Matplotlib:** Create visualizations to enhance the understanding of bank statement data and detected anomalies.
- **PyOD:** Implement the Isolation Forest algorithm for anomaly detection within the financial data.
- **Flask:** Develop a lightweight web application framework to deploy the anomaly detection system with a user interface.

## Expected Outcome

The proposed prototype aims to provide Barclays with a scalable and efficient solution for detecting anomalies in bank statements. By leveraging the Isolation Forest algorithm and the specified tech stack, the system will empower financial institutions to identify potential fraudulent activities promptly, thereby safeguarding their assets and maintaining trust with clients.
