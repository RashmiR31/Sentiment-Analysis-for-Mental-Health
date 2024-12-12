# Sentiment Analysis for Mental Health

## Project Overview

This project focuses on the development of a sentiment analysis system for mental health assessment. By leveraging a Bidirectional LSTM (Bi-LSTM) model, GPT-4, and graph-based sentiment analysis, the system can classify sentiments from user conversations, helping detect and intervene in mental health issues such as depression, anxiety, and stress. It provides a chatbot-based platform that offers users a safe, anonymous space to discuss their mental health without fear of judgment.

## Introduction

Mental health issues such as depression, anxiety, and stress have become increasingly prevalent in society, necessitating effective tools for early detection and intervention. Traditional methods of mental health assessment involve psychiatrists, but many people hesitate to seek help due to the fear of judgment.

## Problem Statement

This project aims to develop a chatbot that allows individuals to openly discuss their mental health without fear of judgment. The system will use sentiment analysis to classify users' mental health statuses based on their textual input. Additionally, a graph-based model will be used to predict the overall sentiment of a conversation.

## Architecture
<img width="476" alt="Screenshot 2024-12-12 at 3 17 50 PM" src="https://github.com/user-attachments/assets/3792f908-3ecd-4ace-ac3b-bbe1203e127e" />

## Project's Approach

1. **Bi-LSTM Model with Context Preservation**: Bi-LSTM captures both past and future context, making it ideal for understanding subtle emotional cues in mental health-related texts.
2. **Graph-based Sentiment Aggregation**: By aggregating sentiments at the conversation level, the system provides better context for mental health professionals.
3. **Safe, Judgment-free User Interaction**: The chatbot offers a non-judgmental space for users to express themselves.
4. **Hybrid Approach**: The combination of Bi-LSTM for message-level sentiment analysis and graph-based aggregation ensures robust sentiment detection.

## Implementation

**Preprocessing Steps**:
1. Tokenization: Breaking down text into words.
2. Lowercasing: Ensuring consistency.
3. Stopword Removal: Removing non-informative words.
4. Lemmatization: Reducing words to their base form.
5. Padding: Ensuring uniform input length.

### Handling Imbalanced Data - SMOTE Oversampling Technique

To address class imbalance, the SMOTE technique is applied to generate synthetic samples for minority classes. This helps the model learn to classify all mental health statuses effectively.

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

# Oversample the training data
counter = Counter(y_train)
print('Before', counter)

smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

counter = Counter(y_train_sm)
print('After', counter)
```
## Model Development & Evaluation

The **Bi-LSTM (Bidirectional Long Short-Term Memory)** model is chosen for sentiment analysis due to its ability to process sequential text in both forward and backward directions, which is crucial for understanding the context of mental health-related text. Bi-LSTM is well-suited to detect subtle emotional cues in conversations, making it ideal for this project.

### Model Architecture

The architecture of the Bi-LSTM model includes the following layers:
1. **Embedding Layer**: Converts words into dense vectors of a fixed size.
2. **Bi-LSTM Layer**: Processes the word embeddings in both forward and backward directions to capture context.
3. **Dense Layer**: Maps the output of the Bi-LSTM layer to sentiment classes (positive, negative, neutral).
4. **Softmax Activation**: Outputs a probability distribution over sentiment classes.

The model is trained on the preprocessed text data using **Categorical Cross-Entropy Loss** and the **Adam Optimizer**.

### Model Training and Evaluation

The model was trained using the preprocessed dataset, which has been tokenized and padded to ensure consistent input length. The tokenizer was configured to retain the top 20,000 most frequent words, and the input sequences were padded to a maximum length of 1,000 tokens.

## Graph Modeling

In addition to analyzing individual messages, the sentiment of the entire conversation is predicted using **graph modeling**. The goal is to build a graph where each **node** represents a message and **edges** represent the flow of conversation.

### Graph Construction

- **Nodes**: Each user's message is represented as a node.
- **Edges**: Directed edges are created between consecutive messages to represent the flow of the conversation.

### Sentiment Prediction and Aggregation

For each node (message), the Bi-LSTM model predicts a sentiment (positive, negative, neutral). These sentiments are then aggregated across the conversation to determine the overall sentiment.

### Example of Cypher Code for Neo4j Graph Database

To store and retrieve conversation data in a graph database (Neo4j), the following **Cypher** query is used:

```cypher
MERGE (u:User {session_id:SESSION_ID}) -[:NEXT]->(c:Conversation {message:MESSAGE})
```

<img width="424" alt="Screenshot 2024-12-12 at 3 24 46 PM" src="https://github.com/user-attachments/assets/b1b4345e-fbcd-4de2-a938-97c8e9eebf03" />
<img width="562" alt="Screenshot 2024-12-12 at 3 25 02 PM" src="https://github.com/user-attachments/assets/bc63e544-78d5-45bd-a723-985b116925f4" />

# Final Output

<img width="654" alt="Screenshot 2024-12-12 at 3 25 18 PM" src="https://github.com/user-attachments/assets/87a6a1fd-bb76-4fc6-a286-700b22d75e63" />
<img width="666" alt="Screenshot 2024-12-12 at 3 25 39 PM" src="https://github.com/user-attachments/assets/1da85a6d-55ff-4793-b611-be1fb36e22d5" />


