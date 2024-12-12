from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from uuid import uuid4
from transformers import pipeline
import streamlit as st

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

chat_llm = ChatOpenAI(openai_api_key = st.secrets["OPENAI_API_KEY"])

graph = Neo4jGraph(
    url="bolt://3.91.170.6:7687",
    username="neo4j",
    password="flame-correction-pack"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a psychiatrist, having conversation with a patient. Respond in doctors slang, ask questions to make patient open up and be polite",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{question}"),
    ]
)

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id,graph=graph)

chat_chain = prompt | chat_llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

list_of_emotions=[]
print("Hi, How are u feeling today?")
while True:
    question = input("User >> ")
    list_of_emotions.append(question)
    response = chat_with_message_history.invoke(
        {
            "question":question
        },
        config={
            "configurable":{"session_id":SESSION_ID}
        }
    )
    if "bye" in question.lower():
        break
    print("Bot: >> ",response)

# Detect Emotion
pipe = pipeline("text-classification", model=st.secrets["SENTIMENT_ANALYSIS_MODEL"], top_k=None)
output = pipe(list_of_emotions)

for option in output:
  print(option[0],option[1])
  print("-----")


######### Over Sampling ###########
# from imblearn.over_sampling import SMOTE
# from collections import Counter

# counter = Counter(y_train)
# print('Before', counter)

# # oversampling the train dataset using SMOTE
# smt = SMOTE()
# X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

# counter = Counter(y_train_sm)
# print('After', counter)