import os
from dotenv import load_dotenv
from zeno_client import ZenoClient, ZenoMetric
import pandas as pd

load_dotenv()
API_KEY = os.getenv("ZENO_API_KEY")

if not API_KEY:
    raise ValueError("API_KEY is not set! Please check your .env file.")

print(f"API Key Loaded: {API_KEY[:5]}... (hidden for security)")

df = pd.read_csv('tweets.csv')
df = df.reset_index()
df["index"] = df["index"].astype(str)  # Fix dtype issue

client = ZenoClient(API_KEY)

project = client.create_project(
    name="Tweet Sentiment Analysis",
    view="text-classification",
    metrics=[
        ZenoMetric(name="accuracy", type="mean", columns=["correct"]),
    ]
)

project.upload_dataset(df, id_column="index", data_column="text", label_column="label")

models = ['roberta', 'gpt2']
for model in models:
    df_system = df[['index', model]].copy()  # Prevent SettingWithCopyWarning
    df_system.loc[:, "correct"] = (df_system[model] == df["label"]).astype(int)  # Fix warning
    project.upload_system(df_system, name=model, id_column="index", output_column=model)

print("Data uploaded successfully to Zeno!")
