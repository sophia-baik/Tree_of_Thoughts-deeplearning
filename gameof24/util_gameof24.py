import openai
import os
from dotenv import load_dotenv
import pandas as pd
from ast import literal_eval
from datetime import datetime


## API KEY ##
load_dotenv()
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

MODEL = "gpt-4o-mini"


def import_data():
    """
    reads all permutations of 1-13 where arithmetic operations on four numbers can equal 24
    returns list of lists where each list is a quad
    """
    df = pd.read_csv("gameof24/24game_problems.csv")
    df['numbers'] = df['numbers'].apply(literal_eval)
    numbers = df['numbers'].tolist()
    return numbers


def ask_chat(query: str, model: str, instruction: str):
    """sends prompt to chattgpt and returns chat's response"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": query
            }
        ]
    )

    return (completion.choices[0].message.content)


def get_file_date_time():
    """returns date and time for filename"""
    # Get the current time
    now = datetime.now()

    # Format it nicely for a filename
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    return date_time


def update_percentage(percent: float, datetime: str, filepath: str):
    """adds percent correct for unique datetime at filepath"""
    df = pd.read_csv(filepath)
    new_df = pd.DataFrame({"idx": [datetime], "percent": [percent]})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(filepath, index=False)
