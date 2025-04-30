import openai
import os
from dotenv import load_dotenv
import pandas as pd
from ast import literal_eval
from datetime import datetime
import tiktoken
from typing import List
from torch import cat


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


def split_data():
    """
    split data by difficulty (number of solutions)
    ---
    one : 90 - rest solutions
    two : 30 - 89 solutions
    three : 20 - 29 solutions
    four : 10 - 19 solutions
    five : 1-9 solutions
    """
    five = []
    four = []
    three = []
    two = []
    one = []

    df = pd.read_csv("gameof24/24game_problems.csv")

    for index, row in df.iterrows():
        if row["num_solutions"] < 10:
            five.append(literal_eval(row["numbers"]))
        elif row["num_solutions"] < 20:
            four.append(literal_eval(row["numbers"]))
        elif row["num_solutions"] < 30:
            three.append(literal_eval(row["numbers"]))
        elif row["num_solutions"] < 90:
            two.append(literal_eval(row["numbers"]))
        else:
            one.append(literal_eval(row["numbers"]))

    return one, two, three, four, five


def string_numbers_to_list(nums: str) -> list:
    """
    puts numbers in [nums] into list
    """
    lis = []
    for n in nums:
        if n.isdigit():
            lis.append(eval(n))
    return lis


def papers_data():
    """
    imports dataset paper uses
    """
    df = pd.read_csv("gameof24/24.csv")
    df["Puzzles"] = df["Puzzles"].apply(extract_numbers)
    # for i in range(len(df["puzzles"])):
    #     df["puzzles", i] = string_numbers_to_list(df["puzzles", i])
    puzzles = df["Puzzles"].tolist()
    return puzzles


def ask_chat(query: str, model: str, instruction: str, temperature: int = 1.0):
    """sends prompt to chattgpt and returns chat's response"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": query
            }
        ],
        temperature = temperature
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


def parse_math_expression(response: str):
    """
    Safely parses the last expression in response that uses some math operations
    Returns: (before_equals, after_equals).
    Throws: ValueError if a math expression cannot be found
    """
    response = response.strip().split('\n')

    math_line = None
    for line in reversed(response):
        # chat may directly copy the "Remaining numbers" line from the prompt.
        if any(op in line for op in ['+', '-', '*', '/', 'x']) and \
                'Remaining numbers' not in line:
            math_line = line.strip()
            math_line = math_line.replace("x", "*")
            break
    if math_line is None:
        raise ValueError("No valid math expression found in response.")

    if "=" in math_line:
        before_equals = math_line.split("=")[0].strip()
        after_equals = math_line.split("=")[1].strip()
    else:
        # If chat doesn't put an equals sign we have to evaluate for him.
        before_equals = math_line.strip()
        after_equals = str(eval(before_equals))
    return before_equals, after_equals


def ask_and_parse(prompt: str, INSTRUCT: str, out_tokens: bool = False, temperature: int = 1.0):
    """
    Asks Chat using prompt. Parses resultant expression.
    Returns None, None for malformed inputs.
    """
    response = ask_chat(prompt, MODEL, INSTRUCT, temperature)

    # print("----- RAW CHAT RESPONSE -----")
    # print(response)
    # print("-----------------------------\n")

    try:
        if out_tokens:
            return parse_math_expression(response), num_tokens(response)
        else:
            return parse_math_expression(response)
    except ValueError:
        return None, None


def extract_numbers(expression: str):
    """
    Extracts numbers from a math expression
    Example: 12 + 6 returns [12,6] (int list)
    """
    numbers = []
    number = ''
    for c in expression:
        # accumulate digits and then add them once we run into an operation
        if c.isdigit():
            number += c
        else:
            if number:
                numbers.append(int(number))
                number = ''
    if number:
        numbers.append(int(number))
    return numbers


def num_tokens(string: str, encoding_name="cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    - from OpenAI
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = len(encoding.encode(string))
    return tokens


def total_cost(input_tokens: int, output_tokens: int) -> float:
    """
    returns total cost using o4-mini
    """
    input_rate = 1.1/1000000  # dollars per one token
    output_rate = 4.4/1000000  # dollars per one token

    return input_rate*input_tokens + output_rate*output_tokens


def pad(state):
    state = list(state)
    return [0] * (4 - len(state)) + state


def extract_features(quad: List[int]) -> List[int]:
    """
    Input: quad of 1 <= length <= 4
    Pads the state until the length is 4. Appends additional features such as 
    if there are factors of 24 in the set.
    Return: list of 17 ints (features)
    """
    features = []
    quad = pad(quad)

    # Are there pairs that sum to nice values?
    for i in range(len(quad)):
        for j in range(i+1, len(quad)):
            a, b = quad[i], quad[j]
            # closeness of sum to 24
            features.append(1 - min(abs((a + b) - 24) / 24, 1.0))
            # closeness of product to 24
            features.append(1 - min(abs((a * b) - 24) / 24, 1.0))
            # closeness of diff to 24
            features.append(1 - min(abs((a - b) - 24) / 24, 1.0))
            if b != 0:
                features.append(1 - min(abs((a / b) - 24) / 24, 1.0))
            else:
                features.append(0.0)

    # Are there factors of 24?
    for key_val in [6, 8, 12, 24, 1]:
        features.append(sum(x == key_val for x in quad))
    mean_val = sum(quad) / len(quad)
    max_val = max(quad)
    min_val = min(quad) if len(quad) > 0 else 0
    std_dev = (sum((x - mean_val)**2 for x in quad) / len(quad))**0.5

    features.extend([mean_val, max_val, std_dev, min_val])
    return quad + features
