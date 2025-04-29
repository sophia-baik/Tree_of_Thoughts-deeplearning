import openai
import os
from dotenv import load_dotenv
import pandas as pd
from ast import literal_eval
from datetime import datetime
import tiktoken


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
        if any(op in line for op in ['+', '-', '*', '/']) and \
                'Remaining numbers' not in line:
            math_line = line.strip()
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


def ask_and_parse(prompt: str, INSTRUCT: str, out_tokens: bool = False):
    """
    Asks Chat using prompt. Parses resultant expression.
    Returns None, None for malformed inputs.
    """
    response = ask_chat(prompt, MODEL, INSTRUCT)
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
