import pandas as pd
import json
import util_gameof24
import random


## instructions ##
INSTRUCT = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''

numbers = util_gameof24.import_data()  # load game24 data


def get_responses(dataset, amount=10):
    """
    dataset is a list of quad numbers of a specific difficulty
    generates chat's responses
    returns file's unique entry number
    """
    entry_number = util_gameof24.get_file_date_time()

    responses = {}

    sampled = random.sample(dataset, amount)

    if amount == "all":
        for i in range(len(numbers)-1, -1, -1):
            query = str(numbers[i])
            responses[query] = util_gameof24.ask_chat(
                query, util_gameof24.MODEL, INSTRUCT)
    else:
        for quad in sampled:
            query = str(quad)
            responses[query] = util_gameof24.ask_chat(
                query, util_gameof24.MODEL, INSTRUCT)

    out_df = pd.DataFrame(list(responses.items()),
                          columns=['query', 'response'])
    out_df.to_csv(
        f"gameof24/data_zeroshot/results_{entry_number}.csv", index=False)

    return entry_number


def eval_correctness(entry_number: str):
    """returns what percent of responses equal 24"""
    correct = 0
    total = 0
    df = pd.read_csv(f"gameof24/data_zeroshot/results_{entry_number}.csv")
    l = len(df['response'])
    not_correct_format = []
    for i in range(l):
        total += 1
        input = df['query'][i]
        try:
            response = df['response'][i].split('\n')
            soln = response[-1]
            soln = soln.split('=')
            soln = soln[0]
            if did_he_follow_instructions_and_is_correct(input, soln):
                correct += 1
        except ZeroDivisionError:
            pass
        except:
            not_correct_format.append(input)

    percentage = round((correct/total), 4)*100
    print(f"percent correct {percentage}%")
    if len(not_correct_format) != 0:
        print(f'num responses w/ incorrect format: {len(not_correct_format)}')
    print(not_correct_format if len(not_correct_format)
          != 0 else print("all correct format!"))
    return percentage


def did_he_follow_instructions_and_is_correct(input: list, s: str):
    """
    evaluates if chat followed the instructions (used all 4 numbers provided only once) and provided equation that equals 24
    returns True or False
    """
    input = json.loads(input)
    empty_s = ''
    for c in s:
        if c.isdigit():
            empty_s += c
        elif c == '+':
            empty_s += ','
        elif c == '-':
            empty_s += ','
        elif c == '*':
            empty_s += ','
        elif c == '/':
            empty_s += ','
    s_lis = empty_s.split(',')
    if len(s_lis) != 4:  # did not use 4 numbers
        return False
    for num in input:
        if str(num) not in s_lis:  # did not use one of the provided 4 numbers
            return False
        else:
            s_lis.remove(str(num))
    if s_lis != []:  # did not use only the provided 4 numbers
        return False
    # get rid of any words before the eqn (they'll typically be precede a :)
    if ':' in s:
        s = s[s.index(':')+1:]
    if 'is' in s:
        s = s[s.index('is')+2:]
    if ',' in s:
        s = s[s.index(',')+1:]
    if eval(s) != 24:  # does not == 24
        return False
    return True


if __name__ == "__main__":
    ### CAREFUL WHEN RUNNING GET_RESPONSES ###
    ### WE MIGHT NOT HAVE TO RUN IT EVERY TIME ###
    one, two, three, four, five = util_gameof24.split_data()

    for i in range(5):
        datetime = get_responses(two)
        percentage = eval_correctness(datetime)
        print(percentage)
    # util_gameof24.update_percentage(
    #     percentage, datetime, "gameof24/data_zeroshot/percent_correct.csv")
