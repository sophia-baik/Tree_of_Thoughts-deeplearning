import pandas as pd
import json
import util_gameof24


## instructions ##
INSTRUCT = "You are a game of 24 grandmaster. Put your final equation at the very end on a new line. Don't put any of the math in latex or markdown formats."

numbers = util_gameof24.import_data()  # load game24 data


def get_responses(amount):
    """
    generates chat's responses
    returns file's unique entry number
    """
    entry_number = util_gameof24.get_file_date_time()

    responses = {}

    if amount == "all":
        for i in range(len(numbers)-1, -1, -1):
            query = str(numbers[i])
            responses[query] = util_gameof24.ask_chat(
                query, util_gameof24.MODEL, INSTRUCT)
    else:
        for i in range(len(numbers)-1, len(numbers)-1-amount, -1):
            query = str(numbers[i])
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
    print(not_correct_format if len(not_correct_format)
          != 0 else print("all correct format!"))
    return percentage


def did_he_follow_instructions_and_is_correct(input: list, s: str):
    """
    evaluates if chat followed the instructions (used all 4 numbers provided only once) and provided equation that equals 24
    returns True or False
    """
    input = json.loads(input)
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
    if eval(s) != 24:  # does not == 24
        return False
    return True


if __name__ == "__main__":
    ### CAREFUL WHEN RUNNING GET_RESPONSES ###
    ### WE MIGHT NOT HAVE TO RUN IT EVERY TIME ###

    datetime = get_responses("all")
    percentage = eval_correctness(datetime)
    util_gameof24.update_percentage(
        percentage, datetime, "gameof24/data_zeroshot/percent_correct.csv")
