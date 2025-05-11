import util_gameof24
from typing import List


# instructions and prompts
INSTRUCT = "You are a game of 24 grandmaster. We are going to take this problem step by step. At each step, you are going to pick only 2 numbers to operate on. Put only the mathematical expression you choose on the last line and nothing else. Don't put any of the math in latex or markdown formats."
PROMPT_1 = "Numbers: "
PROMPT_2 = "Remaining numbers: "
PROMPT_3 = "Remaining numbers: <INSERT NUMBERS>. This is the last step. If you cannot obtain 24, put 'No' as your final token."


numbers = util_gameof24.import_data()  # load game24 data


def before_equals_to_list(before_equals: str, out_type: str) -> list:
    """
    takes before equals string and returns list of numbers
    ---
    format of before equals should be an expression: '12 + 5'
    """
    # Parse the numbers used in the expression
    number_string = ""
    for c in before_equals:
        if c.isdigit():
            number_string += c
        elif c in ['+', '-', '*', '/']:
            number_string += ','

    used_numbers = number_string.split(',')
    if out_type == "str":
        used_numbers = [num for num in used_numbers if num != '']
    elif out_type == "num":
        used_numbers = [eval(num) for num in used_numbers if num != '']

    return used_numbers


def is_valid_equation(given_numbers: List[int], before_equals: str, after_equals: str) -> bool:
    if eval(before_equals) != eval(after_equals):
        # bad thought
        print("bad thought")
        return False

    used_numbers = before_equals_to_list(before_equals, "str")

    # check if used only 2 numbers are used
    if len(used_numbers) != 2:
        print("not 2")
        return False

    # Check if all provided numbers are used exactly once
    temp_input = [str(num) for num in given_numbers]
    for num in used_numbers:
        if num not in temp_input:
            print("check use")
            return False
        temp_input.remove(num)

    return True


def find_remaining_nums(original: list, before: str, after: str) -> str:
    """
    takes original numbers and 2 chosen numbers. returns remaining numbers
    ---
    format of original is : '[1,2,3,4]'
    """
    used_numbers = before_equals_to_list(before, "num")
    output = [i for i in original]
    for s in used_numbers:
        # can assume s is in output because is_valid_equation is run before this
        output.remove(s)
    output.append(eval(after))
    return output


def evaluate_remaining_numbers(remaining_numbers: List[int]) -> str:
    evaluation_prompt = (
        f"Given the numbers {remaining_numbers}, "
        "how likely is it to reach 24 using +, -, *, / only?\n"
        "Answer with one word: Sure, Maybe, or Impossible."
    )
    response = util_gameof24.ask_chat(
        evaluation_prompt, util_gameof24.MODEL, INSTRUCT)
    if "sure" in response.lower():
        return "Sure"
    elif "maybe" in response.lower():
        return "Maybe"
    else:
        return "Impossible"


def complete_one_problem(quad: List[int], b: int):
    """
    uses ToT to solve problem with [quad] numbers
     - b is the max breadth
     - quad is a list of 4 numbers
    ---
    this function doesn't chain the previous answers to chat
    """
    ### step 1 ###

    # ask chat prompt 1 b times
    step_one_responses = []
    valid_responses = []

    # entry_number = util_gameof24.get_file_date_time()

    str_quad = str(quad)

    for i in range(b):
        response = (util_gameof24.ask_chat(
            PROMPT_1 + str_quad, util_gameof24.MODEL, INSTRUCT))
        step_one_responses.append(response)

        # This is quite hard coded; we'll be in trouble if Chat deviates from instructions
        lines = response.split('\n')
        last_line = lines[-1]
        if len(last_line.strip()) == 0:
            last_line = lines[-2]

        if "=" in last_line:
            try:
                before_equals = last_line.split("=")[0].strip()
                after_equals = last_line.split("=")[1].strip()
                if is_valid_equation(quad, before_equals, after_equals):
                    # correct += 1
                    # valid responses contains tuples
                    # each tuple has [0] the 2 numbers chat combined, and [1] the result
                    remaining = find_remaining_nums(
                        quad, before_equals, after_equals)

                    evaluation = evaluate_remaining_numbers(remaining)
                    if evaluation == "Impossible":
                        print("Pruned a Thought")
                        continue
                    valid_responses.append((before_equals, after_equals))
            except:
                print("step 1: i think format was off: yes equals\n")
                print(last_line)
        else:
            try:
                before_equals = last_line.strip()
                after_equals = str(eval(before_equals))
                if is_valid_equation(quad, before_equals, after_equals):
                    # correct += 1
                    # valid responses contains tuples
                    # each tuple has [0] the 2 numbers chat combined, and [1] the result
                    remaining = find_remaining_nums(
                        quad, before_equals, after_equals)

                    evaluation = evaluate_remaining_numbers(remaining)
                    if evaluation == "Impossible":
                        print("Pruned a Thought")
                        continue
                    valid_responses.append((before_equals, after_equals))
            except:
                print("step 1: i think format was off: no equals\n")
                print(last_line)

            # before_equals = last_line.strip()
            # after_equals = eval(before_equals)

        # if is_valid_equation(quad, before_equals, after_equals):
        #     # correct += 1
        #     # valid responses contains tuples
        #     # each tuple has [0] the 2 numbers chat combined, and [1] the result
        #     remaining = find_remaining_nums(quad, before_equals, after_equals)

        #     evaluation = evaluate_remaining_numbers(remaining)
        #     if evaluation == "Impossible":
        #        print("Pruned a Thought")
        #        continue
        #     valid_responses.append((before_equals, after_equals))

    remaining_numbers_with_eval = [(find_remaining_nums(quad, before, after), evaluate_remaining_numbers(find_remaining_nums(quad, before, after)))
                                   for before, after in valid_responses]

    # Sort by evaluation: Sure = 0, Maybe = 1
    remaining_numbers_with_eval.sort(key=lambda x: 0 if x[1] == "Sure" else 1)

    # Keep only top b
    remaining_numbers_with_eval = remaining_numbers_with_eval[:b]

    # Extract remaining numbers
    remaining_numbers = [x[0] for x in remaining_numbers_with_eval]
    # before step 2, we need to find the 3 numbers we are going to give chat next
    # remaining_numbers = []  # is going to contain lists of remaining 3 numbers
    # for before, after in valid_responses:
    #     remaining_numbers.append(find_remaining_nums(quad, before, after))

    assert len(remaining_numbers) == len(
        valid_responses), "length of remaining numbers doesn't match length of valid responses"

    print("\n\ndone step 1\n\n")

    # step 2

    # for each b responses, ask chat using that response with 3*b responses and rank them. choose the best b
    # for each answer chat gave from step 1, sample 3 times

    # prompt 2

    step_two_responses = []
    valid_child_responses = []

    # before step 3, we need to find the 2 numbers we are going to give chat next
    remaining_child_numbers = []  # is going to contain lists of remaining 2 numbers

    val_length = 0
    val_counter = 0

    for nums in remaining_numbers:
        for _ in range(3):
            child_response = util_gameof24.ask_chat(
                PROMPT_2+str(nums), util_gameof24.MODEL, INSTRUCT)
            step_two_responses.append(child_response)

            # This is quite hard coded; we'll be in trouble if Chat deviates from instructions
            lines = child_response.split('\n')
            last_line = lines[-1]
            if len(last_line.strip()) == 0:
                last_line = lines[-2]
            if "=" in last_line:
                try:
                    child_before = last_line.split("=")[0].strip()
                    child_after = last_line.split("=")[1].strip()
                except:
                    print("step 2: i think format was off: yes equals\n")
                    print(last_line)
            else:
                try:
                    child_before = last_line.strip()
                    child_after = str(eval(child_before))
                except:
                    print("step 2: i think format was off: no equals\n")
                    print(last_line)

            # for each new answer, verify chat's answer is valid; if not, prune
            if is_valid_equation(nums, child_before, child_after):
                valid_child_responses.append((child_before, child_after))
                val_length += 1

        for i in range(val_counter, val_length):
            before, after = valid_child_responses[i]
            remaining_child_numbers.append(
                find_remaining_nums(nums, before, after))
            val_counter += 1

    assert len(valid_child_responses) == len(
        remaining_child_numbers), "length of valid child responses doesn't match length of remaining child numbers"

    print("\n\ndone step 2\n\n")

    # run evaluator on b*3 (possibly less) expressions and pick top b  # not sure how to do this

    ### step 3 ###

    # prompt 3 - give chat 2 numbers and see if it can make 24
    got24 = False
    last_responses = []
    valid_last_responses = []
    for last_nums in remaining_child_numbers:
        last_response = util_gameof24.ask_chat(PROMPT_3.replace(
            "<INSERT NUMBERS>", str(last_nums)), util_gameof24.MODEL, INSTRUCT)
        last_responses.append(last_response)

        if "no" in last_response.lower():
            continue

        lines = last_response.split('\n')
        last_line = lines[-1]
        if len(last_line.strip()) == 0:
            last_line = lines[-2]
        if "=" in last_line:
            try:
                last_before = last_line.split("=")[0].strip()
                last_after = last_line.split("=")[1].strip()
            except:
                print("step 3: i think format was off: yes equals\n")
                print(last_line)
        else:
            try:
                last_before = last_line.strip()
                last_after = str(eval(last_before))
            except:
                print("step 3: i think format was off: no equals\n")
                print(last_line)

        # verify chat's answer is valid
        if is_valid_equation(last_nums, last_before, last_after):
            valid_last_responses.append(last_response)
            if eval(last_before) == 24:
                got24 = True
        if got24:
            print("Chat got 24!")
            break

    if not got24:
        print("chat failed")

    # print(remaining_child_numbers)
    # print(last_responses)
    print(valid_last_responses)

    print("\n\ndone one problem\n\n")

    return got24


def run_experiment(amount, b):
    total = 0
    correct = 0
    if amount == "all":
        amount = len(numbers)
    for i in range(len(numbers)-1, len(numbers)-1-amount, -1):
        quad = numbers[i]
        solved = complete_one_problem(quad, b)
        total += 1
        if solved:
            correct += 1
    return correct/total


if __name__ == '__main__':
    # change parameters depending on which dataset you want to run on
    print(run_experiment(25, 5))
