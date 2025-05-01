import util_gameof24
import random
import time
import math

# instructions and prompts
INSTRUCT = "You are a game of 24 grandmaster. We are going to take this problem step by step. At each step, you are going to pick only 2 numbers to operate on. Put only the mathematical expression you choose on the last line and nothing else. Don't put any of the math in latex or markdown formats."
PROMPT_1 = "Numbers: "
PROMPT_1_longer = "Numbers: <NUMBERS>. Don't pick these operations you've already tried: <THOUGHTS>"
PROMPT_2 = "Remaining numbers: "
PROMPT_2_longer = "Remaining numbers: <NUMBERS>. Don't pick these operations you've already tried: <THOUGHTS>"
PROMPT_3 = "Remaining numbers: <INSERT NUMBERS>. This is the last step. If you cannot obtain 24, put 'No' as your final token."

short_instruct = "You are a game of 24 evaluator."

value_prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
{input}
'''


numbers = util_gameof24.import_data()  # load game24 data
puzzles = util_gameof24.papers_data()  # load game24 data from paper


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


def is_valid_equation(given_numbers: list[int], before_equals: str, after_equals: str) -> bool:
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
    format of original is : [1,2,3,4]
    """
    used_numbers = before_equals_to_list(before, "num")
    output = [i for i in original]
    for s in used_numbers:
        # can assume s is in output because is_valid_equation is run before this
        output.remove(s)
    output.append(eval(after))
    return output


def parse_chats_eval_answer(response: str) -> str:
    """takes long response and returns sure/likely/impossible"""
    lines = response.split('\n')
    for i in range(len(lines)-1, -1, -1):
        line = lines[i].lower()
        if "sure" in line:
            return "sure"
        elif "likely" in line:
            return "likely"
        elif "impossible" in line:
            return "impossible"


def check24(a: int, b: int) -> bool:
    """run all possible operations and see if 24 works"""
    if math.isclose(a+b, 24):
        return True
    elif math.isclose(a-b, 24):
        return True
    elif math.isclose(b-a, 24):
        return True
    elif math.isclose(a*b, 24):
        return True
    elif a != 0 and math.isclose(b/a, 24):
        return True
    elif b != 0 and math.isclose(a/b, 24):
        return True
    return False


def complete_one_problem(quad: list[int], b: int, k: int = 3, count_tokens: bool = False):
    """
    uses ToT to solve problem with [quad] numbers
     - b is the max breadth
     - quad is a list of 4 numbers
    ---
    this function doesn't chain the previous answers to chat
    """
    ### step 1 ###

    input_tokens = 0
    output_tokens = 0

    # ask chat prompt 1 b times
    step_one_responses = []
    valid_responses = []

    # entry_number = util_gameof24.get_file_date_time()

    str_quad = str(quad)

    for i in range(b*k):
        # response = (util_gameof24.ask_chat(
        #     PROMPT_1 + str_quad, util_gameof24.MODEL, INSTRUCT))
        if len(step_one_responses) == 0:
            promp = PROMPT_1 + str_quad
        else:
            promp = PROMPT_1_longer.replace("<NUMBERS>", str_quad).replace(
                "<THOUGHTS>", str(step_one_responses))

        (before_equals, after_equals), out_tokens = util_gameof24.ask_and_parse(
            promp, INSTRUCT, out_tokens=True)

        # step_one_responses.append(response)

        # count tokens
        if count_tokens:
            input_tokens += util_gameof24.num_tokens(
                promp) + util_gameof24.num_tokens(INSTRUCT)
            output_tokens += out_tokens

        if is_valid_equation(quad, before_equals, after_equals):
            # correct += 1
            # valid responses contains tuples
            # each tuple has [0] the 2 numbers chat combined, and [1] the result
            if before_equals not in step_one_responses:
                step_one_responses.append(before_equals)
                valid_responses.append((before_equals, after_equals))

    # print(step_one_responses)

    # before step 2, we need to find the 3 numbers we are going to give chat next
    # remaining_numbers = []  # is going to contain lists of remaining 3 numbers
    sure_nums = []
    likely_nums = []
    for before, after in valid_responses:
        # get remanining numbers
        rem_nums = find_remaining_nums(quad, before, after)

        # ask chat to evaluate
        response = util_gameof24.ask_chat(value_prompt.replace(
            "{input}", str(rem_nums)), util_gameof24.MODEL, short_instruct)

        # count tokens
        if count_tokens:
            input_tokens += util_gameof24.num_tokens(value_prompt.replace(
                "{input}", str(rem_nums))) + util_gameof24.num_tokens(short_instruct)
            output_tokens += util_gameof24.num_tokens(response)

        response = parse_chats_eval_answer(response)
        if response == "sure":
            sure_nums.append(rem_nums)
        elif response == "likely":
            likely_nums.append(rem_nums)

    combined = sure_nums + likely_nums
    remaining_numbers = combined[:b]  # select b
    backtrack_list = combined[b:]  # come back to this list if necessary

    print(f"backtrack list: {backtrack_list}")

    # # select b
    # if len(sure_nums) >= b:
    #     remaining_numbers = random.sample(sure_nums, b)
    # elif len(sure_nums) + len(likely_nums) >= b:
    #     left = b - len(sure_nums)
    #     remaining_numbers = sure_nums + random.sample(likely_nums, left)
    # else:
    #     remaining_numbers = sure_nums + likely_nums

    ### step 2 ###

    # for each b responses, ask chat using that response with 3*b responses and rank them. choose the best b
    # for each answer chat gave from step 1, sample 3 times

    # before step 3, we need to find the 2 numbers we are going to give chat next
    remaining_child_numbers = []  # is going to contain lists of remaining 2 numbers

    while len(remaining_child_numbers) < b and len(backtrack_list) > 0:
        # keep going back to backtrack list for previous thoughts until we have b good remaining_child_numbers
        valid_child_responses = []
        val_length = 0
        val_counter = 0

        for nums in remaining_numbers:
            step_two_responses = []
            for _ in range(3):
                str_nums = str(nums)

                if len(step_two_responses) == 0:
                    promp = PROMPT_2 + str_nums
                else:
                    promp = PROMPT_2_longer.replace("<NUMBERS>", str_nums).replace(
                        "<THOUGHTS>", str(step_two_responses))

                (child_before, child_after), out_tokens = util_gameof24.ask_and_parse(
                    promp, INSTRUCT, out_tokens=True)

                # count tokens
                if count_tokens:
                    input_tokens += util_gameof24.num_tokens(
                        promp) + util_gameof24.num_tokens(INSTRUCT)
                    output_tokens += out_tokens

                # for each new answer, verify chat's answer is valid; if not, prune
                if is_valid_equation(nums, child_before, child_after):
                    if child_before not in step_two_responses:
                        step_two_responses.append(child_before)
                        valid_child_responses.append(
                            (child_before, child_after))
                        val_length += 1

            for i in range(val_counter, val_length):
                before, after = valid_child_responses[i]
                # get remanining numbers

                lasttwo = find_remaining_nums(nums, before, after)
                if check24(lasttwo[0], lasttwo[1]):
                    remaining_child_numbers.append(lasttwo)
                    val_counter += 1

        print(f"remaining child numbers: {remaining_child_numbers}")

        # need nn more numbers
        nn = b - len(remaining_child_numbers)
        remaining_numbers = backtrack_list[:nn]
        backtrack_list = backtrack_list[nn:]

    ### step 3 ###

    # prompt 3 - give chat 2 numbers and see if it can make 24
    got24 = False
    last_responses = []
    valid_last_responses = []
    for last_nums in remaining_child_numbers:
        last_response = util_gameof24.ask_chat(PROMPT_3.replace(
            "<INSERT NUMBERS>", str(last_nums)), util_gameof24.MODEL, INSTRUCT)
        # last_responses.append(last_response)

        # count tokens
        if count_tokens:
            input_tokens += util_gameof24.num_tokens(PROMPT_3.replace(
                "<INSERT NUMBERS>", str(last_nums))) + util_gameof24.num_tokens(INSTRUCT)
            output_tokens += util_gameof24.num_tokens(last_response)

        if "no" in last_response.lower():
            continue

        last_before, last_after = util_gameof24.parse_math_expression(
            last_response)

        # verify chat's answer is valid
        if is_valid_equation(last_nums, last_before, last_after):
            valid_last_responses.append(last_response)
            if eval(last_before) == 24:
                got24 = True
        if got24:
            # print("Chat got 24!")
            break

    # if not got24:
    #     print("chat failed")

    return got24, input_tokens, output_tokens


def run_experiment(amount, b):
    total = 0
    correct = 0
    if amount == "all":
        amount = len(numbers)
    for i in range(362, 462):
        # for i in range(len(numbers)-1, len(numbers)-1-amount, -1):
        quad = numbers[i]
        solved, _, _ = complete_one_problem(quad, b)
        total += 1
        if solved:
            correct += 1
    return correct/total


def run_difficulties_experiment(dataset, amount=10, b=5):
    """dataset is a list of quad numbers of a specific difficulty"""
    total = 0
    correct = 0
    avg_time = 0

    sampled = random.sample(dataset, amount)
    for quad in sampled:
        start = time.time()
        solved, _, _ = complete_one_problem(quad, b)
        end = time.time()
        avg_time += end - start
        total += 1
        if solved:
            print(f"chat got 24!")
            correct += 1
        else:
            print(f"chat failed")
        print(f"done {total}")

    return correct/total, avg_time/total


def run_papers_experiment():
    total = 0
    correct = 0
    for i in range(901, 906):
        quad = puzzles[i]
        solved, _, _ = complete_one_problem(quad, 5)
        total += 1
        if solved:
            correct += 1
    return correct/total


def cost_of_one_problem(index):
    quad = numbers[index]
    solved, input_tokens, output_tokens = complete_one_problem(
        quad, 5, count_tokens=True)
    return util_gameof24.total_cost(input_tokens, output_tokens), input_tokens, output_tokens


if __name__ == '__main__':
    one, two, three, four, five = util_gameof24.split_data()
    score, ti = run_difficulties_experiment(one)
    print(f"score: {score}")
    print(f"time: {ti}")

    # for i in range(10):
    #     x = cost_of_one_problem(i)
    #     print(x[0], x[1], x[2])
    # outputs = []
    # x = run_experiment(1, 5)
    # outputs.append(x)
    # x = run_papers_experiment()
    # print(x)  # returns 0.68 on first run
    # print("\nfinal outputs\n")
    # print(outputs)
    # complete_one_problem([4, 6, 12, 13], 5)
    # print(is_valid_equation([2, 3, 5, 12], "12 + 5", "17"))
