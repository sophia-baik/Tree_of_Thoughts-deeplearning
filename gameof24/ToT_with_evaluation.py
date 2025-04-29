import util_gameof24
import random


# instructions and prompts
INSTRUCT = "You are a game of 24 grandmaster. We are going to take this problem step by step. At each step, you are going to pick only 2 numbers to operate on. Put only the mathematical expression you choose on the last line and nothing else. Don't put any of the math in latex or markdown formats."
PROMPT_1 = "Numbers: "
PROMPT_2 = "Remaining numbers: "
PROMPT_3 = "Remaining numbers: <INSERT NUMBERS>. This is the last step. If you cannot obtain 24, put 'No' as your final token."

short_instruct = "You are a game of 24 evaluator."

value_prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
{input}
'''


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
    format of original is : '[1,2,3,4]'
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

        (before_equals, after_equals), out_tokens = util_gameof24.ask_and_parse(
            PROMPT_1 + str_quad, INSTRUCT, out_tokens=True)

        # step_one_responses.append(response)

        # count tokens
        if count_tokens:
            input_tokens += util_gameof24.num_tokens(PROMPT_1 +
                                                     str_quad) + util_gameof24.num_tokens(INSTRUCT)
            output_tokens += out_tokens

        # This is quite hard coded; we'll be in trouble if Chat deviates from instructions
        # lines = response.split('\n')
        # last_line = lines[-1]
        # if len(last_line.strip()) == 0:
        #     last_line = lines[-2]

        # if "=" in last_line:
        #     try:
        #         before_equals = last_line.split("=")[0].strip()
        #         after_equals = last_line.split("=")[1].strip()
        #     except:
        #         print("step 1: i think format was off: yes equals\n")
        #         print(last_line)
        # else:
        #     try:
        #         before_equals = last_line.strip()
        #         after_equals = str(eval(before_equals))
        #     except:
        #         print("step 1: i think format was off: no equals\n")
        #         print(last_line)

            # before_equals = last_line.strip()
            # after_equals = eval(before_equals)

        if is_valid_equation(quad, before_equals, after_equals):
            # correct += 1
            # valid responses contains tuples
            # each tuple has [0] the 2 numbers chat combined, and [1] the result
            valid_responses.append((before_equals, after_equals))

    # before step 2, we need to find the 3 numbers we are going to give chat next
    remaining_numbers = []  # is going to contain lists of remaining 3 numbers
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

    # select b
    if len(sure_nums) >= b:
        remaining_numbers = random.sample(sure_nums, b)
    elif len(sure_nums) + len(likely_nums) >= b:
        left = b - len(sure_nums)
        remaining_numbers = sure_nums + random.sample(likely_nums, left)
    else:
        remaining_numbers = sure_nums + likely_nums

    # print(remaining_numbers)

    # assert len(remaining_numbers) == len(
    #     valid_responses), "length of remaining numbers doesn't match length of valid responses"

    # print("\n\ndone step 1\n\n")

    ### step 2 ###

    # for each b responses, ask chat using that response with 3*b responses and rank them. choose the best b
    # for each answer chat gave from step 1, sample 3 times

    # prompt 2

    step_two_responses = []
    valid_child_responses = []
    sure_nums = []
    likely_nums = []

    # before step 3, we need to find the 2 numbers we are going to give chat next
    remaining_child_numbers = []  # is going to contain lists of remaining 2 numbers

    val_length = 0
    val_counter = 0

    for nums in remaining_numbers:
        for _ in range(3):
            # child_response = util_gameof24.ask_chat(
            #     PROMPT_2+str(nums), util_gameof24.MODEL, INSTRUCT)
            # step_two_responses.append(child_response)

            (child_before, child_after), out_tokens = util_gameof24.ask_and_parse(
                PROMPT_2+str(nums), INSTRUCT, out_tokens=True)

            # count tokens
            if count_tokens:
                input_tokens += util_gameof24.num_tokens(PROMPT_2 +
                                                         str(nums)) + util_gameof24.num_tokens(INSTRUCT)
                output_tokens += out_tokens

            # # This is quite hard coded; we'll be in trouble if Chat deviates from instructions
            # lines = child_response.split('\n')
            # last_line = lines[-1]
            # if len(last_line.strip()) == 0:
            #     last_line = lines[-2]
            # if "=" in last_line:
            #     try:
            #         child_before = last_line.split("=")[0].strip()
            #         child_after = last_line.split("=")[1].strip()
            #     except:
            #         print("step 2: i think format was off: yes equals\n")
            #         print(last_line)
            # else:
            #     try:
            #         child_before = last_line.strip()
            #         child_after = str(eval(child_before))
            #     except:
            #         print("step 2: i think format was off: no equals\n")
            #         print(last_line)

            # for each new answer, verify chat's answer is valid; if not, prune
            if is_valid_equation(nums, child_before, child_after):
                valid_child_responses.append((child_before, child_after))
                val_length += 1

        for i in range(val_counter, val_length):
            before, after = valid_child_responses[i]
            # get remanining numbers
            # rem_nums = find_remaining_nums(nums, before, after)

            # # ask chat to evaluate
            # response = util_gameof24.ask_chat(value_prompt.replace(
            #     "{input}", str(rem_nums)), util_gameof24.MODEL, short_instruct)
            # response = parse_chats_eval_answer(response)
            # if response == "sure":
            #     sure_nums.append(rem_nums)
            # elif response == "maybe":
            #     maybe_nums.append(rem_nums)

            remaining_child_numbers.append(
                find_remaining_nums(nums, before, after))
            val_counter += 1

    # # select b
    # if len(sure_nums) >= b:
    #     remaining_child_numbers = random.sample(sure_nums, b)
    # elif len(sure_nums) + len(maybe_nums) >= b:
    #     left = b - len(sure_nums)
    #     remaining_child_numbers = sure_nums + random.sample(maybe_nums, left)
    # else:
    #     remaining_child_numbers = sure_nums + maybe_nums

    # print(remaining_child_numbers)

    # assert len(valid_child_responses) == len(
    #     remaining_child_numbers), "length of valid child responses doesn't match length of remaining child numbers"

    # print("\n\ndone step 2\n\n")

    # run evaluator on b*3 (possibly less) expressions and pick top b  # not sure how to do this

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
            # print("Chat got 24!")
            break

    # if not got24:
    #     print("chat failed")

    # print(remaining_child_numbers)
    # print(last_responses)
    # print(valid_last_responses)

    print("\n\ndone one problem\n\n")

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


def cost_of_one_problem(index):
    quad = numbers[index]
    solved, input_tokens, output_tokens = complete_one_problem(quad, 5)
    return util_gameof24.total_cost(input_tokens, output_tokens)


if __name__ == '__main__':
    # print(cost_of_one_problem(362))
    # outputs = []
    x = run_experiment(1, 5)
    # outputs.append(x)
    print(x)  # returns 0.68 on first run
    # print("\nfinal outputs\n")
    # print(outputs)
    # complete_one_problem([4, 6, 12, 13], 5)
    # print(is_valid_equation([2, 3, 5, 12], "12 + 5", "17"))
