import util_gameof24
from value_function_gameof24 import ValueNetwork 
import torch

# Instructions and Prompts
INSTRUCT = "You are a game of 24 grandmaster. We are going to take this problem step by step. At each step, you are going to pick only 2 numbers to operate on. Put only the mathematical expression you choose on the last line and nothing else. Don't put any of the math in latex or markdown formats."
PROMPTS = {
    "step1": "Numbers: ",
    "step2": "Remaining numbers: ",
    "step3": "Remaining numbers: <INSERT NUMBERS>. This is the last step. If you cannot obtain 24, put 'No'."
}

# Load game of 24 data
# Note that the hardest problems are stored at the beginning.
df = util_gameof24.import_data() 

# Load weights from pretrained network
value_model = ValueNetwork()
value_model.load_state_dict(torch.load('gameof24/value_network.pth', weights_only = True))
value_model.eval()

def evaluate_state(numbers):
    """
    Since the ValueNetwork was trained on states that were sorted and padded,
    we should sort and pad correspondingly. Returns score of state.
    """
    numbers = sorted(numbers)
    padded = [0] * (4 - len(numbers)) + numbers
    x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        score = value_model(x).item()
    return score

def before_equals_to_list(before_equals: str, out_type: str) -> list:
    """
    takes before equals string and returns list of numbers
    ---
    format of before equals should be an expression: '12 + 5'
    can remove
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
    """
    Checks whether the given expression is valid using the following criteria:
    1. Evaluates whether before_equals evaluates to after_equals.
    2. Checks that only two numbers were chosen.
    3. Checks that the a given number was used at most once.
    """
    if eval(before_equals) != eval(after_equals):
        print("Invalid Equation: before_equals is not equivalent to after_equals")
        return False

    used_numbers = util_gameof24.extract_numbers(before_equals)
    if len(used_numbers) != 2:
        print("Invalid Equation: More than two numbers used.")
        return False

    temp = given_numbers.copy()
    for num in used_numbers:
        if num not in temp:
            print("Invalid Equation: %d incorrectly used (duplicate usage/unavailable)",num)
            return False
        temp.remove(num)
    return True


def find_remaining_nums(original: list, before: str, after: str) -> str:
    """
    Input: original: list of numbers
    ---
    format of original is : '[1,2,3,4]'
    Example: find_remaining_nums([1,2,3,4], "(1 + 2)", "3") will return 3, 4
    """
    used_numbers = util_gameof24.extract_numbers(before)
    output = original.copy()
    for s in used_numbers:
        # can assume s is in output because is_valid_equation is run before this
        output.remove(int(s))
    output.append(eval(after))
    return output

def ask_and_parse(prompt):
    """
    Ask the model and parse safely.
    """
    response = util_gameof24.ask_chat(prompt, util_gameof24.MODEL, INSTRUCT)
    try:
        return util_gameof24.parse_math_expression(response)
    except ValueError:
        return None, None


def complete_one_problem(quad: list[int], b: int, k: int = 3):
    """
    Input:
        quad: list of four numbers we are trying to solve.
        b: number of remaining candidates after each step.
        k: determines breadth of candidates we create before pruning.
    ---
    Solves one game of 24 problem with Tree of Thoughts using the following architecture:
        Step 1: Query Chat to combine two of the four initial numbers b * k times.
                Find the k states with the highest scores according to the evaluator.
                Proceed to Step 2 using those k states.
        Step 2: With three numbers remaining, query Chat to combine two numbers b * k times.
                Find the k states with the highest scores according to the evaluator.
                Proceed to Step 3 using those k states.
        Step 3: Check if Chat has reached 24 (and claims to reach 24) in any of the states.
    Total: 2 * b * k queries per problem.
    Note: This function does not append responses from Step 1 to the Step 2 prompt (chaining).
    ---
    Returns: solved?: bool, 
    """
    ### Step 1 ###
    step1_candidates = []

    # ask chat prompt 1 b times
    solution_trace = []

    # Step 1: Query Chat to combine two of the four initial numbers b * k times
    for i in range(b * k):
        before, after = ask_and_parse(PROMPTS["step1"] + str(quad))
        if before and after and is_valid_equation(quad, before, after):
            new_nums = find_remaining_nums(quad, before, after)
            score = evaluate_state(new_nums)
            trace = [before + " = " + after]
            step1_candidates.append((score, new_nums, trace))

    if not step1_candidates:
        print("Error: Step 1 failed with no valid candidates.")
        return False

    # Step 1: Find the k states with the highest scores according to the evaluator.
    step1_candidates.sort(reverse=True, key=lambda x: x[0])
    top_states = step1_candidates[:b]
    print("\Step 1 Complete.\n")

    ### Step 2 ###
    step2_candidates = []

    # Step 2: Query Chat to combine two of the four initial numbers b * k times
    for score, state, trace in top_states:
        for i in range(k):
            before, after = ask_and_parse(PROMPTS["step2"] + str(quad))
            if before and after and is_valid_equation(quad, before, after):
                new_nums = find_remaining_nums(quad, before, after)
                score = evaluate_state(new_nums)
                trace = trace + [", " + before + " = " + after]
                step2_candidates.append((score, new_nums, trace))

    if not step2_candidates:
        print("Error: Step 2 failed with no valid candidates.")
        return False

    # Step 2: Find the k states with the highest scores according to the evaluator.
    step2_candidates.sort(reverse=True, key=lambda x: x[0])
    top_states = [:b]
    print("\Step 2 Complete.\n")

    ### Step 3 ###
    valid_response_attempts = []

    # Step 3: Query Chat to combine the remaining two numbers min(k, 4) times.
    for score, state, trace in step2_candidates:
        prompt = PROMPTS["step3"].replace("<INSERT NUMBERS>", str(state))
        before, after = ask_and_parse(prompt)
        if before and after and is_valid_equation(state, before, after):
            trace = trace + [", " + before + " = " + after]
            if abs(eval(before) - 24) < 1e-6:
                print("Chat solved it using expression " + trace)
                return True
            valid_response_attempts.append(trace)

    print("Chat failed.")

    print(valid_response_attempts)

    print("\n\nFinished one problem.\n\n")

    return False


def run_experiment(amount, b):
    total = 0
    correct = 0
    if amount == "all":
        amount = len(df)
    for i in range(len(df)-1, len(df)-1-amount, -1):
        quad = df[i]
        solved = complete_one_problem(quad, b)
        total += 1
        if solved:
            correct += 1

    return correct/total


if __name__ == '__main__':
    print(run_experiment(1, 5))  # returns 0.68 on first run
    # complete_one_problem([4, 6, 12, 13], 5)
    # print(is_valid_equation([2, 3, 5, 12], "12 + 5", "17"))