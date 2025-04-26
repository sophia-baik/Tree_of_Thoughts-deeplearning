import util_gameof24


# instructions and prompts
INSTRUCT = "You are a game of 24 grandmaster. We are going to take this problem step by step. At each step, you are going to pick only 2 numbers to operate on. Put the expression you choose on the last line. Don't put any of the math in latex or markdown formats."
PROMPT_1 = "Numbers: "
PROMPT_2 = "Remaining numbers: <INSERT NUMBERS>."
PROMPT_3 = "Remaining numbers: <INSERT NUMBERS>. If you  If you cannot obtain 24, put 'No' as your final token."


numbers = util_gameof24.import_data()  # load game24 data


def is_valid_equation(given_numbers, before_equals, after_equals):
     if eval(before_equals) != eval(after_equals):
            return False

        # Parse the numbers used in the expression
     number_string = ""
     for c in before_equals:
            if c.isdigit():
                number_string += c
            elif c in ['+', '-', '*', '/']:
                number_string += ','

     used_numbers = number_string.split(',')
     used_numbers = [num for num in used_numbers if num != '']

        # Check if exactly 4 numbers are used
     if len(used_numbers) != 4:
            return False

        # Check if all provided numbers are used exactly once
     temp_input = [str(num) for num in given_numbers]
     for num in used_numbers:
            if num not in temp_input:
                return False
            temp_input.remove(num)

     if temp_input != []:  # Some input numbers weren't used
            return False

        # Check if the result equals 24
     if eval(before_equals) != 24:
            return False

     return True

      # 5 3 2 4
      # 8 2 4


def complete_one_problem(quad, b):
    """
    uses ToT to solve problem with [quad] numbers
     - b is the max breadth
    ---
    this function doesn't chain the previous answers to chat
    """
    ### step 1 ###

    # ask chat prompt 1 b times
    step_one_responses = []
    valid_responses = []

    entry_number = util_gameof24.get_file_date_time()

    for i in range(b):
        response = (util_gameof24.ask_chat(
            PROMPT_1 + quad, util_gameof24.MODEL, INSTRUCT))
        step_one_responses.append(response)

        # This is quite hard coded; we'll be in trouble if Chat deviates from instructions
        last_line = response.split('\n')[-1]
        before_equals = last_line[0]
        after_equals = last_line[1]

        if is_valid_equation(quad, before_equals, after_equals):
            correct += 1
            valid_responses.append(response)
                 
    ### step 2 ###

    # for each b responses, ask chat using that response with 3*b responses and rank them. choose the best b
    # for each answer chat gave from step 1, sample 3 times

    
    # prompt 2

    expanded_child_responses = []
    for response in valid_responses:
        last_line = response.split('\n')[-1]
        before_equals = last_line[0]
        after_equals = last_line[1]

        for _ in range(3):  # 3 child thoughts per valid parent
            child_response = util_gameof24.ask_chat(PROMPT_2 + response, util_gameof24.MODEL, INSTRUCT)
            
            last_line = child_response.strip().split('\n')[-1]
            child_before = last_line[0]
            child_after = last_line[1]

            if is_valid_equation(quad, child_before, child_after):
                expanded_child_responses.append(child_response)
    # for each new answer, verify chat's answer is valid; if not, prune

    # run evaluator on b*3 (possibly less) expressions and pick top b
     
    ### step 3 ###

    # prompt 3 - give chat 2 numbers and see if it can make 24
    # verify chat's answer is valid


def run_experiment(amount):
    if amount == "all":
        amount = len(numbers)
    for i in range(len(numbers)-1, len(numbers)-1-amount, -1):
        complete_one_problem()


if __name__ == '__main__':
    complete_one_problem("[2,3,5,12]", 1)
