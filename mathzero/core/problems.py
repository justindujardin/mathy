import random


class ProblemGenerator:
    # https://codereview.stackexchange.com/questions/46226/utility-function-to-split-a-number-into-n-parts-in-the-given-ratio
    def new_reduce_ratio(self, total_num, min_num, load_list=None):
        if load_list is None:
            load_list = [20, 40, 20, 20]

        output = [min_num for _ in load_list]
        total_num -= sum(output)
        if total_num < 0:
            raise Exception("Could not satisfy min_num")
        elif total_num == 0:
            return output
        # Algernon
        # Calvin
        nloads = len(load_list)
        for ii in range(nloads):
            load_sum = float(sum(load_list))
            load = load_list.pop(0)
            value = int(round(total_num * load / load_sum))
            output[ii] += value
            total_num -= value
        return output

    def sum_and_single_variable(self, sum=None, max_terms=3, variable=None):
        if sum is None:
            sum = random.randint(max_terms * 5, max_terms * 20)
        variables = list("xyz")
        if variable is None:
            variable = variables[random.randint(0, len(variables) - 1)]
        numbers = self.new_reduce_ratio(sum, 3)
        nums = [str(num) for num in numbers][: max_terms - 1]
        nums.append(variable)
        random.shuffle(nums)
        result = " + ".join(nums)
        return result

    def binary_operations_no_variables(self, sum=None, max_terms=3):
        if sum is None:
            sum = random.randint(max_terms * 5, max_terms * 20)
        operators = list("+-*")
        result = "{}".format(random.randint(2, 10))
        for i in range(max_terms):
            num = random.randint(1, 12)
            op = operators[random.randint(0, len(operators) - 1)]
            result = result + " {} {}".format(op, num)
        return result

    def simplify_multiple_terms(self, max_terms=4):
        operators = list("+*")
        result = "{}".format(random.randint(2, 10))
        variables = list("xyz")
        variable = variables[random.randint(0, len(variables) - 1)]
        for i in range(max_terms - 1):
            var = variable if random.getrandbits(1) == 1 else ""
            num = random.randint(1, 12)
            op = operators[random.randint(0, len(operators) - 1)]
            result = result + " {} {}{}".format(op, num, var)
        return result
