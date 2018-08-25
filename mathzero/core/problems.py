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
        for _ in range(max_terms):
            num = random.randint(1, 12)
            op = operators[random.randint(0, len(operators) - 1)]
            result = result + " {} {}".format(op, num)
        return result

    def simplify_multiple_terms(self, max_terms=4):
        operators = list("+*")
        variables = list("xyz")
        variable = variables[random.randint(0, len(variables) - 1)]
        result = "{}{}".format(random.randint(2, 10), variable)
        for _ in range(max_terms - 1):
            variable = variables[random.randint(0, len(variables) - 1)]
            num = random.randint(1, 12)
            var = variable if random.getrandbits(1) == 0 else ""
            op = operators[random.randint(0, len(operators) - 1)]
            result = result + " {} {}{}".format(op, num, var)
        return result

    def variable_multiplication(self, max_terms=4):
        variables = list("xyz")
        variable = variables[random.randint(0, len(variables) - 1)]
        constant = random.randint(1, 3)
        exp = '^{}'.format(constant) if constant > 1 else ''
        result = "{}{}".format(variable, exp)
        for _ in range(max_terms - 1):
            constant = random.randint(1, 12)
            exp = '^{}'.format(constant) if constant > 1 else ''
            result = result + " * {}{}".format(variable, exp)
        return result

    def basic_combine_like_terms(self):
        """Generate a two term addition problem of the form [n][var] + [n][var]"""
        variables = list("xyz")
        variable = variables[random.randint(0, len(variables) - 1)]
        coefficient_one = random.randint(1, 12)
        coefficient_two = random.randint(1, 12)
        result = "{}{} + {}{}".format(
            coefficient_one, variable, coefficient_two, variable
        )
        return result

    def combine_like_terms(self, min_terms=2, max_terms=4):
        """Generate a (n) term addition problem of the form [n][var] + [n][var]"""
        # TODO: add exponents=bool param and optionally generate terms with exps
        variables = list("xyz")
        num_terms = random.randint(min_terms, max_terms)
        variable = variables[random.randint(0, len(variables) - 1)]
        result = "{}{}".format(random.randint(2, 10), variable)
        for _ in range(num_terms - 1):
            num = random.randint(1, 12)
            result = result + " + {}{}".format(num, variable)
        return result
