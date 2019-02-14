import random
import sys

MODE_ARITHMETIC = 0
MODE_SOLVE_FOR_VARIABLE = 1
MODE_SIMPLIFY_POLYNOMIAL = 2


class ProblemGenerator:
    def __init__(self):
        self.max_int = 4096
        self.variables = list("xyz")
        self.operators = list("+*")
        self.problem_types = [
            MODE_ARITHMETIC,
            MODE_SIMPLIFY_POLYNOMIAL,
            MODE_SOLVE_FOR_VARIABLE,
        ]

    def random_problem(self, from_types=None):
        if from_types is None:
            from_types = self.problem_types
        # Pick a random problem type (TODO: is this wise?)
        type = from_types[random.randint(0, len(from_types) - 1)]
        complexity = random.randint(3, 5)
        if type == MODE_ARITHMETIC:
            problem = self.arithmetic_expression(terms=complexity)
        elif type == MODE_SIMPLIFY_POLYNOMIAL:
            problem = self.simplify_multiple_terms(terms=complexity)
        elif type == MODE_SOLVE_FOR_VARIABLE:
            problem = self.solve_for_variable(terms=complexity)
        return problem, type, complexity

    def random_var(self, exclude_var=None):
        """Generate a random variable from with an optional variable to exclude"""
        var = self.variables[random.randint(0, len(self.variables) - 1)]
        while var == exclude_var:
            var = self.variables[random.randint(0, len(self.variables) - 1)]
        return var

    # From stackoverflow: https://bit.ly/2zxeQGf
    def split_into_parts(self, total_num, min_num, load_list=None):
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
        if variable is None:
            variable = self.random_var()
        numbers = self.split_into_parts(sum, 3)
        nums = [str(num) for num in numbers][: max_terms - 1]
        nums.append(variable)
        random.shuffle(nums)
        result = " + ".join(nums)
        return result

    def binary_operations_no_variables(self, sum=None, terms=3):
        if sum is None:
            sum = random.randint(terms * 5, terms * 20)
        operators = list("+-*")
        result = "{}".format(random.randint(2, 10))
        for _ in range(terms):
            num = random.randint(1, 12)
            op = operators[random.randint(0, len(operators) - 1)]
            result = result + " {} {}".format(op, num)
        return result

    def simplify_multiple_terms(self, terms=4):
        operators = list("+*")
        variables = list("xyz")
        variable = variables[random.randint(0, len(variables) - 1)]
        # Guarantee at least one set of like terms
        result = "{}{}".format(random.randint(2, self.max_int), variable)
        suffix = " + {}{}".format(random.randint(2, self.max_int), variable)
        for _ in range(terms - 2):
            variable = variables[random.randint(0, len(variables) - 1)]
            num = random.randint(1, self.max_int)
            var = variable if random.getrandbits(1) == 0 else ""
            op = operators[random.randint(0, len(operators) - 1)]
            result = result + " {} {}{}".format(op, num, var)
        return result + suffix

    def arithmetic_expression(self, terms=4):
        operators = list("+*/-")
        result = "{}".format(random.randint(1, 10))
        for _ in range(terms - 1):
            num = random.randint(1, 12)
            op = operators[random.randint(0, len(operators) - 1)]
            result = result + " {} {}".format(op, num)
        return result

    def variable_multiplication(self, max_terms=4):
        variables = list("xyz")
        variable = variables[random.randint(0, len(variables) - 1)]
        constant = random.randint(1, 3)
        exp = "^{}".format(constant) if constant > 1 else ""
        result = "{}{}".format(variable, exp)
        for _ in range(max_terms - 1):
            constant = random.randint(1, 12)
            exp = "^{}".format(constant) if constant > 1 else ""
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
        result = "{}{}".format(random.randint(2, max_int), variable)
        for _ in range(num_terms - 1):
            num = random.randint(0, self.max_int)
            result = result + " + {}{}".format(num, variable)
        return result

    def solve_for_variable(self, terms=4):
        """Generate a solve for x type problem, e.g. `4x + 2 = 8x`"""
        variable = self.random_var()
        # Guarantee at least one set of like terms
        result = "{}{} = {}".format(
            random.randint(2, self.max_int), variable, random.randint(2, self.max_int)
        )
        suffix = " + {}{}".format(random.randint(2, self.max_int), variable)
        for _ in range(terms - 3):
            num = random.randint(1, self.max_int)
            op = self.operators[random.randint(0, len(self.operators) - 1)]
            var = variable if random.getrandbits(1) == 0 else ""
            result = result + " {} {}{}".format(op, num, var)
        return result + suffix
