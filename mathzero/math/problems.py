import random


class ProblemGenerator:
    # https://codereview.stackexchange.com/questions/46226/utility-function-to-split-a-number-into-n-parts-in-the-given-ratio
    def new_reduce_ratio(self, load_list=None, total_num=10, min_num=2):
        if load_list is None:
            load_list = [20, 40, 40]

        output = [min_num for _ in load_list]
        total_num -= sum(output)
        if total_num < 0:
            raise Exception("Could not satisfy min_num")
        elif total_num == 0:
            return output

        nloads = len(load_list)
        for ii in range(nloads):
            load_sum = float(sum(load_list))
            load = load_list.pop(0)
            value = int(round(total_num * load / load_sum))
            output[ii] += value
            total_num -= value
        return output

    def sum_and_single_variable(self, sum=15, variable=None):
        variables = list("xyz")
        if variable is None:
            variable = variables[random.randint(0, len(variables) - 1)]
        numbers = self.new_reduce_ratio(total_num=sum)
        nums = [str(num) for num in numbers[:3]]
        nums.append(variable)
        random.shuffle(nums)
        result = " + ".join(nums)
        return result
