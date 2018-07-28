import time
_profile_stack = []


def profile_start(name):
    global _profile_stack
    _profile_stack.append((name, time.time()))


def profile_end(name):
    global _profile_stack
    old_name, start_time = _profile_stack.pop(0)
    if name != old_name:
        raise Exception(
            'Imbalance in calls to profile_start/end! Expected end for "{}" but got "{}" instead'.format(
                old_name, name
            )
        )
    end_time = time.time()
    print("{} took {} seconds".format(name, float("%.2f" % (end_time - start_time))))

