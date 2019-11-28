Mathy uses machine learning to choose which [actions](/cas/rules/overview) to apply to which nodes in [environments](/envs/overview).

Solving complex math problems requires combinations of low-level actions that lead to higher-level ones. Humans are smart, so we often do multiple steps at once in our heads and consider it to be just one. For example, we think of of the transformation `4x + 2x => 6x` a single action but it actually requires a factoring operation and an artithmetic operation to accomplish.

When it comes to solving math problems, there are at least two broad ways to approach combining low-level rules into high-level actions. Some CAS systems write a ton of custom lower-level rule compositions to explicitly capture the higher-level actions in a usable form. This is extremely effective and can yield systems that are able to solve many different types of math problems with confidence, but it comes at the cost of requiring expert knowledge to craft all the specific problem-set rules. Mathy uses Machine Learning (ML) to pick combinations of low-level rules, and relies on the ML model to put them together to form higher-level actions. This has the benefit of not necessarily requiring expert knowledge, but adds the complexity of crafting an ML model that can pick reasonable sets of actions for many types of problems.


