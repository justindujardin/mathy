import re

import svgwrite
from typing import List

from mathy import (
    BinaryExpression,
    ExpressionParser,
    MathExpression,
    MathyEnvState,
    MathyObservation,
    Token,
    Tokenizer,
    TreeLayout,
    TreeMeasurement,
    VariableExpression,
    testing,
)

tokenizer = Tokenizer()
parser = ExpressionParser()

expression_re = r"<code>([a-z\_]*):([\d\w\^\*\+\-\=\/\.\s\(\)\[\]]*)<\/code>"
rules_matcher_re = r"`rule_tests:([a-z\_]*)`"
snippet_matcher_re = r"```[pP]ython[\n]+{!\.(\/snippets\/[a-z\_\/]+).py!}[\n]+```"
# Add animations? http://zulko.github.io/blog/2014/09/20/vector-animations-with-python/

# TODO: add links to code highlight blocks next to clipboard
link_template = "https://colab.research.google.com/github/justindujardin/mathy/blob/master/libraries/website/docs{}.ipynb"  # noqa


def to_math_ml_fragment(match):
    global parser
    match = match.group(1)
    try:
        expression: MathExpression = parser.parse(match)
        return expression.to_math_ml()
    except BaseException as error:
        return f"Failed to parse: '{match}' with error: {error}"


rules_note = """!!! info

    All the examples shown below are drawn from the mathy test suite
    that verifies the expected input/output combinations for rule
    transformations."""


def render_examples_from_tests(match):
    """render a set of rule to/from examples in markdown from the rules tests"""
    rule_file_name = match.group(1)
    # If an example is longer than this, omit it for vanity.
    max_len = 32
    try:
        data = testing.get_rule_tests(rule_file_name)
        result = f"\r\n\r\n{rules_note}\r\n"
        result += f"\r\n|Input|Output|Valid|"
        result += f"\r\n|---|---|:---:|"
        valid = "\u2714"
        for v in data["valid"]:
            in_text = v["input"]
            out_text = v["output"]
            if len(in_text) > max_len or len(out_text) > max_len:
                continue

            result += f"\r\n|{in_text}|{out_text}|{valid}|"
        valid = "---"
        for v in data["invalid"]:
            in_text = v["input"]
            out_text = "---"
            if len(in_text) > max_len or len(out_text) > max_len:
                continue

            result += f"\r\n|{in_text}|{out_text}|{valid}|"

        return result
    except ValueError:
        return f"Rule file not found: __{rule_file_name}.json__"


def render_tree_from_text(input_text: str):
    global parser
    layout = TreeLayout()
    try:
        expression: MathExpression = parser.parse(input_text)
        measure: TreeMeasurement = layout.layout(expression, 50, 50)
        padding = 25

        double_padding = padding * 2
        width = measure.maxX - measure.minX + double_padding
        height = measure.maxY - measure.minY + double_padding
        offset_x = padding + abs(measure.minX)
        offset_y = padding + abs(measure.minY)
        text_height = 6
        char_width = 8

        view_x = measure.minX - padding
        view_y = measure.minY - padding
        view_w = abs(measure.maxX - view_x) + double_padding
        view_h = abs(measure.maxY - view_y) + double_padding

        tree = svgwrite.Drawing(size=(width, height))
        tree.viewbox(minx=0, miny=0, width=view_w, height=view_h)

        def node_visit(node: MathExpression, depth, data):
            color = svgwrite.rgb(180, 200, 255)
            value = str(node)
            if isinstance(node, BinaryExpression):
                color = svgwrite.rgb(230, 230, 230)
                value = node.name
            elif isinstance(node, VariableExpression):
                color = svgwrite.rgb(150, 250, 150)
            if node == node.get_root():
                color = svgwrite.rgb(250, 220, 200)

            if node.parent:
                tree.add(
                    tree.line(
                        (node.x + offset_x, node.y + offset_y),
                        (node.parent.x + offset_x, node.parent.y + offset_y),
                        stroke="#aaa",
                        stroke_width=4,
                    )
                )

            tree.add(
                tree.circle(
                    center=(node.x + offset_x, node.y + offset_y), r=20, fill=color
                )
            )

            text_x = -(char_width * len(value) // 2) + node.x + offset_x
            text_y = text_height + node.y + offset_y
            tree.add(
                tree.text(
                    value, insert=(text_x, text_y), fill=svgwrite.rgb(25, 25, 25),
                )
            )

        expression.visit_postorder(node_visit)

        return svgwrite.utils.pretty_xml(tree.tostring(), indent=2)
    except BaseException as error:
        return f"Failed to parse: '{input_text}' with error: {error}"


BOX_SIZE = 48
BORDER_WIDTH = 2


def box_with_char(
    drawing: svgwrite.Drawing,
    text: str,
    x=0,
    y=0,
    width=BOX_SIZE,
    height=BOX_SIZE,
    border_width=BORDER_WIDTH,
    fill="#fff",
    border="#888",
    char_width=4,
    char_height=12,
):
    """Render a box with a single character inside of it"""
    drawing.add(
        drawing.rect(
            insert=(x, y),
            size=(width, height),
            fill=fill,
            stroke=border,
            stroke_width=border_width,
        )
    )
    text_x = x - char_width * len(str(text)) + width // 2
    text_y = y + height // 2 + char_height // 2
    drawing.add(
        drawing.text(text, insert=(text_x, text_y), fill=svgwrite.rgb(50, 50, 50))
    )


def render_features_from_text(input_text: str):
    global parser
    try:
        expression: MathExpression = parser.parse(input_text)
        state = MathyEnvState(problem=input_text)
        observation: MathyObservation = state.to_observation(hash_type=[13, 37])

        length = len(observation.nodes)
        types = observation.nodes
        values = observation.values
        nodes = expression.to_list()
        chars = [n.name for n in nodes]
        assert len(types) == len(values) == len(chars)

        view_x = 0
        view_y = 0
        view_w = BOX_SIZE * length
        view_h = BOX_SIZE * 3 + BORDER_WIDTH * 2

        tree = svgwrite.Drawing(size=(view_w, view_h))
        tree.viewbox(minx=view_x, miny=view_y, width=view_w, height=view_h)

        curr_x = BORDER_WIDTH
        for n, t, v, c in zip(nodes, types, values, chars):

            color = svgwrite.rgb(180, 200, 255)
            if isinstance(n, BinaryExpression):
                color = svgwrite.rgb(230, 230, 230)
            elif isinstance(n, VariableExpression):
                color = svgwrite.rgb(150, 250, 150)
            if n == n.get_root():
                color = svgwrite.rgb(250, 220, 200)

            box_with_char(tree, c, x=curr_x, y=BORDER_WIDTH, fill=color)
            box_with_char(tree, v, x=curr_x, y=BOX_SIZE + BORDER_WIDTH)
            box_with_char(tree, t, x=curr_x, y=BOX_SIZE * 2 + BORDER_WIDTH)
            curr_x += BOX_SIZE - (BORDER_WIDTH)

        return svgwrite.utils.pretty_xml(tree.tostring(), indent=2)
    except BaseException as error:
        return f"Failed to parse: '{input_text}' with error: {error}"


def render_types_from_text(input_text: str, visit_order: str):
    global parser
    try:
        expression: MathExpression = parser.parse(input_text)
        nodes = expression.to_list(visit_order)
        length = len(nodes)
        chars = [n.name for n in nodes]
        types = [n.type_id for n in nodes]
        assert len(types) == len(chars)

        view_x = 0
        view_y = 0
        view_w = BOX_SIZE * length
        view_h = BOX_SIZE * 2 + BORDER_WIDTH * 2

        tree = svgwrite.Drawing(size=(view_w, view_h))
        tree.viewbox(minx=view_x, miny=view_y, width=view_w, height=view_h)

        curr_x = BORDER_WIDTH
        for n, t, c in zip(nodes, types, chars):

            color = svgwrite.rgb(180, 200, 255)
            if isinstance(n, BinaryExpression):
                color = svgwrite.rgb(230, 230, 230)
            elif isinstance(n, VariableExpression):
                color = svgwrite.rgb(150, 250, 150)
            if n == n.get_root():
                color = svgwrite.rgb(250, 220, 200)

            box_with_char(tree, c, x=curr_x, y=BORDER_WIDTH, fill=color)
            box_with_char(tree, t, x=curr_x, y=BOX_SIZE + BORDER_WIDTH)
            curr_x += BOX_SIZE - (BORDER_WIDTH)

        return svgwrite.utils.pretty_xml(tree.tostring(), indent=2)
    except BaseException as error:
        return f"Failed to parse: '{input_text}' with error: {error}"


def render_tokens_from_text(input_text: str):
    global tokenizer
    try:
        tokens: List[Token] = tokenizer.tokenize(input_text)
        length = len(tokens)
        values = [t.value for t in tokens]
        types = [t.type for t in tokens]
        assert len(types) == len(values)

        box_size = 64
        view_x = 0
        view_y = 0
        view_w = box_size * length
        view_h = box_size * 2 + BORDER_WIDTH * 2

        tree = svgwrite.Drawing(size=(view_w, view_h))
        tree.viewbox(minx=view_x, miny=view_y, width=view_w, height=view_h)

        curr_x = BORDER_WIDTH
        for t, v in zip(types, values):
            color = svgwrite.rgb(180, 200, 255)
            box_with_char(
                tree,
                v,
                x=curr_x,
                char_width=6,
                y=BORDER_WIDTH,
                width=box_size,
                height=box_size,
                fill=color,
            )
            box_with_char(
                tree,
                t,
                char_width=6,
                x=curr_x,
                y=box_size + BORDER_WIDTH,
                width=box_size,
                height=box_size,
            )
            curr_x += box_size - (BORDER_WIDTH)

        return svgwrite.utils.pretty_xml(tree.tostring(), indent=2)
    except BaseException as error:
        return f"Failed to parse: '{input_text}' with error: {error}"


def render_colab_link_to_snippet(match):
    global link_template
    input_text = match.group(1)
    url = link_template.format(input_text)
    target = "{target=_blank}"
    return f"""[![Open Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url}){target}
{match.group(0)}"""


def render_markdown(input_text: str):
    global rules_matcher_re, snippet_matcher_re
    input_text = re.sub(
        snippet_matcher_re, render_colab_link_to_snippet, input_text, flags=re.MULTILINE
    )
    input_text = re.sub(
        rules_matcher_re, render_examples_from_tests, input_text, flags=re.IGNORECASE
    )
    return input_text


def render_code_match(match):
    command = match.group(1)
    input_text = match.group(2)
    if command == "mathy":
        return render_tree_from_text(input_text)
    elif command == "features":
        return render_features_from_text(input_text)
    elif command == "types_pre":
        return render_types_from_text(input_text, "preorder")
    elif command == "types_post":
        return render_types_from_text(input_text, "postorder")
    elif command == "types_in":
        return render_types_from_text(input_text, "inorder")
    elif command == "tokens":
        return render_tokens_from_text(input_text)
    return input_text


def render_html(input_text: str):
    global expression_re
    text = re.sub(expression_re, render_code_match, input_text, flags=re.IGNORECASE)
    return text


if __name__ == "__main__":
    res = render_html("<code>features:4x^3 * 2x - 7</code>")
    # with open("./features.svg", "w") as f:
    #     f.write(res)
    print(res)
    print(render_html("<code>mathy:4x^3 * 2x - 7</code>"))
    print(render_markdown("`rule_tests:constants_simplify`"))
    print(
        render_markdown(
            """### Extensions

Because algebra problems are only a tiny sliver of what can be represented using math expression trees, Mathy has customization points to allow altering or creating entirely new environments with little effort.

#### New Problems

Generating a new problem type while subclassing a base environment is probably the simplest way to create a custom challenge for the agent.

You can inherit from a base environment like [Poly Simplify](/envs/poly_simplify) which has win-conditions that require all the like-terms to be gone from an expression, and all complex terms be simplified. From there you can provide any valid input expression:

```Python
{!./snippets/envs/custom_problem_text.py!}
```

#### New Actions

Build your own tree transformation actions and use them with the built-in agents:

```Python
{!./snippets/envs/custom_actions.py!}
```"""
        )
    )
else:
    from mkdocs.plugins import BasePlugin

    class MathyMkDocsPlugin(BasePlugin):
        def on_page_markdown(self, markdown, **kwargs):
            return render_markdown(markdown)

        def on_page_content(self, content, **kwargs):
            return render_html(content)
