import re

import svgwrite

from mathy import (
    BinaryExpression,
    ExpressionParser,
    MathExpression,
    MathyEnvState,
    MathyObservation,
    TreeLayout,
    TreeMeasurement,
    VariableExpression,
    testing,
)

parser = ExpressionParser()
layout = TreeLayout()
matcher_re = r"<code>mathy:([\d\w\^\*\+\-\=\/\.\s\(\)\[\]]*)<\/code>"
features_re = r"<code>features:([\d\w\^\*\+\-\=\/\.\s\(\)\[\]]*)<\/code>"
# TODO: add tokenizer visualization svg
tokens_re = r"<code>tokens:([\d\w\^\*\+\-\=\/\.\s\(\)\[\]]*)<\/code>"
rules_matcher_re = r"`rule_tests:([a-z\_]*)`"
# Add animations? http://zulko.github.io/blog/2014/09/20/vector-animations-with-python/


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


def render_tree_from_match(match):
    global parser, layout
    input_text = match.group(1)
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


def render_features_from_match(match):
    global parser, layout
    input_text = match.group(1)
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

        box_size = 48
        char_width = 4
        char_height = 12
        border_width = 2

        view_x = 0
        view_y = 0
        view_w = box_size * length
        view_h = box_size * 3 + border_width * 2

        tree = svgwrite.Drawing(size=(view_w, view_h))
        tree.viewbox(minx=view_x, miny=view_y, width=view_w, height=view_h)

        def box_with_char(text, x, y, fill="#fff", border="#888"):
            tree.add(
                tree.rect(
                    insert=(x, y),
                    size=(box_size, box_size),
                    fill=fill,
                    stroke=border,
                    stroke_width=border_width,
                )
            )
            text_x = x - char_width * len(str(text)) + box_size // 2
            text_y = y + box_size // 2 + char_height // 2
            tree.add(
                tree.text(text, insert=(text_x, text_y), fill=svgwrite.rgb(50, 50, 50))
            )

        curr_x = border_width
        for n, t, v, c in zip(nodes, types, values, chars):

            color = svgwrite.rgb(180, 200, 255)
            if isinstance(n, BinaryExpression):
                color = svgwrite.rgb(230, 230, 230)
            elif isinstance(n, VariableExpression):
                color = svgwrite.rgb(150, 250, 150)
            if n == n.get_root():
                color = svgwrite.rgb(250, 220, 200)

            box_with_char(c, curr_x, border_width, fill=color)
            box_with_char(v, curr_x, box_size + border_width)
            box_with_char(t, curr_x, box_size * 2 + border_width)
            curr_x += box_size - (border_width)

        return svgwrite.utils.pretty_xml(tree.tostring(), indent=2)
    except BaseException as error:
        return f"Failed to parse: '{input_text}' with error: {error}"


def render_markdown(input_text: str):
    global rules_matcher_re
    text = re.sub(
        rules_matcher_re, render_examples_from_tests, input_text, flags=re.IGNORECASE
    )
    return text


def render_html(input_text: str):
    global matcher_re
    text = re.sub(matcher_re, render_tree_from_match, input_text, flags=re.IGNORECASE)
    text = re.sub(features_re, render_features_from_match, text, flags=re.IGNORECASE)
    return text


if __name__ == "__main__":
    res = render_html("<code>features:4x^3 * 2x - 7</code>")
    # with open("./features.svg", "w") as f:
    #     f.write(res)
    print(res)
    print(render_html("<code>mathy:4x^3 * 2x - 7</code>"))
    print(render_markdown("`rule_tests:constants_simplify`"))
else:
    from mkdocs.plugins import BasePlugin

    class MathyMkDocsPlugin(BasePlugin):
        def on_page_markdown(self, markdown, **kwargs):
            return render_markdown(markdown)

        def on_page_content(self, content, **kwargs):
            return render_html(content)
