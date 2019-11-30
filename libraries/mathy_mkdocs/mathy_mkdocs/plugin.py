import re

import svgwrite

from mathy import (
    BinaryExpression,
    ExpressionParser,
    MathExpression,
    TreeLayout,
    TreeMeasurement,
    VariableExpression,
    testing,
)

parser = ExpressionParser()
layout = TreeLayout()
matcher_re = r"<code>mathy:([\d\w\^\*\+\-\=\/\.\s\(\)\[\]]*)<\/code>"
rules_matcher_re = r"`rule_tests:([a-z\_]*)`"


def to_math_ml_fragment(match):
    global parser
    match = match.group(1)
    try:
        expression: MathExpression = parser.parse(match)
        return expression.to_math_ml()
    except BaseException as error:
        return f"Failed to parse: '{match}' with error: {error}"


def render_examples_from_tests(match):
    """render a set of rule to/from examples in markdown from the rules tests"""
    rule_file_name = match.group(1)
    # If an example is longer than this, omit it for vanity.
    max_len = 32
    try:
        data = testing.get_rule_tests(rule_file_name)
        result = f"\r\n|Input|Output|Valid|"
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


def render_mathy_templates(input_text: str):
    global matcher_re
    text = re.sub(matcher_re, render_tree_from_match, input_text, flags=re.IGNORECASE)
    text = re.sub(
        rules_matcher_re, render_examples_from_tests, text, flags=re.IGNORECASE
    )
    return text


if __name__ == "__main__":
    print(render_mathy_templates("<code>mathy:4x^3 * 2x - 7</code>"))
    print(render_mathy_templates("`rule_tests:constants_simplify`"))
else:
    from mkdocs.plugins import BasePlugin

    class MathyMkDocsPlugin(BasePlugin):
        def on_page_markdown(self, markdown, **kwargs):
            return render_mathy_templates(markdown)

        def on_page_content(self, content, **kwargs):
            return render_mathy_templates(content)
