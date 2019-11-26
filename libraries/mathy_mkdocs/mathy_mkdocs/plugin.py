import re

import svgwrite

from mathy import (
    BinaryExpression,
    ExpressionParser,
    MathExpression,
    TreeLayout,
    TreeMeasurement,
    VariableExpression,
)

parser = ExpressionParser()
layout = TreeLayout()
matcher_re = r"\{\{\s*mathy:([^\}]*)\s*\}\}"


def to_math_ml(match):
    global parser
    match = match.group(1)
    try:
        expression: MathExpression = parser.parse(match)
        return expression.to_math_ml_element()
    except BaseException as error:
        return f"Failed to parse: '{match}' with error: {error}"
        pass
    return match + "--done by mathy_fn"


def replace_match(match):
    global parser, layout
    input_text = match.group(1)
    try:
        expression: MathExpression = parser.parse(input_text)
        m: TreeMeasurement = layout.layout(expression, 70, 70)
        padding = 25
        offset_x = padding + abs(m.minX)
        offset_y = padding + abs(m.minY)
        text_height = 6
        char_width = 8

        tree = svgwrite.Drawing(size=(200, 200))

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

            # html.push(
            #     drawText(
            #         node.x + offsetX, node.y + offsetY, value, options.nodeSize, color
            #     )
            # )

            # pass

        expression.visit_postorder(node_visit)

        return svgwrite.utils.pretty_xml(tree.tostring(), indent=2)
    except BaseException as error:
        return f"Failed to parse: '{input_text}' with error: {error}"


def render_mathy_templates(input_text: str):
    global matcher_re
    text = re.sub(matcher_re, replace_match, input_text, flags=re.IGNORECASE)
    return text


if __name__ == "__main__":
    print(render_mathy_templates("{{mathy:4x + 2 }}"))
else:
    from mkdocs.plugins import BasePlugin

    class MathyMkDocsPlugin(BasePlugin):
        def on_page_markdown(self, markdown, **kwargs):
            return render_mathy_templates(markdown)
