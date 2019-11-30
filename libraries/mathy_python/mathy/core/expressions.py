from typing import Union, Optional
from .tree import BinaryTreeNode, STOP
import numpy
from colr import color
import math

OOO_FUNCTION = 4
OOO_PARENS = 3
OOO_EXPONENT = 2
OOO_MULTDIV = 1
OOO_ADDSUB = 0
OOO_INVALID = -1


MathTypeKeys = {
    "empty": 0,
    "negate": 1,
    "equal": 2,
    "add": 3,
    "subtract": 4,
    "multiply": 5,
    "divide": 6,
    "power": 7,
    "term_root": 8,
    "term_connector": 9,
    # NOTE: reserved 10-14 for future expression types (such as functions)
    "constant": 15,
    "constant_0": 16,
    "constant_1": 17,
    "constant_2": 18,
    "constant_3": 19,
    "constant_4": 20,
    "constant_5": 21,
    "constant_6": 22,
    "constant_7": 23,
    "constant_8": 24,
    "constant_9": 25,
    "variable": 26,
    "variable_a": 27,
    "variable_b": 28,
    "variable_c": 29,
    "variable_d": 30,
    "variable_e": 31,
    "variable_f": 32,
    "variable_g": 33,
    "variable_h": 34,
    "variable_i": 35,
    "variable_j": 36,
    "variable_k": 37,
    "variable_l": 38,
    "variable_m": 39,
    "variable_n": 40,
    "variable_o": 41,
    "variable_p": 42,
    "variable_q": 43,
    "variable_r": 44,
    "variable_s": 45,
    "variable_t": 46,
    "variable_u": 47,
    "variable_v": 48,
    "variable_w": 49,
    "variable_x": 50,
    "variable_y": 51,
    "variable_z": 52,
}


# The maximum value in type keys (for one-hot encoding)
MathTypeKeysMax = max(MathTypeKeys.values()) + 1


class MathExpression(BinaryTreeNode):
    """Math tree node with helpers for manipulating expressions.
    
    `mathy:x+y=z`
    """

    @property
    def raw(self):
        """raw text representation of the expression."""
        return str(self)

    @property
    def type_id(self):
        raise NotImplementedError("must be implemented in subclass")

    @property
    def terminal_text(self):
        """Text output of this node that includes terminal color codes that
        highlight which nodes have been changed in this tree as a result of
        a transformation."""

        def visit_fn(node, depth, data):
            node._rendering_change = data

        self.visit_inorder(visit_fn, data=True)
        result = str(self)
        self.visit_inorder(visit_fn, data=False)
        return result

    @property
    def color(self):
        """Color to use for this node when rendering it as changed with `.terminal_text`"""
        return "green"

    _rendering_change: bool

    def __init__(self, id=None, left=None, right=None, parent=None):
        super().__init__(left, right, parent, id)
        self._rendering_change = None
        self._changed = False
        self.classes = [self.id]
        self.cloned_node = None
        self.cloned_target = None

    def evaluate(self, context=None):
        """Evaluate the expression, resolving all variables to constant values"""
        return 0.0

    def set_changed(self):
        """Mark this node as having been changed by the application of a Rule"""
        self._changed = True

    def all_changed(self):
        """Mark this node and all of its children as changed"""

        def visit_fn(node, depth, data):
            node.set_changed()

        self.visit_inorder(visit_fn)

    def differentiate(self, by_variable):
        """Differentiate the expression by a given variable"""
        raise Exception("cannot differentiate an abstract MathExpression node")

    def with_color(self, text: str, style="bright") -> str:
        """Render a string that is colored if something has changed"""
        if self._rendering_change is True and self._changed is True:
            return color(text, fore=self.color, style=style)
        return text

    def add_class(self, classes):
        """Associate a class name with an expression. This class name will be
        attached to nodes when the expression is converted to a capable output
        format.  See #MathExpression.to_math_ml_fragment"""
        if type(classes) == str:
            classes = [classes]
        self.classes = list(set(self.classes).union(classes))
        return self

    def count_nodes(self):
        """Return the number of nodes in this expression"""
        count = 0

        def visit_fn(node, depth, data):
            nonlocal count
            count = count + 1

        self.visit_inorder(visit_fn)
        return count

    def to_list(self, visit="postorder"):
        """Convert this node hierarchy into a list."""
        results = []

        def visit_fn(node, depth, data):
            return results.append(node)

        if visit == "inorder":
            self.visit_inorder(visit_fn)
        elif visit == "preorder":
            self.visit_preorder(visit_fn)
        elif visit == "postorder":
            self.visit_postorder(visit_fn)
        else:
            raise ValueError(f"invalid visit order: {visit}")
        return results

    def clear_classes(self):
        """Clear all the classes currently set on the nodes in this expression."""
        results = []

        def visit_fn(node, depth, data):
            node.classes = []

        return results

    def find_type(self, instanceType):
        """Find an expression in this tree by type.

        - instanceType: The type to check for instances of
        
        Returns the found #MathExpression objects of the given type.
        """
        results = []

        def visit_fn(node, depth, data):
            if isinstance(node, instanceType):
                return results.append(node)

        self.visit_inorder(visit_fn)
        return results

    def find_id(self, id):
        """Find an expression by its unique ID.

        Returns: The found #MathExpression or `None`
        """
        result = None

        def visit_fn(node, depth, data):
            nonlocal result
            if node.id == id:
                result = node
                return STOP

        self.visit_inorder(visit_fn)
        return result

    def to_math_ml_fragment(self):
        """Convert this single node into MathML."""
        return ""

    def to_math_ml(self):
        """Convert this expression into a MathML container."""
        return "\n".join(
            [
                "<math xmlns='http:#www.w3.org/1998/Math/MathML'>",
                self.to_math_ml_fragment(),
                "</math>",
            ]
        )

    def make_ml_tag(self, tag: str, content, classes=[]) -> str:
        """Make a MathML tag for the given content while respecting the node's given
        classes.

        Params:

            - `tag` The ML tag name to create.
            - `content` The ML content to place inside of the tag.
            - `classes` An array of classes to attach to this tag.
        """
        if len(classes) == 0:
            classes = ""
        else:
            classes = " class='{}'".format(" ".join(classes))
        return "<{}{}>{}</{}>".format(tag, classes, content, tag)

    def path_to_root(self) -> str:
        """Generate a namespaced path key to from the current node to the root.
        This key can be used to identify a node inside of a tree."""
        points = []

        def path_mark(node):
            points.append(node.__class__.__name__)

        node = self
        path_mark(node)
        while node.parent:
            node = node.parent
            path_mark(node)
        return ".".join(points)

    def clone_from_root(self, node=None) -> "MathExpression":
        """Clone this node including the entire parent hierarchy that it has. This
        is useful when you want to clone a subtree and still maintain the overall
        hierarchy.

        Params:

            - `node` The node to clone.

        Returns: The cloned #MathExpression node.
        """
        node = node if node is not None else self
        self.cloned_node = None
        self.cloned_target = node.path_to_root()
        result = node.get_root().clone()
        if not self.cloned_node:
            print("While cloning root of: {}".format(node))
            print(" Which is this       : {}".format(node.get_root()))
            print("Did not set the clone: {}".format(self.cloned_node))
            raise Exception("cloning root hierarchy did not clone this node")

        result = self.cloned_node
        self.cloned_node = None
        self.cloned_target = None
        return result

    def clone(self) -> "MathExpression":
        """A specialization of the clone method that can track and report a cloned
        subtree node. See #MathExpression.clone_from_root for more details."""
        result = super().clone()
        if self.cloned_target is not None and self.path_to_root() == self.cloned_target:
            self.cloned_node = result

        return result


class UnaryExpression(MathExpression):
    """An expression that operates on one sub-expression"""

    def __init__(self, child=None, operatorOnLeft=True):
        super().__init__()
        self.child = child
        self.operatorleft = operatorOnLeft
        self.set_child(child)

    def set_child(self, child):
        if self.operatorleft:
            return self.set_left(child)
        else:
            return self.set_right(child)

    def get_child(self):
        if self.operatorleft:
            return self.left
        else:
            return self.right

    def evaluate(self, context=None):
        return self.operate(self.get_child().evaluate(context))

    def operate(self, value):
        raise Exception("Must be implemented in subclass")


# ### Negation


class NegateExpression(UnaryExpression):
    """Negate an expression, e.g. `4` becomes `-4`"""

    @property
    def type_id(self):
        return MathTypeKeys["negate"]

    @property
    def name(self):
        return "-"

    def operate(self, value):
        return -value

    def __str__(self):
        return self.with_color("-{}".format(self.get_child()))

    def to_math_ml_fragment(self):
        """Convert this single node into MathML."""
        return f"-{super().to_math_ml_fragment()}"

    def differentiate(self, by_variable):
        """
        ```
        .            f(x) = -g(x)
        .        d( f(x) ) = -( d( g(x) ) )
        ```
        """
        return NegateExpression(self.child.differentiate(by_variable))


# ### Function


class FunctionExpression(UnaryExpression):
    """A Specialized UnaryExpression that is used for functions.  The function name in
    text (used by the parser and tokenizer) is derived from the name() method on the
    class."""

    @property
    def name(self):
        return ""

    def __str__(self):
        child = self.get_child()
        output = self.name
        if child:
            output = "{}({})".format(self.name, child)
        return self.with_color(output)


# ## Binary Expressions


class BinaryExpression(MathExpression):
    """An expression that operates on two sub-expressions"""

    def __init__(self, left=None, right=None):
        super().__init__(left=left, right=right)

    def evaluate(self, context=None):
        return self.operate(self.left.evaluate(context), self.right.evaluate(context))

    @property
    def name(self):
        raise Exception("Must be implemented in subclass")

    def get_ml_name(self):
        return self.name

    def operate(self, one, two):
        raise Exception("Must be implemented in subclass")

    def get_priority(self):
        """Return a number representing the order of operations priority
        of this node.  This can be used to check if a node is `locked`
        with respect to another node, i.e. the other node must be resolved
        first during evaluation because of it's priority.
        """
        if not isinstance(self, BinaryExpression):
            priority = OOO_INVALID

        if isinstance(self, EqualExpression):
            priority = OOO_INVALID

        if isinstance(self, AddExpression) or isinstance(self, SubtractExpression):
            priority = OOO_ADDSUB

        if isinstance(self, MultiplyExpression) or isinstance(self, DivideExpression):
            priority = OOO_MULTDIV

        if isinstance(self, PowerExpression):
            priority = OOO_EXPONENT

        if isinstance(self, FunctionExpression):
            priority = OOO_FUNCTION

        return priority

    def self_parens(self) -> bool:
        """Return a boolean indicating whether this node should render itself with
        a set of enclosing parnetheses or not. This is used when serializing an
        expression, to ensure the tree maintains the proper order of operations."""
        parent = isinstance(self.parent, BinaryExpression)
        self_pri = self.get_priority()
        if not parent:
            return False

        parent_pri = self.parent.get_priority()
        if parent_pri > self_pri:
            return True

        parent_side = self.parent.get_side(self)
        # If we're a left child of a parent
        if parent_pri == self_pri:
            self_addsub = isinstance(self, (AddExpression, SubtractExpression))
            parent_addsub = isinstance(self.parent, (AddExpression, SubtractExpression))
            self_muldiv = isinstance(self, (MultiplyExpression, DivideExpression))
            parent_muldiv = isinstance(
                self.parent, (MultiplyExpression, DivideExpression)
            )
            if parent_side == "right" and self_addsub and parent_addsub:
                return True
            if parent_side == "left" and self_muldiv and parent_muldiv:
                return True
        return False

    def __str__(self):
        if self.left is None or self.right is None:
            raise ValueError(
                "{}: left/right children must both be valid".format(
                    self.__class__.__name__
                )
            )

        out = f"{self.left} {self.with_color(self.name)} {self.right}"
        return f"({out})" if self.self_parens() else out

    def to_math_ml_fragment(self):
        """Render this node as a MathML element fragment"""
        right_ml = self.right.to_math_ml_fragment()
        left_ml = self.left.to_math_ml_fragment()
        op_ml = self.make_ml_tag("mo", self.get_ml_name())
        if self.self_parens():
            return self.make_ml_tag(
                "mrow",
                "<mo>(</mo>{}{}{}<mo>)</mo>".format(left_ml, op_ml, right_ml),
                self.classes,
            )
        return self.make_ml_tag(
            "mrow", "{}{}{}".format(left_ml, op_ml, right_ml), self.classes
        )


class EqualExpression(BinaryExpression):
    """Evaluate equality of two expressions"""

    @property
    def type_id(self):
        return MathTypeKeys["equal"]

    @property
    def name(self):
        return "="

    def operate(self, one: BinaryExpression, two: BinaryExpression):
        """This is where assignment of context variables might make sense. But context
        is not present in the expression's `operate` method.

        !!! warning

            TODO: Investigate this thoroughly.
        """
        raise Exception("Unsupported operation. Euqality has no operation to perform.")


class AddExpression(BinaryExpression):
    """Add one and two"""

    @property
    def type_id(self):
        return MathTypeKeys["add"]

    @property
    def name(self):
        return "+"

    def operate(self, one, two):
        return one + two

    def differentiate(self, by_variable):
        """```
        .               f(x) = g(x) + h(x)
        .          d( f(x) ) = d( g(x) ) + d( h(x) )
        .              f'(x) = g'(x) + h'(x)
        ```"""
        return AddExpression(
            self.left.differentiate(by_variable), self.right.differentiate(by_variable)
        )


class SubtractExpression(BinaryExpression):
    """Subtract one from two"""

    @property
    def type_id(self):
        return MathTypeKeys["subtract"]

    @property
    def name(self):
        return "-"

    def operate(self, one, two):
        return one - two

    def differentiate(self, by_variable):
        """```
        .               f(x) = g(x) - h(x)
        .          d( f(x) ) = d( g(x) ) - d( h(x) )
        .              f'(x) = g'(x) - h'(x)
        ```"""
        return AddExpression(
            self.left.differentiate(by_variable), self.right.differentiate(by_variable)
        )


class MultiplyExpression(BinaryExpression):
    """Multiply one and two"""

    @property
    def type_id(self):
        return MathTypeKeys["multiply"]

    @property
    def name(self):
        return "*"

    def get_ml_name(self):
        return "&#183;"

    def operate(self, one, two):
        return one * two

    def differentiate(self, by_variable):
        """```
        .             f(x) = g(x)*h(x)
        .            f'(x) = g(x)*h'(x) + g'(x)*h(x)
        ```"""
        return AddExpression(
            MultiplyExpression(self.left, self.right.differentiate(by_variable)),
            MultiplyExpression(self.left.differentiate(by_variable), self.right),
        )

    def __str__(self):
        """Multiplication special cases constant*variable to output `4x` instead of
        `4 * x`"""
        if isinstance(self.left, ConstantExpression):
            # const * var
            one = isinstance(self.right, VariableExpression)
            # const * var^power
            two = isinstance(self.right, PowerExpression) and isinstance(
                self.right.left, VariableExpression
            )
            if one or two:
                return self.with_color("{}{}".format(self.left, self.right))

        return super().__str__()

    def to_math_ml_fragment(self):
        right_ml = self.right.to_math_ml_fragment()
        left_ml = self.left.to_math_ml_fragment()
        if isinstance(self.left, ConstantExpression):
            if isinstance(self.right, VariableExpression) or isinstance(
                self.right, PowerExpression
            ):
                return "{}{}".format(left_ml, right_ml)

        return super().to_math_ml_fragment()


class DivideExpression(BinaryExpression):
    """Divide one by two"""

    @property
    def type_id(self):
        return MathTypeKeys["divide"]

    @property
    def name(self):
        return "/"

    def get_ml_name(self):
        return "&#247;"

    def to_math_ml_fragment(self):
        left_ml = self.left.to_math_ml_fragment()
        right_ml = self.right.to_math_ml_fragment()
        return f"<mfrac><mi>{left_ml}</mi><mi>{right_ml}</mi></mfrac>"

    def operate(self, one, two):
        if two == 0:
            return float("nan")
        else:
            return one / two

    def differentiate(self, by_variable):
        """```
        .          f(x) = g(x)/h(x)
        .         f'(x) = ( g'(x)*h(x) - g(x)*h'(x) ) / ( h(x)^2 )
        ```"""
        gprimeh = MultiplyExpression(self.left.differentiate(by_variable), self.right)
        ghprime = MultiplyExpression(self.left, self.right.differentiate(by_variable))
        hsquare = PowerExpression(self.right, ConstantExpression(2))
        return DivideExpression(SubtractExpression(gprimeh, ghprime), hsquare)


class PowerExpression(BinaryExpression):
    """Raise one to the power of two"""

    @property
    def type_id(self):
        return MathTypeKeys["power"]

    @property
    def name(self):
        return "^"

    def to_math_ml_fragment(self):
        right_ml = self.right.to_math_ml_fragment()
        left_ml = self.left.to_math_ml_fragment()
        # if left is mult, enclose only right in msup
        if isinstance(self.left, MultiplyExpression):
            left_ml = self.make_ml_tag("mrow", left_ml, self.classes)

        return self.make_ml_tag("msup", "{}{}".format(left_ml, right_ml), self.classes)

    def operate(self, one, two):
        return numpy.power(one, two)

    def differentiate(self, by_variable):
        """
        !!! warn Unimplemented

            This needs to be implemented
        """
        raise Exception("Unimplemented")

    def __str__(self):
        return "{}{}{}".format(self.left, self.with_color(self.name), self.right)


class ConstantExpression(MathExpression):
    """A Constant value node, where the value is accessible as `node.value`"""

    value: Union[float, int]

    @property
    def name(self):
        return f"const({self.value})"

    @property
    def type_id(self):
        id = f"_{int(self.value % 10)}" if self.value is not None else ""
        return MathTypeKeys[f"constant{id}"]

    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def clone(self):
        result = super().clone()
        result.value = self.value
        return result

    def evaluate(self, context=None):
        return self.value

    def __str__(self):
        return self.with_color("{}".format(self.value))

    def to_json(self):
        result = super().to_json()
        result.name = self.value
        return result

    def to_math_ml_fragment(self) -> str:
        return self.make_ml_tag("mn", self.value, self.classes)


class VariableExpression(MathExpression):
    identifier: str

    @property
    def name(self):
        return f"var({self.identifier})"

    @property
    def type_id(self):
        id = f"_{self.identifier.lower()[0]}" if self.identifier is not None else ""
        return MathTypeKeys[f"variable{id}"]

    def __init__(self, identifier=None):
        super().__init__()
        self.identifier = identifier

    def clone(self):
        result = super().clone()
        result.identifier = self.identifier
        return result

    def __str__(self):
        if self.identifier is None:
            return ""
        else:
            return self.with_color("{}".format(self.identifier))

    def to_math_ml_fragment(self):
        if self.identifier is None:
            return ""
        else:
            return self.make_ml_tag("mi", self.identifier)

    def to_json(self):
        result = super().to_json()
        result.name = self.identifier
        return result

    def evaluate(self, context=None):
        if context and context[self.identifier]:
            return context[self.identifier]

        raise Exception(
            "cannot evaluate statement with None variable: {}".format(self.identifier)
        )

    def differentiate(self, by_variable):
        """
        Differentiating by this variable yields 1

        ```
        .             f(x) = x
        .        d( f(x) ) = 1 * d( x )
        .           d( x ) = 1
        .            f'(x) = 1
        ```

        Differentiating by any other variable yields 0

        ```
        .             f(x) = c
        .        d( f(x) ) = c * d( c )
        .           d( c ) = 0
        .            f'(x) = 0
        ```
        """
        if by_variable == self.identifier:
            return ConstantExpression(1)
        return ConstantExpression(0)


class AbsExpression(FunctionExpression):
    """Evaluates the absolute value of an expression."""

    @property
    def type_id(self):
        return MathTypeKeys["abs"]

    @property
    def name(self):
        return "abs"

    def operate(self, value):
        return numpy.absolute(value)

    def differentiate(self, by_variable):
        """```
        .           f(x)   = abs( g(x) )
        .        d( f(x) ) = sgn( g(x) ) * d( g(x) )
        ```"""
        return MultiplyExpression(
            SgnExpression(self.child), self.child.Differentiate(by_variable)
        )


class SgnExpression(FunctionExpression):
    @property
    def type_id(self):
        return MathTypeKeys["sgn"]

    @property
    def name(self):
        return "sgn"

    def operate(self, value):
        """Determine the sign of an value.

        Returns: -1 if negative, 1 if positive, 0 if 0"""
        if value < 0:
            return -1

        if value > 0:
            return 1

        return 0

    def differentiate(self, by_variable):
        """```
        .             f(x) = sgn( g(x) )
        .        d( f(x) ) = 0
        ```
        
        Note: in general sgn'(x) = 2δ(x) where δ(x) is the Dirac delta function."""
        return ConstantExpression(0)
