import math
from typing import Dict, List, Optional, Tuple, Type, Union, cast, TypeVar

import numpy as np
from colr import color

from .tree import STOP, BinaryTreeNode

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
    "constant": 10,
    "sgn": 11,
    "abs": 12,
    # NOTE: reserved 13-25 for future expression types (such as functions)
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

NodeType = TypeVar("NodeType", bound="MathExpression")


class MathExpression(BinaryTreeNode):
    """Math tree node with helpers for manipulating expressions.
    
    `mathy:x+y=z`
    """

    left: Optional["MathExpression"]
    right: Optional["MathExpression"]
    parent: Optional["MathExpression"]

    @property
    def raw(self) -> str:
        """raw text representation of the expression."""
        return str(self)

    @property
    def type_id(self) -> int:
        raise NotImplementedError("must be implemented in subclass")

    @property
    def terminal_text(self) -> str:
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
    def color(self) -> str:
        """Color to use for this node when rendering it as changed with `.terminal_text`"""
        return "green"

    _rendering_change: bool
    _changed: bool
    classes: List[str]
    cloned_node: Optional["MathExpression"]
    cloned_target: str

    def __init__(
        self,
        id: str = None,
        left: "MathExpression" = None,
        right: "MathExpression" = None,
        parent: "MathExpression" = None,
    ):
        super().__init__(left, right, parent, id)
        self._rendering_change = False
        self._changed = False
        self.classes = [self.id]
        self.cloned_node = None
        self.cloned_target = ""

    def evaluate(
        self, context: Dict[str, Union[float, int]] = None
    ) -> Union[float, int]:
        """Evaluate the expression, resolving all variables to constant values"""
        raise NotImplementedError("must be implemented in subclass")

    def set_changed(self) -> None:
        """Mark this node as having been changed by the application of a Rule"""
        self._changed = True

    def all_changed(self) -> None:
        """Mark this node and all of its children as changed"""

        def visit_fn(node, depth, data):
            node.set_changed()

        self.visit_inorder(visit_fn)

    def with_color(self, text: str, style="bright") -> str:
        """Render a string that is colored if something has changed"""
        if self._rendering_change is True and self._changed is True:
            return color(text, fore=self.color, style=style)
        return text

    def add_class(self, classes) -> "MathExpression":
        """Associate a class name with an expression. This class name will be
        attached to nodes when the expression is converted to a capable output
        format.  

        See #MathExpression.to_math_ml_fragment"""
        if type(classes) == str:
            classes = [classes]
        self.classes = list(set(self.classes).union(classes))
        return self

    def to_list(self, visit: str = "preorder") -> List["MathExpression"]:
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

    def clear_classes(self) -> None:
        """Clear all the classes currently set on the nodes in this expression."""

        def visit_fn(node, depth, data):
            node.classes = []

        self.visit_inorder(visit_fn)

    def find_type(self, instanceType: Type[NodeType]) -> List[NodeType]:
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

    def find_id(self, id: str) -> Optional["MathExpression"]:
        """Find an expression by its unique ID.

        Returns: The found #MathExpression or `None`
        """
        result: Optional[MathExpression] = None

        def visit_fn(node, depth, data):
            nonlocal result
            if node.id == id:
                result = node
                return STOP

        self.visit_inorder(visit_fn)
        return result

    def to_math_ml_fragment(self) -> str:
        """Convert this single node into MathML."""
        return ""

    def to_math_ml(self) -> str:
        """Convert this expression into a MathML container."""
        return "\n".join(
            [
                "<math xmlns='http:#www.w3.org/1998/Math/MathML'>",
                self.to_math_ml_fragment(),
                "</math>",
            ]
        )

    def make_ml_tag(self, tag: str, content: str, classes: List[str] = []) -> str:
        """Make a MathML tag for the given content while respecting the node's given
        classes.

        # Arguments
        tag (str): The ML tag name to create.
        content (str): The ML content to place inside of the tag.
        classes (List[str]) An array of classes to attach to this tag.

        # Returns
        (str): A MathML element with the given tag, content, and classes
        """
        classes_attr = ""
        if len(classes) > 0:
            classes_attr = f' class="{" ".join(classes)}"'
        return f"<{tag}{classes_attr}>{content}</{tag}>"

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

    def clone_from_root(self, node: "MathExpression" = None) -> "MathExpression":
        """Clone this node including the entire parent hierarchy that it has. This
        is useful when you want to clone a subtree and still maintain the overall
        hierarchy.

        # Arguments
        node (MathExpression): The node to clone.

        # Returns
        (MathExpression): The cloned node.
        """
        node = node if node is not None else self
        self.cloned_node = None
        self.cloned_target = node.path_to_root()
        result = node.get_root().clone()
        if not self.cloned_node:  # pragma: nocover
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
        subtree node. 
        
        See #MathExpression.clone_from_root for more details."""
        result = super().clone()
        if self.cloned_target is not None and self.path_to_root() == self.cloned_target:
            self.cloned_node = result

        return result


class UnaryExpression(MathExpression):
    """An expression that operates on one sub-expression"""

    def __init__(self, child: MathExpression = None, child_on_left: bool = True):
        super().__init__()
        self.child = child
        self.left_child = child_on_left
        self.set_child(child)

    def set_child(self, child: MathExpression = None) -> MathExpression:
        if self.left_child:
            return self.set_left(child)  # type:ignore
        else:
            return self.set_right(child)  # type:ignore

    def get_child(self) -> Optional[MathExpression]:
        if self.left_child:
            return self.left
        else:
            return self.right

    def evaluate(self, context: Dict[str, Union[float, int]] = None) -> float:
        child = self.get_child()
        if child is None:
            raise ValueError("cannot evaluate unary expression without a valid child")
        return self.operate(child.evaluate(context))

    def operate(self, value: Union[float, int]) -> Union[float, int]:
        raise NotImplementedError("Must be implemented in subclass")


# ### Negation


class NegateExpression(UnaryExpression):
    """Negate an expression, e.g. `4` becomes `-4`"""

    @property
    def type_id(self) -> int:
        return MathTypeKeys["negate"]

    @property
    def name(self) -> str:
        return "-"

    def operate(self, value: Union[float, int]) -> Union[float, int]:
        return -value

    def __str__(self) -> str:
        return self.with_color("-{}".format(self.get_child()))

    def to_math_ml_fragment(self) -> str:
        """Convert this single node into MathML."""
        return f"-{super().to_math_ml_fragment()}"


# ### Function


class FunctionExpression(UnaryExpression):
    """A Specialized UnaryExpression that is used for functions.  The function name in
    text (used by the parser and tokenizer) is derived from the name() method on the
    class."""

    @property
    def name(self) -> str:
        raise NotImplementedError(
            "Must be implemented in subclass. Function is an abstract node"
        )

    def __str__(self) -> str:
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

    def evaluate(
        self, context: Dict[str, Union[float, int]] = None
    ) -> Union[float, int]:
        left, right = self._check()
        return self.operate(left.evaluate(context), right.evaluate(context))

    @property
    def name(self) -> str:
        raise NotImplementedError("Must be implemented in subclass")

    def get_ml_name(self) -> str:
        return self.name

    def operate(
        self, one: Union[float, int], two: Union[float, int]
    ) -> Union[float, int]:
        raise NotImplementedError("Must be implemented in subclass")

    def _check(self) -> Tuple[MathExpression, MathExpression]:
        if self.left is None or self.right is None:
            raise ValueError(
                "{}: left/right children must both be valid".format(
                    self.__class__.__name__
                )
            )
        return self.left, self.right

    def get_priority(self) -> int:
        """Return a number representing the order of operations priority
        of this node.  This can be used to check if a node is `locked`
        with respect to another node, i.e. the other node must be resolved
        first during evaluation because of it's priority.
        """
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
        self._check()
        """Return a boolean indicating whether this node should render itself with
        a set of enclosing parnetheses or not. This is used when serializing an
        expression, to ensure the tree maintains the proper order of operations. """
        binary_parent = cast(BinaryExpression, self.parent)
        parent_is_binary = isinstance(binary_parent, BinaryExpression)
        self_pri = self.get_priority()
        if not parent_is_binary or not binary_parent:
            return False

        parent_pri = binary_parent.get_priority()
        if parent_pri > self_pri:
            return True

        parent_side = binary_parent.get_side(self)
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

    def __str__(self) -> str:
        self._check()
        out = f"{self.left} {self.with_color(self.name)} {self.right}"
        return f"({out})" if self.self_parens() else out

    def to_math_ml_fragment(self) -> str:
        """Render this node as a MathML element fragment"""
        right, left = self._check()
        right_ml = right.to_math_ml_fragment()
        left_ml = left.to_math_ml_fragment()
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
    def type_id(self) -> int:
        return MathTypeKeys["equal"]

    @property
    def name(self) -> str:
        return "="

    def operate(
        self, one: Union[float, int], two: Union[float, int]
    ) -> Union[float, int]:
        """This is where assignment of context variables might make sense. But context
        is not present in the expression's `operate` method.

        !!! warning

            TODO: Investigate this thoroughly.
        """
        raise ValueError("Unsupported operation. Euqality has no operation to perform.")


class AddExpression(BinaryExpression):
    """Add one and two"""

    @property
    def type_id(self) -> int:
        return MathTypeKeys["add"]

    @property
    def name(self) -> str:
        return "+"

    def operate(
        self, one: Union[float, int], two: Union[float, int]
    ) -> Union[float, int]:
        return one + two


class SubtractExpression(BinaryExpression):
    """Subtract one from two"""

    @property
    def type_id(self) -> int:
        return MathTypeKeys["subtract"]

    @property
    def name(self) -> str:
        return "-"

    def operate(
        self, one: Union[float, int], two: Union[float, int]
    ) -> Union[float, int]:
        return one - two


class MultiplyExpression(BinaryExpression):
    """Multiply one and two"""

    @property
    def type_id(self) -> int:
        return MathTypeKeys["multiply"]

    @property
    def name(self) -> str:
        return "*"

    def get_ml_name(self) -> str:
        return "&#183;"

    def operate(
        self, one: Union[float, int], two: Union[float, int]
    ) -> Union[float, int]:
        return one * two

    def __str__(self) -> str:
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

    def to_math_ml_fragment(self) -> str:
        left, right = self._check()
        right_ml = right.to_math_ml_fragment()
        left_ml = left.to_math_ml_fragment()
        if isinstance(self.left, ConstantExpression):
            if isinstance(self.right, VariableExpression) or isinstance(
                self.right, PowerExpression
            ):
                return "{}{}".format(left_ml, right_ml)

        return super().to_math_ml_fragment()


class DivideExpression(BinaryExpression):
    """Divide one by two"""

    @property
    def type_id(self) -> int:
        return MathTypeKeys["divide"]

    @property
    def name(self) -> str:
        return "/"

    def get_ml_name(self) -> str:
        return "&#247;"

    def to_math_ml_fragment(self) -> str:
        left, right = self._check()
        left_ml = left.to_math_ml_fragment()
        right_ml = right.to_math_ml_fragment()
        return f"<mfrac><mi>{left_ml}</mi><mi>{right_ml}</mi></mfrac>"

    def operate(
        self, one: Union[float, int], two: Union[float, int]
    ) -> Union[float, int]:
        if two == 0:
            return float("nan")
        else:
            return one / two


class PowerExpression(BinaryExpression):
    """Raise one to the power of two"""

    @property
    def type_id(self) -> int:
        return MathTypeKeys["power"]

    @property
    def name(self) -> str:
        return "^"

    def to_math_ml_fragment(self) -> str:
        left, right = self._check()
        right_ml = right.to_math_ml_fragment()
        left_ml = left.to_math_ml_fragment()
        # if left is mult, enclose only right in msup
        if isinstance(self.left, MultiplyExpression):
            left_ml = self.make_ml_tag("mrow", left_ml, self.classes)

        return self.make_ml_tag("msup", "{}{}".format(left_ml, right_ml), self.classes)

    def operate(
        self, one: Union[float, int], two: Union[float, int]
    ) -> Union[float, int]:
        return np.power(one, two)

    def __str__(self):
        return "{}{}{}".format(self.left, self.with_color(self.name), self.right)


class ConstantExpression(MathExpression):
    """A Constant value node, where the value is accessible as `node.value`"""

    value: Union[float, int]

    @property
    def name(self):
        if self.value % 1 == 0:
            return f"{int(self.value)}"
        return np.format_float_positional(self.value, trim="-")

    @property
    def type_id(self):
        return MathTypeKeys["constant"]

    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def clone(self):
        result = super().clone()
        result.value = self.value
        return result

    def evaluate(self, context: Dict[str, Union[float, int]] = None):
        return self.value

    def __str__(self) -> str:
        return self.with_color(self.name)

    def to_math_ml_fragment(self) -> str:
        return self.make_ml_tag("mn", str(self.value), self.classes)


class VariableExpression(MathExpression):
    identifier: Optional[str]

    @property
    def name(self) -> str:
        return f"{self.identifier}"

    @property
    def type_id(self) -> int:
        id = f"_{self.identifier.lower()[0]}" if self.identifier is not None else ""
        return MathTypeKeys[f"variable{id}"]

    def __init__(self, identifier: str = None):
        super().__init__()
        self.identifier = identifier

    def clone(self) -> "VariableExpression":
        result = cast(VariableExpression, super().clone())
        result.identifier = self.identifier
        return result

    def _check(self):
        if self.identifier is None:
            raise ValueError("identifier must be a letter")

    def __str__(self) -> str:
        self._check()
        return self.with_color("{}".format(self.identifier))

    def to_math_ml_fragment(self) -> str:
        self._check()
        return self.make_ml_tag("mi", str(self.identifier), self.classes)

    def evaluate(self, context: Dict[str, Union[int, float]] = None):
        self._check()
        id = cast(str, self.identifier)
        if context and context.get(id, None) is not None:
            return context[id]

        raise ValueError(
            "cannot evaluate statement with None variable: {}".format(self.identifier)
        )


class AbsExpression(FunctionExpression):
    """Evaluates the absolute value of an expression."""

    @property
    def type_id(self) -> int:
        return MathTypeKeys["abs"]

    @property
    def name(self) -> str:
        return "abs"

    def operate(self, value: Union[float, int]) -> Union[float, int]:
        return np.absolute(value)


class SgnExpression(FunctionExpression):
    @property
    def type_id(self):
        return MathTypeKeys["sgn"]

    @property
    def name(self):
        return "sgn"

    def operate(self, value: Union[float, int]) -> Union[float, int]:
        """Determine the sign of an value.

        # Returns
        (int): -1 if negative, 1 if positive, 0 if 0"""
        if value < 0:
            return -1

        if value > 0:
            return 1

        return 0
