from typing import Dict, List, Optional, Union

from .expressions import (
    AddExpression,
    ConstantExpression,
    DivideExpression,
    EqualExpression,
    MathExpression,
    MultiplyExpression,
    NegateExpression,
    PowerExpression,
    SubtractExpression,
    VariableExpression,
)
from .tokenizer import (
    Token,
    TokenCloseParen,
    TokenConstant,
    TokenContext,
    TokenDivide,
    TokenEOF,
    TokenEqual,
    TokenExponent,
    TokenFactorial,
    TokenFunction,
    Tokenizer,
    TokenMinus,
    TokenMultiply,
    TokenNone,
    TokenOpenParen,
    TokenPlus,
    TokenVariable,
    coerce_to_number,
)


class ParserException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class InvalidExpression(ParserException):
    pass


class OutOfTokens(ParserException):
    pass


class InvalidSyntax(ParserException):
    pass


class UnexpectedBehavior(ParserException):
    pass


class TrailingTokens(ParserException):
    pass


class TokenSet:
    """TokenSet objects are bitmask combinations for checking to see
    if a token is part of a valid set. """

    tokens: int

    def __init__(self, source: int):
        self.tokens = source

    def add(self, addTokens: int) -> "TokenSet":
        """Add tokens to self set and return a TokenSet representing
        their combination of flags.  Value can be an integer or an instance
        of `TokenSet`"""
        return TokenSet(self.tokens | addTokens)

    def contains(self, type: int) -> bool:
        """Returns true if the given type is part of this set"""
        return (self.tokens & type) != 0


_FIRST_FUNCTION: TokenSet = TokenSet(TokenFunction)
_FIRST_FACTOR: TokenSet = _FIRST_FUNCTION.add(TokenVariable | TokenOpenParen)
_FIRST_FACTOR_PREFIX: TokenSet = _FIRST_FACTOR.add(TokenConstant)
_FIRST_UNARY: TokenSet = _FIRST_FACTOR_PREFIX.add(TokenMinus)
_FIRST_EXP: TokenSet = _FIRST_UNARY
_FIRST_MULT: TokenSet = _FIRST_UNARY
_FIRST_ADD: TokenSet = _FIRST_UNARY
_FIRST_EQUAL: TokenSet = _FIRST_UNARY

# Precedence checks
_IS_ADD: TokenSet = TokenSet(TokenPlus | TokenMinus)
_IS_MULT: TokenSet = TokenSet(TokenMultiply | TokenDivide)
_IS_EXP: TokenSet = TokenSet(TokenExponent)
_IS_EQUAL: TokenSet = TokenSet(TokenEqual)


# NOTE: This cannot be shared between threads because it stores state in self.current_token and self.tokens
class ExpressionParser:
    """Parser for converting text into binary trees. Trees encode the order of
    operations for an input, and allow evaluating it to detemrine the expression
    value.

    ### Grammar Rules

    Symbols:
    ```
    ( )    == Non-terminal
    { }*   == 0 or more occurrences
    { }+   == 1 or more occurrences
    { }?   == 0 or 1 occurrences
    [ ]    == Mandatory (1 must occur)
    |      == logical OR
    " "    == Terminal symbol (literal)
    ```

    Non-terminals defined/parsed by Tokenizer:
    ```
    (Constant) = anything that can be parsed by `float(in)`
    (Variable) = any string containing only letters (a-z and A-Z)
    ```

    Rules:
    ```
    (Function)     = [ functionName ] "(" (AddExp) ")"
    (Factor)       = { (Variable) | (Function) | "(" (AddExp) ")" }+ { { "^" }? (UnaryExp) }?
    (FactorPrefix) = [ (Constant) { (Factor) }? | (Factor) ]
    (UnaryExp)     = { "-" }? (FactorPrefix)
    (ExpExp)       = (UnaryExp) { { "^" }? (UnaryExp) }?
    (MultExp)      = (ExpExp) { { "*" | "/" }? (ExpExp) }*
    (AddExp)       = (MultExp) { { "+" | "-" }? (MultExp) }*
    (EqualExp)     = (AddExp) { { "=" }? (AddExp) }*
    (start)        = (EqualExp)
    ```
    """

    _parse_cache: Dict[str, MathExpression]
    _tokens_cache: Dict[str, List[Token]]

    # Initialize the tokenizer.
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.clear_cache()

    def clear_cache(self):
        self._tokens_cache = {}
        self._parse_cache = {}

    def tokenize(self, input_text: str):
        if input_text not in self._tokens_cache:
            self._tokens_cache[input_text] = self.tokenizer.tokenize(input_text)
        return self._tokens_cache[input_text][:]

    def parse(self, input_text: str) -> MathExpression:
        """Parse a string representation of an expression into a tree
        that can be later evaluated.

        Returns : The evaluatable expression tree.
        """
        if input_text in self._parse_cache:
            return self._parse_cache[input_text]
        self._parse_cache[input_text] = self._parse(self.tokenize(input_text))
        return self._parse_cache[input_text]

    def _parse(self, tokens: List[Token]) -> MathExpression:
        """Parse a given list of tokens into an expression tree"""
        self.tokens = tokens
        self.current_token = Token("", TokenNone)
        if not self.next():
            raise InvalidExpression("Cannot parse an empty function")

        expression: MathExpression = self.parse_equal()
        leftover = ""
        while self.current_token.type != TokenEOF:
            leftover = f"{leftover}{self.current_token.value}"
            self.next()

        if leftover != "":
            raise TrailingTokens("Trailing characters: {}".format(leftover))
        return expression

    def parse_equal(self) -> MathExpression:
        if not self.check(_FIRST_ADD):
            raise InvalidSyntax("Invalid expression")

        exp = self.parse_add()
        while self.check(_IS_EQUAL):
            opType = self.current_token.type
            opValue = self.current_token.value
            self.eat(opType)
            expected = self.check(_FIRST_ADD)
            right = None
            if expected:
                right = self.parse_add()

            if not expected or not right:
                raise UnexpectedBehavior(
                    "Expected an expression after = operator, got: {}".format(
                        self.current_token.value
                    )
                )

            if opType != TokenEqual:
                raise UnexpectedBehavior(
                    "Expected plus or minus, got: {}".format(opValue)
                )

            exp = EqualExpression(exp, right)

        return exp

    def parse_add(self) -> MathExpression:
        if not self.check(_FIRST_MULT):
            raise InvalidSyntax("Invalid expression")

        exp = self.parse_mult()
        while self.check(_IS_ADD):
            opType = self.current_token.type
            opValue = self.current_token.value
            self.eat(opType)
            expected = self.check(_FIRST_MULT)
            right = None
            if expected:
                right = self.parse_mult()

            if not expected or not right:
                raise UnexpectedBehavior(
                    "Expected an expression after + or - operator, got: {}".format(
                        self.current_token.value
                    )
                )

            if opType == TokenPlus:
                exp = AddExpression(exp, right)
            elif opType == TokenMinus:
                exp = SubtractExpression(exp, right)
            else:
                raise UnexpectedBehavior(
                    "Expected plus or minus, got: {}".format(opValue)
                )

        return exp

    def parse_mult(self) -> MathExpression:
        if not self.check(_FIRST_EXP):
            raise InvalidSyntax("Invalid expression")

        exp = self.parse_exponent()
        while self.check(_IS_MULT):
            opType = self.current_token.type
            opValue = self.current_token.value
            self.eat(opType)
            expected = self.check(_FIRST_EXP)
            right = None
            if expected:
                right = self.parse_mult()

            if not expected or right is None:
                raise InvalidSyntax(
                    "Expected an expression after * or / operator, got: {}".format(
                        opValue
                    )
                )

            if opType == TokenMultiply:
                exp = MultiplyExpression(exp, right)
            elif opType == TokenDivide:
                exp = DivideExpression(exp, right)
            else:
                raise UnexpectedBehavior(
                    "Expected mult or divide, got: {}".format(opValue)
                )
        return exp

    def parse_exponent(self) -> MathExpression:
        if not self.check(_FIRST_UNARY):
            raise InvalidSyntax("Invalid expression")

        exp = self.parse_unary()
        if self.check(TokenSet(TokenExponent)):
            opType = self.current_token.type
            self.eat(opType)
            if not self.check(_FIRST_UNARY):
                raise InvalidSyntax("Expected an expression after ^ operator")

            right = self.parse_unary()
            if opType == TokenExponent:
                exp = PowerExpression(exp, right)
            else:
                raise UnexpectedBehavior("Expected exponent, got: {}".format(opType))
        return exp

    def parse_unary(self) -> MathExpression:
        value: Union[float, int] = 0
        negate = False
        if self.current_token.type == TokenMinus:
            self.eat(TokenMinus)
            negate = True
        expected = self.check(_FIRST_FACTOR_PREFIX)
        exp: Optional[MathExpression] = None
        if expected:
            if self.current_token.type == TokenConstant:
                if isinstance(self.current_token.value, str):
                    value = coerce_to_number(self.current_token.value)
                else:
                    value = self.current_token.value
                # Flip parse as float/int based on whether the value text
                if negate:
                    value = -value
                    negate = False

                exp = ConstantExpression(value)
                self.eat(TokenConstant)

            if self.check(_FIRST_FACTOR):
                if exp is None:
                    exp = self.parse_factors()
                else:
                    exp = MultiplyExpression(exp, self.parse_factors())

        if not expected or exp is None:
            raise InvalidSyntax(
                "Expected a function, variable or parenthesis after - or + but got : {}".format(
                    self.current_token.value
                )
            )
        if negate:
            return NegateExpression(exp)

        return exp

    def parse_factors(self) -> MathExpression:
        right = None
        found = True
        factors: List[MathExpression] = []
        while found:
            right = None
            opType = self.current_token.type
            if opType == TokenVariable:
                factors.append(VariableExpression(str(self.current_token.value)))
                self.eat(TokenVariable)
            elif opType == TokenFunction:
                factors.append(self.parse_function())
            elif opType == TokenOpenParen:
                self.eat(TokenOpenParen)
                factors.append(self.parse_add())
                self.eat(TokenCloseParen)
            else:
                raise UnexpectedBehavior(
                    "Unexpected token in Factor: {}".format(self.current_token.type)
                )

            found = self.check(_FIRST_FACTOR)

        if len(factors) == 0:
            raise InvalidExpression("No factors")

        exp: Optional[MathExpression] = None
        if self.check(_IS_EXP):
            opType = self.current_token.type
            self.eat(opType)
            if not self.check(_FIRST_UNARY):
                raise InvalidSyntax("Expected an expression after ^ operator")

            right = self.parse_unary()
            exp = PowerExpression(factors[-1], right)

        if len(factors) == 1:
            return exp or factors[0]

        while len(factors) > 0:
            if exp is None:
                exp = factors.pop(0)

            exp = MultiplyExpression(exp, factors.pop(0))

        assert exp is not None
        return exp

    def parse_function(self) -> MathExpression:
        opFn = self.current_token.value
        self.eat(self.current_token.type)
        self.eat(TokenOpenParen)
        exp = self.parse_add()
        self.eat(TokenCloseParen)
        func = self.tokenizer.functions[opFn]
        if func is None:
            raise UnexpectedBehavior("Unknown Function type: {}".format(opFn))

        return func(exp)

    def next(self) -> bool:
        """Assign the next token in the queue to `self.current_token`.

        Return True if there are still more tokens in the queue, or False if there
        are no more tokens to look at."""

        if self.current_token.type == TokenEOF:
            raise OutOfTokens("Parsed beyond the end of the expression")

        self.current_token = self.tokens.pop(0)
        return self.current_token.type != TokenEOF

    def eat(self, type) -> bool:
        """Assign the next token in the queue to current_token if its type
        matches that of the specified parameter. If the type does not match,
        raise a syntax exception.

        Args:
            - `type` The type that your syntax expects @current_token to be
        """
        if self.current_token.type != type:
            raise InvalidSyntax("Missing: {}".format(type))

        return self.next()

    def check(self, tokens) -> bool:
        """Check if the `self.current_token` is a member of a set Token types
        
        Args:
            - `tokens` The set of Token types to check against
        
        `Returns` True if the `current_token`'s type is in the set else False"""

        return tokens.contains(self.current_token.type)
