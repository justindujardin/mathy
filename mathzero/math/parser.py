from .expressions import (
    SubtractExpression,
    AddExpression,
    EqualExpression,
    MultiplyExpression,
    DivideExpression,
    PowerExpression,
    VariableExpression,
    ConstantExpression,
    NegateExpression,
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
)

#  Copyright (c) DuJardin Consulting, 2011
#  Portions Copyright (c) Microsoft Corporation, 2004

# # Parser


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


class UndefinedVariable(ParserException):
    pass


class CannotDifferentiate(ParserException):
    pass


## Token Sets

# TokenSet objects are simple bitmask combinations for checking to see
# if a token is part of a valid set.
# Represents a bitmask of Tokens that is used to determine if a token
# is valid for some state.
class TokenSet:
    def __init__(self, source):
        if isinstance(source, TokenSet):
            self.tokens = source.tokens
        elif type(source) is int:
            self.tokens = source

    # Add tokens to self set and return a TokenSet representing
    # their combination of flags.  Value can be an integer or
    # an instance of `TokenSet`.
    def add(self, addTokens):
        if isinstance(addTokens, TokenSet):
            addTokens = addTokens.tokens

        return TokenSet(self.tokens | addTokens)

    def contains(self, type):
        return (self.tokens & type) != 0


# # The Grammar Rules
#
#  Symbols:
#  -------------------------
#  ( )    == Non-terminal
#  { }*   == 0 or more occurrences
#  { }+   == 1 or more occurrences
#  { }?   == 0 or 1 occurrences
#  [ ]    == Mandatory (1 must occur)
#  |      == logical OR
#  " "    == Terminal symbol (literal)
#
#  Non-terminals defined/parsed by Tokenizer:
#  ------------------------------------------
#  (Constant) = anything that can be parsed by `float(in)`
#  (Variable) = any string containing only letters (a-z and A-Z)


#
#     (Function)     = [ functionName ] "(" (AddExp) ")"
#     (Factor)       = { (Variable) | (Function) | "(" (AddExp) ")" }+ { { "^" }? (UnaryExp) }?
#     (FactorPrefix) = [ (Constant) { (Factor) }? | (Factor) ]
#     (UnaryExp)     = { "-" }? (FactorPrefix)
#     (ExpExp)       = (UnaryExp) { { "^" }? (UnaryExp) }?
#     (MultExp)      = (ExpExp) { { "*" | "/" }? (ExpExp) }*
#     (AddExp)       = (MultExp) { { "+" | "-" }? (MultExp) }*
#     (EqualExp)     = (AddExp) { { "=" }? (AddExp) }*
#     (start)        = (EqualExp)

FIRST_FUNCTION = TokenSet(TokenFunction)
FIRST_FACTOR = FIRST_FUNCTION.add(TokenVariable | TokenOpenParen)
FIRST_FACTOR_PREFIX = FIRST_FACTOR.add(TokenConstant)
FIRST_UNARY = FIRST_EXP = FIRST_FACTOR_PREFIX.add(TokenMinus)
FIRST_MULT = FIRST_ADD = FIRST_EQUAL = FIRST_UNARY

# Precedence checks
IS_ADD = TokenSet(TokenPlus | TokenMinus)
IS_MULT = TokenSet(TokenMultiply | TokenDivide)
IS_EXP = TokenSet(TokenExponent)
IS_EQUAL = TokenSet(TokenEqual)

_parse_cache = {}


class ExpressionParser:
    # Initialize the tokenizer.
    def __init__(self):
        self.tokenizer = Tokenizer()

    # Parse a string representation of an expression into a tree that can be
    # later evaluated.
    # Returns : The evaluatable expression tree.
    def parse(self, input):
        global _parse_cache
        if input in _parse_cache:
            return _parse_cache[input].clone()

        self.tokens = self.tokenizer.tokenize(input)
        self.currentToken = Token("", TokenNone)
        if not self.next():
            raise InvalidExpression("Cannot parse an empty function")

        expression = self.parseEqual()
        leftover = ""
        while self.currentToken.type != TokenEOF:
            leftover = leftover + self.currentToken.value
            self.next()

        if leftover != "":
            raise TrailingTokens("Trailing characters: {}".format(leftover))
        _parse_cache[input] = expression
        return expression

    def clear_cache(self):
        global _parse_cache
        _parse_cache = {}

    def parseEqual(self):
        if not self.check(FIRST_ADD):
            raise InvalidSyntax("Invalid expression")

        exp = self.parseAdd()
        while self.check(IS_EQUAL):
            opType = self.currentToken.type
            opValue = self.currentToken.value
            self.eat(opType)
            expected = self.check(FIRST_ADD)
            right = None
            if expected:
                right = self.parseAdd()

            if not expected or not right:
                raise UnexpectedBehavior(
                    "Expected an expression after = operator, got: {}".format(
                        self.currentToken.value
                    )
                )

            if opType != TokenEqual:
                raise UnexpectedBehavior(
                    "Expected plus or minus, got: {}".format(opValue)
                )

            exp = EqualExpression(exp, right)

        return exp

    def parseAdd(self):
        if not self.check(FIRST_MULT):
            raise InvalidSyntax("Invalid expression")

        exp = self.parseMult()
        while self.check(IS_ADD):
            opType = self.currentToken.type
            opValue = self.currentToken.value
            self.eat(opType)
            expected = self.check(FIRST_MULT)
            right = None
            if expected:
                right = self.parseMult()

            if not expected or not right:
                raise UnexpectedBehavior(
                    "Expected an expression after + or - operator, got: {}".format(
                        self.currentToken.value
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

    def parseMult(self):
        if not self.check(FIRST_EXP):
            raise InvalidSyntax("Invalid expression")

        exp = self.parseExp()
        while self.check(IS_MULT):
            opType = self.currentToken.type
            opValue = self.currentToken.value
            self.eat(opType)
            expected = self.check(FIRST_EXP)
            right = None
            if expected:
                right = self.parseMult()

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

    def parseExp(self):
        if not self.check(FIRST_UNARY):
            raise InvalidSyntax("Invalid expression")

        exp = self.parseUnary()
        if self.check(TokenSet(TokenExponent)):
            opType = self.currentToken.type
            self.eat(opType)
            if not self.check(FIRST_UNARY):
                raise InvalidSyntax("Expected an expression after ^ operator")

            right = self.parseUnary()
            if opType == TokenExponent:
                exp = PowerExpression(exp, right)
            else:
                raise UnexpectedBehavior("Expected exponent, got: {}".format(opType))
        return exp

    def parseUnary(self):
        value = 0
        negate = False
        if self.currentToken.type == TokenMinus:
            self.eat(TokenMinus)
            negate = True
        expected = self.check(FIRST_FACTOR_PREFIX)
        exp = None
        if expected:
            if self.currentToken.type == TokenConstant:
                value = self.currentToken.value
                # Flip parse as float/int based on whether the value text
                value = float(value) if "e" in value or "." in value else int(value)
                if negate:
                    value = -value
                    negate = False

                exp = ConstantExpression(value)
                self.eat(TokenConstant)

            if self.check(FIRST_FACTOR):
                if exp is None:
                    exp = self.parseFactors()
                else:
                    exp = MultiplyExpression(exp, self.parseFactors())

        if not expected or exp is None:
            raise InvalidSyntax(
                "Expected a function, variable or parenthesis after - or + but got : {}".format(
                    self.currentToken.value
                )
            )
        if negate:
            return NegateExpression(exp)

        return exp

    def parseFactorPrefix(self):
        exp = None
        if self.currentToken.type == TokenConstant:
            exp = ConstantExpression(float(self.currentToken.value))
            self.eat(TokenConstant)

        if self.check(FIRST_FACTOR):
            if exp is None:
                return self.parseFactors()

            return MultiplyExpression(exp, self.parseFactors())

        return exp

    def parseFactors(self):
        right = None
        found = True
        factors = []
        while found:
            right = None
            opType = self.currentToken.type
            if opType == TokenVariable:
                factors.append(VariableExpression(self.currentToken.value))
                self.eat(TokenVariable)
            elif opType == TokenFunction:
                factors.append(self.parseFunction())
            elif opType == TokenOpenParen:
                self.eat(TokenOpenParen)
                factors.append(self.parseAdd())
                self.eat(TokenCloseParen)
            else:
                raise UnexpectedBehavior(
                    "Unexpected token in Factor: {}".format(self.currentToken.type)
                )

            found = self.check(FIRST_FACTOR)

        if len(factors) == 0:
            raise InvalidExpression("No factors")

        exp = None
        if self.check(IS_EXP):
            opType = self.currentToken.type
            self.eat(opType)
            if not self.check(FIRST_UNARY):
                raise InvalidSyntax("Expected an expression after ^ operator")

            right = self.parseUnary()
            exp = PowerExpression(factors[-1], right)

        if len(factors) == 1:
            return exp or factors[0]

        while len(factors) > 0:
            if exp is None:
                exp = factors.pop(0)

            exp = MultiplyExpression(exp, factors.pop(0))

        return exp

    def parseFunction(self):
        opFn = self.currentToken.value
        self.eat(self.currentToken.type)
        self.eat(TokenOpenParen)
        exp = self.parseAdd()
        self.eat(TokenCloseParen)
        func = self.tokenizer.functions[opFn]
        if func is None:
            raise UnexpectedBehavior("Unknown Function type: {}".format(opFn))

        return func(exp)

    # Assign the next token in the queue to @currentToken
    # Returns True if there are still more tokens in the queue, or
    # False if we have looked at all available tokens already
    def next(self):
        if self.currentToken.type == TokenEOF:
            raise OutOfTokens("Parsed beyond the end of the expression")

        self.currentToken = self.tokens.pop(0)
        return self.currentToken.type != TokenEOF

    # Assign the next token in the queue to @currentToken if the @currentToken's
    # type matches that of the specified parameter.  If the @currentToken's
    # type does not match the parameter, raise a syntax exception
    #
    # `type` The type that your syntax expects @currentToken to be
    def eat(self, type):
        if self.currentToken.type != type:
            raise InvalidSyntax("Missing: {}".format(type))

        return self.next()

    # Check if the @currentToken is a member of a set Token types
    #
    # `tokens` The set of Token types to check against
    # Returns True if the @currentToken's type is in the set or False if it is not
    def check(self, tokens):
        return tokens.contains(self.currentToken.type)

