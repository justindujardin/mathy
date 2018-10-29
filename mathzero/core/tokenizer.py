import math

# # Tokenizer

# ##Constants

# Define the known types of tokens for the Tokenizer.

TokenNone = 1 << 0
TokenConstant = 1 << 1
TokenVariable = 1 << 2
TokenPlus = 1 << 3
TokenMinus = 1 << 4
TokenMultiply = 1 << 5
TokenDivide = 1 << 6
TokenExponent = 1 << 7
TokenFactorial = 1 << 8
TokenOpenParen = 1 << 9
TokenCloseParen = 1 << 10
TokenFunction = 1 << 11
TokenEqual = 1 << 12
TokenEOF = 1 << 13

# ##Tokenizer

# Define a token
class Token:
    def __init__(self, value: str, type: int):
        self.value = value
        self.type = type

    def to_feature(self):
        # Constant values aren't turned into ordinals
        if self.type == TokenConstant:
            token_value = coerce_to_number(self.value)
        elif self.type == TokenEOF:
            token_value = 0
        elif self.type == TokenFunction:
            token_value = self.value
        else:
            token_value = ord(self.value)
        return [token_value, float(self.type)]

    @classmethod
    def from_feature(cls, feature):
        if not type(feature) == list:
            raise TypeError("feature must be a list of numbers")
        token_type = feature[1]
        token_value = feature[0]
        if token_type == TokenConstant:
            token_value = str(int(token_value) if math.isclose(token_value % 1, 0.0) else float(token_value))
        elif token_type == TokenEOF:
            token_value = ""
        else:
            token_value = chr(int(token_value))
        return Token(token_value, token_type)

    def __str__(self):
        return "[type={}],[value={}]".format(self.type, self.value)


class TokenContext:
    def __init__(self, tokens=None, index=0, buffer="", chunk=""):
        self.tokens = tokens if tokens is not None else []
        self.index = index
        self.buffer = buffer
        self.chunk = chunk


# The Tokenizer produces a list of tokens from an input string.
class Tokenizer:
    # ###Functions Registry
    def __init__(self):
        self.find_functions()

    # Populate the `@functions` object with all known `FunctionExpression`s
    # in Expressions
    def find_functions(self):
        self.functions = {}
        # for (key in Expressions) {
        #   val = Expressions[key];
        #   check = {};
        #   if (check.toString.call(val) != "[object Function]") {
        #     continue;
        #   }
        #   inst = val();
        #   if (not (inst instanceof FunctionExpression)) {
        #     continue;
        #   }
        #   if (`${inst}` === "") {
        #     continue;
        #   }
        #   self.functions[`${inst}`] = val;
        # }
        return self

    # ###Token Utilities

    # Is this character a letter
    def is_alpha(self, c) -> bool:
        return ("a" <= c and c <= "z") or ("A" <= c and c <= "Z")

    # Is this character a number
    def is_number(self, c) -> bool:
        return "." == c or ("0" <= c and c <= "9")

    # Eat all of the tokens of a given type from the front of the stream
    # until a different type is hit, and return the text.
    def eat_token(self, context: TokenContext, typeFn):
        res = ""
        for ch in list(context.chunk):
            if not typeFn(ch):
                return res
            res = res + str(ch)

        return res

    # ###Tokenizantion
    # Return an array of `Token`s from a given string input.
    # This throws an exception if an unknown token type is found in
    # the input.
    def tokenize(self, buffer: str, terms=False):
        context = TokenContext(buffer=buffer, chunk=str(buffer))
        while context.chunk and (
            self.identify_constants(context)
            or self.identify_alphas(context)
            or self.identify_operators(context)
        ):
            context.chunk = context.buffer[context.index :]

        context.tokens.append(Token("", TokenEOF))
        return context.tokens

    # Identify and tokenize operators.
    def identify_operators(self, context):
        ch = context.chunk[0]
        if ch == " " or ch == "\t" or ch == "\r" or ch == "\n":
            pass
        elif ch == "+":
            context.tokens.append(Token("+", TokenPlus))
        elif ch == "-" or ch == "–":
            context.tokens.append(Token("-", TokenMinus))
        elif ch == "*":
            context.tokens.append(Token("*", TokenMultiply))
        elif ch == "/":
            context.tokens.append(Token("/", TokenDivide))
        elif ch == "^":
            context.tokens.append(Token("^", TokenExponent))
        elif ch == "!":
            context.tokens.append(Token("!", TokenFactorial))
        elif ch == "(" or ch == "[":
            context.tokens.append(Token("(", TokenOpenParen))
        elif ch == ")" or ch == "]":
            context.tokens.append(Token(")", TokenCloseParen))
        elif ch == "=":
            context.tokens.append(Token("=", TokenEqual))
        else:
            raise Exception(
                'Invalid token "{}" in expression: {}'.format(ch, context.buffer)
            )
        context.index = context.index + 1
        return True

    # Identify and tokenize functions and variables.
    def identify_alphas(self, context):
        if not self.is_alpha(context.chunk[0]):
            return False

        variable = self.eat_token(context, self.is_alpha)
        if variable in self.functions:
            context.tokens.append(Token(variable, TokenFunction))
        else:
            # Each letter is its own variable
            [context.tokens.append(Token(c, TokenVariable)) for c in variable]

        context.index += len(variable)
        return len(variable)

    # Identify and tokenize a constant number.
    def identify_constants(self, context):
        if not self.is_number(context.chunk[0]):
            return 0

        val = self.eat_token(context, self.is_number)
        context.tokens.append(Token(val, TokenConstant))
        context.index += len(val)
        return len(val)


def coerce_to_number(value):
    return float(value) if "e" in value or "." in value else int(value)
