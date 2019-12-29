from typing import Dict, List, Optional, Union

# # Tokenizer

# ##Constants

# Define the known types of tokens for the Tokenizer.
TokensMap: Dict[str, int] = {
    "None": 1 << 0,
    "Constant": 1 << 1,
    "Variable": 1 << 2,
    "Plus": 1 << 3,
    "Minus": 1 << 4,
    "Multiply": 1 << 5,
    "Divide": 1 << 6,
    "Exponent": 1 << 7,
    "Factorial": 1 << 8,
    "OpenParen": 1 << 9,
    "CloseParen": 1 << 10,
    "Function": 1 << 11,
    "Equal": 1 << 12,
    "EOF": 1 << 13,
}

TokenNone = TokensMap["None"]
TokenConstant = TokensMap["Constant"]
TokenVariable = TokensMap["Variable"]
TokenPlus = TokensMap["Plus"]
TokenMinus = TokensMap["Minus"]
TokenMultiply = TokensMap["Multiply"]
TokenDivide = TokensMap["Divide"]
TokenExponent = TokensMap["Exponent"]
TokenFactorial = TokensMap["Factorial"]
TokenOpenParen = TokensMap["OpenParen"]
TokenCloseParen = TokensMap["CloseParen"]
TokenFunction = TokensMap["Function"]
TokenEqual = TokensMap["Equal"]
TokenEOF = TokensMap["EOF"]


class Token:
    value: Union[str, int, float]
    type: int

    def __init__(self, value: Union[str, int, float], type: int):
        self.value = value
        self.type = type

    def __str__(self):
        return "[type={}],[value={}]".format(self.type, self.value)


class TokenContext:
    tokens: List[Token]
    index: int
    buffer: str
    chunk: str

    def __init__(
        self,
        *,
        tokens: Optional[List[Token]] = None,
        index: int = 0,
        buffer: str = "",
        chunk: str = "",
    ):
        self.tokens = tokens if tokens is not None else []
        self.index = index
        self.buffer = buffer
        self.chunk = chunk


class Tokenizer:
    """The Tokenizer produces a list of tokens from an input string."""

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

    def is_alpha(self, c: str) -> bool:
        """Is this character a letter"""
        return ("a" <= c and c <= "z") or ("A" <= c and c <= "Z")

    def is_number(self, c: str) -> bool:
        """Is this character a number"""
        return "." == c or ("0" <= c and c <= "9")

    def eat_token(self, context: TokenContext, typeFn):
        """Eat all of the tokens of a given type from the front of the stream
        until a different type is hit, and return the text."""
        res = ""
        for ch in list(context.chunk):
            if not typeFn(ch):
                return res
            res = res + str(ch)

        return res

    def tokenize(self, buffer: str, terms=False) -> List[Token]:
        """Return an array of `Token`s from a given string input.
        This throws an exception if an unknown token type is found in the input."""
        context = TokenContext(buffer=buffer, chunk=str(buffer))
        while context.chunk and (
            self.identify_constants(context)
            or self.identify_alphas(context)
            or self.identify_operators(context)
        ):
            context.chunk = context.buffer[context.index :]

        context.tokens.append(Token("", TokenEOF))
        return context.tokens

    def identify_operators(self, context: TokenContext) -> bool:
        """Identify and tokenize operators."""
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
            raise Exception(f'Invalid token "{ch}" in expression: {context.buffer}')
        context.index = context.index + 1
        return True

    def identify_alphas(self, context: TokenContext) -> int:
        """Identify and tokenize functions and variables."""
        if not self.is_alpha(context.chunk[0]):
            return False

        variable = self.eat_token(context, self.is_alpha)
        if variable in self.functions:
            context.tokens.append(Token(variable, TokenFunction))
        else:
            # Each letter is its own variable
            for c in variable:
                context.tokens.append(Token(c, TokenVariable))

        context.index += len(variable)
        return len(variable)

    def identify_constants(self, context: TokenContext) -> int:
        """Identify and tokenize a constant number."""
        if not self.is_number(context.chunk[0]):
            return 0

        val = self.eat_token(context, self.is_number)
        context.tokens.append(Token(val, TokenConstant))
        context.index += len(val)
        return len(val)


def coerce_to_number(value: str) -> Union[int, float]:
    return float(value) if "e" in value or "." in value else int(value)
