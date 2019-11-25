import { FunctionExpression } from './expressions'
// # Tokenizer

// ##Constants

// Define the known types of tokens for the Tokenizer.
export const TOKEN_TYPES = {
  // tslint:disable:no-bitwise
  None: 1 << 0,
  Constant: 1 << 1,
  Variable: 1 << 2,
  Plus: 1 << 3,
  Minus: 1 << 4,
  Multiply: 1 << 5,
  Divide: 1 << 6,
  Exponent: 1 << 7,
  Factorial: 1 << 8,
  OpenParen: 1 << 9,
  CloseParen: 1 << 10,
  Function: 1 << 11,
  Equal: 1 << 12,
  EOF: 1 << 13
  // tslint:enable:no-bitwise
}

// ##Tokenizer

// Define a token
export class Token {
  constructor(public value: string, public type: number) {}
}

export interface ITokenContext {
  tokens: Token[]
  index: number
  buffer: string
  chunk: string
}

// The Tokenizer produces a list of tokens from an input string.
export class Tokenizer {
  // ###Functions Registry
  public functions: {
    [name: string]: FunctionExpression
  } = {}
  constructor() {
    this.find_functions()
  }

  // Populate the `@functions` object with all known `FunctionExpression`s
  // in Expressions
  public find_functions() {
    this.functions = {}
    // for (const key in Expressions) {
    //   const val = Expressions[key];
    //   const check = {};
    //   if (check.toString.call(val) !== '[object Function]') {
    //     continue;
    //   }
    //   const inst = new val();
    //   if (!(inst instanceof FunctionExpression)) {
    //     continue;
    //   }
    //   if (`${inst}` === '') {
    //     continue;
    //   }
    //   this.functions[`${inst}`] = val;
    // }
    return this
  }

  // ###Token Utilities

  // Is this character a letter
  public is_alpha(c: string) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
  }
  // Is this character a number
  public is_number(c: string) {
    return '.' === c || ('0' <= c && c <= '9')
  }
  // Eat all of the tokens of a given type from the front of the stream
  // until a different type is hit, and return the text.
  public eat_token(context: ITokenContext, typeFn: (c: string) => boolean) {
    let res = ''
    for (const ch of context.chunk) {
      if (!typeFn(ch)) {
        return res
      }
      res += `${ch}`
    }
    return res
  }

  // ###Tokenizantion
  // Return an array of `Token`s from a given string input.
  // This throws an exception if an unknown token type is found in
  // the input.
  public tokenize(buffer: string): Token[] {
    const context: ITokenContext = {
      buffer,
      tokens: [],
      index: 0,
      chunk: buffer.slice(0)
    }
    while (context.chunk && (this.identify_constants(context) || this.identify_alphas(context) || this.identify_operators(context))) {
      context.chunk = context.buffer.slice(context.index)
    }

    context.tokens.push(new Token('', TOKEN_TYPES.EOF))
    return context.tokens
  }

  // Identify and tokenize operators.
  public identify_operators(context: ITokenContext) {
    switch (context.chunk[0]) {
      case ' ':
      case '\t':
      case '\r':
      case '\n':
        break
      case '+':
        context.tokens.push(new Token('+', TOKEN_TYPES.Plus))
        break
      case '-':
        context.tokens.push(new Token('-', TOKEN_TYPES.Minus))
        break
      case '*':
        context.tokens.push(new Token('*', TOKEN_TYPES.Multiply))
        break
      case '/':
        context.tokens.push(new Token('/', TOKEN_TYPES.Divide))
        break
      case '^':
        context.tokens.push(new Token('^', TOKEN_TYPES.Exponent))
        break
      case '!':
        context.tokens.push(new Token('!', TOKEN_TYPES.Factorial))
        break
      case '(':
        context.tokens.push(new Token('(', TOKEN_TYPES.OpenParen))
        break
      case ')':
        context.tokens.push(new Token(')', TOKEN_TYPES.CloseParen))
        break
      case '=':
        context.tokens.push(new Token('=', TOKEN_TYPES.Equal))
        break
      default:
        throw new Error(`Invalid token '${context.chunk[0]}' in expression: ${context.buffer}`)
    }
    context.index += 1
    return true
  }

  // Identify and tokenize functions and variables.
  public identify_alphas(context: ITokenContext) {
    if (!this.is_alpha(context.chunk[0])) {
      return false
    }
    const variable = this.eat_token(context, this.is_alpha)
    let tokenType = TOKEN_TYPES.Variable
    if (this.functions[variable] !== undefined) {
      tokenType = TOKEN_TYPES.Function
    }
    context.tokens.push(new Token(variable, tokenType))
    context.index += variable.length
    return variable.length
  }

  // Identify and tokenize a constant number.
  public identify_constants(context: ITokenContext) {
    if (!this.is_number(context.chunk[0])) {
      return 0
    }
    const val = this.eat_token(context, this.is_number)
    context.tokens.push(new Token(val, TOKEN_TYPES.Constant))
    context.index += val.length
    return val.length
  }
}
