//  Copyright (c) DuJardin Consulting, 2011
//  Portions Copyright (c) Microsoft Corporation, 2004
import { TOKEN_TYPES, Tokenizer, Token } from './tokenizer';
import {
  FunctionExpression,
  EqualExpression,
  AddExpression,
  SubtractExpression,
  MultiplyExpression,
  DivideExpression,
  PowerExpression,
  ConstantExpression,
  NegateExpression,
  VariableExpression,
  MathExpression
} from './expressions';

// # Parser

export class ParserException {
  constructor(public message: string) {}
  toString() {
    return this.message;
  }
}

export class InvalidExpression extends ParserException {}
export class OutOfTokens extends ParserException {}
export class InvalidSyntax extends ParserException {}
export class UnexpectedBehavior extends ParserException {}
export class TrailingTokens extends ParserException {}
export class UndefinedVariable extends ParserException {}
export class CannotDifferentiate extends ParserException {}

// Token Sets

// TokenSet objects are simple bitmask combinations for checking to see
// if a token is part of a valid set.
// Represents a bitmask of Tokens that is used to determine if a token
// is valid for some state.
export class TokenSet {
  public tokens: number = TOKENS.None;
  constructor(source: TokenSet | number) {
    if (source instanceof TokenSet) {
      this.tokens = source.tokens;
    } else if (typeof source === 'number') {
      this.tokens = source;
    }
  }

  // Add tokens to this set and return a new TokenSet representing
  // their combination of flags.  Value can be an integer or
  // an instance of `TokenSet`.
  add(tokens: number) {
    return new TokenSet(this.tokens | tokens);
  }

  contains(type: number) {
    return (this.tokens & type) !== 0;
  }
}

// # The Grammar Rules
//
//  Symbols:
//  -------------------------
//  ( )    == Non-terminal
//  { }*   == 0 or more occurrences
//  { }+   == 1 or more occurrences
//  { }?   == 0 or 1 occurrences
//  [ ]    == Mandatory (1 must occur)
//  |      == logical OR
//  " "    == Terminal symbol (literal)
//
//  Non-terminals defined/parsed by Tokenizer:
//  ------------------------------------------
//  (Constant) = anything that can be parsed by parseFloat
//  (Variable) = any string containing only letters (a-z and A-Z)

// Brief constant to token types
export const TOKENS = TOKEN_TYPES;
//
//     (Function)     = [ functionName ] "(" (AddExp) ")"
//     (Factor)       = { (Variable) | (Function) | "(" (AddExp) ")" }+ { { "^" }? (UnaryExp) }?
//     (FactorPrefix) = [ (Constant) { (Factor) }? | (Factor) ]
//     (UnaryExp)     = { "-" }? (FactorPrefix)
//     (ExpExp)       = (UnaryExp) { { "^" }? (UnaryExp) }?
//     (MultExp)      = (ExpExp) { { "*" | "/" }? (ExpExp) }*
//     (AddExp)       = (MultExp) { { "+" | "-" }? (MultExp) }*
//     (EqualExp)     = (AddExp) { { "=" }? (AddExp) }*
//     (start)        = (EqualExp)

export const FIRST_FUNCTION = new TokenSet(TOKENS.Function);
export const FIRST_FACTOR = FIRST_FUNCTION.add(TOKENS.Variable | TOKENS.OpenParen);
export const FIRST_FACTOR_PREFIX = FIRST_FACTOR.add(TOKENS.Constant);
export const FIRST_EXP = FIRST_FACTOR_PREFIX.add(TOKENS.Minus);
export const FIRST_UNARY = FIRST_EXP;
export const FIRST_MULT = FIRST_UNARY;
export const FIRST_ADD = FIRST_UNARY;
export const FIRST_EQUAL = FIRST_UNARY;

// Precedence checks
export const IS_ADD = new TokenSet(TOKENS.Plus | TOKENS.Minus);
export const IS_MULT = new TokenSet(TOKENS.Multiply | TOKENS.Divide);
export const IS_EXP = new TokenSet(TOKENS.Exponent);
export const IS_EQUAL = new TokenSet(TOKENS.Equal);
export class ExpressionParser {
  tokenizer: Tokenizer = new Tokenizer();
  tokens: Token[] = [];
  currentToken: Token = new Token('', TOKENS.None);

  // Parse a string representation of an expression into a tree that can be
  // later evaluated.
  // Returns : The evaluatable expression tree.
  parse(input: string): MathExpression {
    this.tokens = this.tokenizer.tokenize(input);
    this.currentToken = new Token('', TOKENS.None);
    if (!this.next()) {
      throw new InvalidExpression('Cannot parse an empty function');
    }
    const expression = this.parseEqual();
    let leftover = '';
    while (this.currentToken.type !== TOKENS.EOF) {
      leftover += this.currentToken.value;
      this.next();
    }
    if (leftover !== '') {
      throw new TrailingTokens(`Trailing characters: ${leftover}`);
    }
    return expression;
  }

  parseEqual() {
    if (!this.check(FIRST_ADD)) {
      throw new InvalidSyntax('Invalid expression');
    }
    let exp = this.parseAdd();
    while (this.check(IS_EQUAL)) {
      const opType = this.currentToken.type;
      const opValue = this.currentToken.value;
      this.eat(opType);
      const expected = this.check(FIRST_ADD);
      let right = null;
      if (expected) {
        right = this.parseAdd();
      }
      if (!expected || !right) {
        throw new UnexpectedBehavior(`Expected an expression after = operator, got: ${this.currentToken.value}`);
      }
      if (opType !== TOKENS.Equal) {
        throw new UnexpectedBehavior(`Expected plus or minus, got: ${opValue}`);
      }
      exp = new EqualExpression(exp, right);
    }
    return exp;
  }

  parseAdd(): MathExpression {
    if (!this.check(FIRST_MULT)) {
      throw new InvalidSyntax('Invalid expression');
    }
    let exp = this.parseMult();
    while (this.check(IS_ADD)) {
      const opType = this.currentToken.type;
      const opValue = this.currentToken.value;
      this.eat(opType);
      const expected = this.check(FIRST_MULT);
      let right = null;
      if (expected) {
        right = this.parseMult();
      }
      if (!expected || !right) {
        throw new UnexpectedBehavior(`Expected an expression after + or - operator, got: ${this.currentToken.value}`);
      }
      switch (opType) {
        case TOKENS.Plus:
          exp = new AddExpression(exp, right);
          break;
        case TOKENS.Minus:
          exp = new SubtractExpression(exp, right);
          break;
        default:
          throw new UnexpectedBehavior(`Expected plus or minus, got: ${opValue}`);
      }
    }
    return exp;
  }

  parseMult(): MathExpression {
    if (!this.check(FIRST_EXP)) {
      throw new InvalidSyntax('Invalid expression');
    }
    let exp = this.parseExp();
    while (this.check(IS_MULT)) {
      const opType = this.currentToken.type;
      const opValue = this.currentToken.value;
      this.eat(opType);
      const expected = this.check(FIRST_EXP);
      let right = null;
      if (expected) {
        right = this.parseMult();
      }
      if (!expected || right === null) {
        throw new InvalidSyntax(`Expected an expression after * or / operator, got: ${opValue}`);
      }
      switch (opType) {
        case TOKENS.Multiply:
          exp = new MultiplyExpression(exp, right);
          break;
        case TOKENS.Divide:
          exp = new DivideExpression(exp, right);
          break;
        default:
          throw new UnexpectedBehavior(`Expected mult or divide, got: ${opValue}`);
      }
    }
    return exp;
  }

  parseExp(): MathExpression {
    if (!this.check(FIRST_UNARY)) {
      throw new InvalidSyntax('Invalid expression');
    }
    let exp = this.parseUnary();
    if (this.check(new TokenSet(TOKENS.Exponent))) {
      const opType = this.currentToken.type;
      this.eat(opType);
      if (!this.check(FIRST_UNARY)) {
        throw new InvalidSyntax('Expected an expression after ^ operator');
      }
      const right = this.parseUnary();
      switch (opType) {
        case TOKENS.Exponent:
          exp = new PowerExpression(exp, right);
          break;
        default:
          throw new UnexpectedBehavior(`Expected exponent, got: ${opType}`);
      }
    }
    return exp;
  }

  parseUnary(): MathExpression {
    let value;
    let negate = false;
    if (this.currentToken.type === TOKENS.Minus) {
      this.eat(TOKENS.Minus);
      negate = true;
    }
    const expected = this.check(FIRST_FACTOR_PREFIX);
    let exp = null;
    if (expected) {
      if (this.currentToken.type === TOKENS.Constant) {
        value = parseFloat(this.currentToken.value);
        if (negate) {
          value = -value;
          negate = false;
        }
        exp = new ConstantExpression(value);
        this.eat(TOKENS.Constant);
      }
      if (this.check(FIRST_FACTOR)) {
        if (exp === null) {
          exp = this.parseFactors();
        } else {
          exp = new MultiplyExpression(exp, this.parseFactors());
        }
      }
    }
    if (!expected || exp === null) {
      throw new InvalidSyntax(`Expected a function, variable or parenthesis after - or + but got : ${this.currentToken.value}`);
    }
    if (negate) {
      return new NegateExpression(exp);
    }
    return exp;
  }

  parseFactorPrefix(): MathExpression {
    let exp = null;
    if (this.currentToken.type === TOKENS.Constant) {
      exp = new ConstantExpression(parseFloat(this.currentToken.value));
      this.eat(TOKENS.Constant);
    }
    if (this.check(FIRST_FACTOR)) {
      if (exp === null) {
        return this.parseFactors();
      }
      return new MultiplyExpression(exp, this.parseFactors());
    }
    if (exp === null) {
      throw new InvalidExpression('no expression returned while parsing factor prefix');
    }
    return exp;
  }

  parseFactors(): MathExpression {
    let right;
    let found = true;
    const factors = [];
    while (found) {
      right = null;
      switch (this.currentToken.type) {
        case TOKENS.Variable:
          factors.push(new VariableExpression(this.currentToken.value));
          this.eat(TOKENS.Variable);
          break;
        case TOKENS.Function:
          factors.push(this.parseFunction());
          break;
        case TOKENS.OpenParen:
          this.eat(TOKENS.OpenParen);
          factors.push(this.parseAdd());
          this.eat(TOKENS.CloseParen);
          break;
        default:
          throw new UnexpectedBehavior(`Unexpected token in Factor: ${this.currentToken.type}`);
      }
      found = this.check(FIRST_FACTOR);
    }
    if (factors.length === 0) {
      throw new InvalidExpression('No factors');
    }

    let exp = null;
    if (this.check(IS_EXP)) {
      const opType = this.currentToken.type;
      this.eat(opType);
      if (!this.check(FIRST_UNARY)) {
        throw new InvalidSyntax('Expected an expression after ^ operator');
      }
      right = this.parseUnary();
      exp = new PowerExpression(factors[factors.length - 1], right);
    }

    if (factors.length === 1) {
      return exp || factors[0];
    }
    while (factors.length > 0) {
      if (exp == null) {
        exp = factors.shift();
      }
      const next = factors.shift();
      if (!next || !exp) {
        throw new OutOfTokens('missing expected factors for multiplication');
      }
      exp = new MultiplyExpression(exp, next);
    }
    if (!exp) {
      throw new InvalidExpression('parseFactors returned invalid expression');
    }
    return exp;
  }

  parseFunction(): MathExpression {
    const opFn = this.currentToken.value;
    this.eat(this.currentToken.type);
    this.eat(TOKENS.OpenParen);
    const exp = this.parseAdd();
    this.eat(TOKENS.CloseParen);
    const func: FunctionExpression = this.tokenizer.functions[opFn];
    if (func === undefined) {
      throw new UnexpectedBehavior(`Unknown Function type: ${opFn}`);
    }
    return new (func as any)(exp);
  }

  // Assign the next token in the queue to @currentToken
  // Returns true if there are still more tokens in the queue, or
  // false if we have looked at all available tokens already
  next(): boolean {
    if (this.currentToken.type === TOKENS.EOF) {
      throw new OutOfTokens('Parsed beyond the end of the expression');
    }
    const newToken = this.tokens.shift();
    if (newToken) {
      this.currentToken = newToken;
      return this.currentToken.type !== TOKENS.EOF;
    }
    return false;
  }

  // Assign the next token in the queue to @currentToken if the @currentToken's
  // type matches that of the specified parameter.  If the @currentToken's
  // type does not match the parameter, throw a syntax exception
  //
  // `type` The type that your syntax expects @currentToken to be
  eat(type: number) {
    if (this.currentToken.type !== type) {
      throw new InvalidSyntax(`Missing: ${type}`);
    }
    return this.next();
  }

  // Check if the @currentToken is a member of a set Token types
  //
  // `tokens` The set of Token types to check against
  // Returns true if the @currentToken's type is in the set or false if it is not
  check(tokens: TokenSet) {
    return tokens.contains(this.currentToken.type);
  }
}
