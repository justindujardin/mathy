import * as React from 'react'
import { MathExpression } from '../mathy/math/expressions'
import { ExpressionParser } from '../mathy/math/parser'

interface MathTextProps {
  input: string
}
interface MathTextState {
  text: string
  parser: ExpressionParser | null
  expression: MathExpression | null
}

export class MathText extends React.Component<MathTextProps, MathTextState> {
  static getDerivedStateFromProps(props: MathTextProps, state: MathTextState) {
    if (!state) {
      return state
    }
    const { parser } = state
    if (props.input && props.input !== state.text && parser) {
      return {
        text: props.input,
        expression: parser.parse(props.input)
      }
    }
    return state
  }
  state = {
    parser: null,
    expression: null,
    text: ''
  }

  componentDidMount() {
    this.setState({
      parser: new ExpressionParser()
    })
  }

  render() {
    const expression: MathExpression | null = this.state.expression
    if (expression === null) {
      return null
    }
    return <code>{`${expression}`}</code>
  }
}
