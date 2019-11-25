import { BinaryTreeTidier } from '../mathy/math/treeNode'
import {
  MathExpression,
  BinaryExpression,
  VariableExpression
} from '../mathy/math/expressions'
import { ExpressionParser } from '../mathy/math/parser'
import React, { Fragment } from 'react'
import { uuidv4 } from '../mathy/math/util'
const { Paper, Text, Line, Rect } = require('react-raphael')

/**
 * Recursive Tree Rendering directive
 */

/** Draw a raphael line */
export function drawLine(x1: number, y1: number, x2: number, y2: number) {
  return <Line x1={x1} y1={y1} x2={x2} y2={y2} attr={{ stroke: '#aaa', 'stroke-width': 4 }} />
}

function drawText(
  x: number,
  y: number,
  text: string,
  fontSize: number,
  background = 'rgb(220,220,220)'
) {
  const renderText = text.length > 5 ? text.replace(/[\(\s]/g, '') : text.replace(/[\+*/-]/g, '')
  const charWidth = 18
  const charHeight = 32
  const rectWidth = Math.max(renderText.length, 2) * charWidth
  return (
    <Fragment>
      <Rect
        r={4}
        x={x - rectWidth / 2}
        y={y - charHeight / 2}
        width={rectWidth}
        height={charHeight}
        attr={{
          fill: background,
          'stroke-width': 0
        }}
      />
      <Text
        x={x}
        y={y}
        text={text}
        attr={{
          fill: 'rgba(50,50,50,0.75)',
          'font-size': `${fontSize}px`,
          'font-weight': 'bold'
        }}
      />
    </Fragment>
  )
}

interface MathTreeProps {
  input: string
  width?: number
  height?: number
  center?: boolean
}
interface MathTreeState {
  text: string
  layout: BinaryTreeTidier | null
  parser: ExpressionParser | null
  expression: MathExpression | null
}

export class MathTree extends React.Component<MathTreeProps, MathTreeState> {
  static getDerivedStateFromProps(props: MathTreeProps, state: MathTreeState) {
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
  state: MathTreeState = {
    parser: new ExpressionParser(),
    layout: new BinaryTreeTidier(),
    expression: null,
    text: ''
  }

  render() {
    const expression: MathExpression | null = this.state.expression
    const layout: BinaryTreeTidier | null = this.state.layout
    if (layout === null || expression === null) {
      return null
    }
    const options = {
      nodeSize: 20,
      spacing: 70,
      padding: 25
    }
    const measure = layout.layout(expression, options.spacing)
    const root = expression.getRoot()
    // console.log(`expression: ${expression}`)
    // console.log(`root: ${root.x}, ${root.y}`)
    // console.log(`measurement:`, measure)
    const doublePadding = options.padding * 2
    const viewX = measure.minX - options.padding
    const viewY = measure.minY - options.padding
    const viewW = measure.maxX + doublePadding
    const viewH = measure.maxY + doublePadding
    const compWidth = this.props.width || measure.maxX - measure.minX + doublePadding
    const compHeight = this.props.height || measure.maxY - measure.minY + doublePadding
    const viewBox = `${viewX},${viewY},${viewW},${viewH}`
    const html: any[] = []
    // TODO: TS was saying expression is never type. I don't know why. :( :/
    ;(expression as MathExpression).visitPostorder((node: any) => {
      const offsetX = this.props.center ? compWidth / 2 : Math.abs(measure.minX) + options.padding
      const offsetY = Math.abs(measure.minY) + options.padding
      let color = 'rgb(180,200,255)'
      let value = node.toString()
      if (node instanceof BinaryExpression) {
        color = 'rgb(230,230,230)'
        value = node.getName()
      } else if (node instanceof VariableExpression) {
        color = 'rgb(150,250,150)'
      }
      if (node === node.getRoot()) {
        color = 'rgb(250,220,200)'
      }
      if (node.parent) {
        html.push(
          drawLine(
            node.x + offsetX,
            node.y + offsetY,
            node.parent.x + offsetX,
            node.parent.y + offsetY
          )
        )
      }
      html.push(drawText(node.x + offsetX, node.y + offsetY, value, options.nodeSize, color))
    })
    return (
      <Paper width={compWidth} height={compHeight} viewBox={viewBox}>
        {html.map((jsx: any, i) => (
          <React.Fragment key={uuidv4()}>{jsx}</React.Fragment>
        ))}
      </Paper>
    )
  }
}
