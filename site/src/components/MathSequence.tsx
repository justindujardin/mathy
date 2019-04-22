import React, { useEffect, useState, useCallback } from 'react'
import { MathTree } from '../components/MathTree'
import { MathText } from '../components/MathText'

export interface MathSequenceProps {
  steps: string[]
  stepInterval: number
  showText: boolean
}
export interface MathSequenceState {
  current: number
  timer: NodeJS.Timer | number
}
export function MathSequence(props: MathSequenceProps) {
  const [current, setCurrent] = useState(0)
  const [, setTimer] = useState<NodeJS.Timer | number>(-1)
  const { steps, stepInterval = 1500, showText = true } = props
  // state = {
  //   current: 0,
  //   timer: -1
  // }
  const tick = useCallback(
    () => {
      let newValue = current + 1
      if (current >= steps.length) {
        newValue = 0
      }
      setCurrent(newValue)
    },
    [current, steps]
  )
  useEffect(function componentDidMount() {
    setTimer(setInterval(tick, stepInterval))
  }, [])
  return (
    <>
      {showText ? (
        <p>
          <MathText input={steps[current]} />
        </p>
      ) : null}
      <MathTree width={640} height={480} center={true} input={steps[current]} />
    </>
  )
}
