import * as React from 'react'
// @ts-ignore
import { MDXProvider } from '@mdx-js/react'
import Page from './Page'
import Container from './Container'
import IndexLayout from '../layouts'
import styled from 'react-emotion'
import { transparentize } from 'polished'
import { heights, dimensions, colors } from '../styles/variables'

const StyledH1 = styled.h1`
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  font-size: 3em;
  font-weight: 400;
  color: ${colors.brand};
`
const StyledA = styled.a`
  font-weight: 300;
  color: ${transparentize(0.1, colors.brand)};
`

const StyledCode = styled.code`
  padding: 1.5em;
  background-color: ${transparentize(0.9, colors.brand)};
  border-radius: 0.5em;
  white-space: normal;
  color: ${colors.brand};
`

const StyledParagraph = styled.p`
  font-size: 1.2em;
  color: ${transparentize(0.1, colors.black)};
  margin-bottom: 1.25em;
`

// import * as DesignSystem from 'your-design-system'
// @ts-ignore
export default function Layout({ children }) {
  return (
    <IndexLayout>
      <Page>
        <Container>
          <MDXProvider
            components={{
              h1: StyledH1,
              a: StyledA,
              code: StyledCode,
              p: StyledParagraph
            }}
          >
            {children}
          </MDXProvider>
        </Container>
      </Page>
    </IndexLayout>
  )
}
