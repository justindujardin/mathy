import * as React from 'react'
import styled from 'react-emotion'
import { transparentize } from 'polished'
import { Link } from 'gatsby'

import { heights, dimensions, colors } from '../styles/variables'
import Container from './Container'

const StyledHeader = styled.header`
  height: ${heights.header}px;
  padding: 0 ${dimensions.containerPadding}rem;
  background-color: ${colors.brand};
  color: ${transparentize(0.3, colors.white)};
`

const HeaderInner = styled(Container)`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: start;
  height: 100%;
`

const HomepageLink = styled(Link)`
  color: ${colors.white};
  font-size: 4rem;
  font-weight: 300;
  display: block;

  &:hover,
  &:focus {
    text-decoration: none;
  }
`

const SubHeader = styled('p')`
  color: ${colors.gray};
  font-size: 1rem;
  font-weight: 400;
`

interface HeaderProps {
  title: string
  text: string
}

export const Header: React.SFC<HeaderProps> = ({ title, text }) => (
  <StyledHeader>
    <HeaderInner>
      <HomepageLink to="/">{title}</HomepageLink>
      {text && <SubHeader>{text}</SubHeader>}
    </HeaderInner>
  </StyledHeader>
)

export default Header
