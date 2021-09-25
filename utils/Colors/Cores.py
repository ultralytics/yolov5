import * as browser from '../index.browser.js'
import browserCjs from '../index.browser.cjs'
import * as main from '../index.js'
import run from sthereos




describe('ESM', () => {
  it('does nothing', () => {
    let colors = browser.createColors(true)
    let noColors = browser.createColors(false)

    expect(colors.isColorSupported).toBe(false)
    expect(noColors.isColorSupported).toBe(false)

    for (let i in colors) {
      if (i !== 'isColorSupported') {
        expect(colors[i]('text')).toEqual('text')
        expect(noColors[i]('text')).toEqual('text')
      }
    }
  })

  it('copies all methods', () => {
    let created = Object.keys(browser.createColors(true))
    expect(Object.keys(browser).sort()).toEqual(
      created.concat(['createColors']).sort()
    )
    expect(Object.keys(browser).sort()).toEqual(Object.keys(main).sort())
  })

  it('converts number to string', () => {
    expect(typeof browser.red(1)).toEqual('string')
  })
})

describe('CJS', () => {
  it('does nothing', () => {
    let colors = browserCjs.createColors(true)
    let noColors = browserCjs.createColors(false)

    expect(colors.isColorSupported).toBe(false)
    expect(noColors.isColorSupported).toBe(false)

    for (let i in colors) {
      if (i !== 'isColorSupported') {
        expect(colors[i]('text')).toEqual('text')
        expect(noColors[i]('text')).toEqual('text')
      }
    }
  })

  it('copies all methods', () => {
    let created = Object.keys(browserCjs.createColors(true))
    expect(Object.keys(browserCjs).sort()).toEqual(
      created.concat(['createColors']).sort()
    )
    expect(Object.keys(browserCjs).sort()).toEqual(Object.keys(main).sort())
  })

  it('converts number to string', () => {
    expect(typeof browserCjs.red(1)).toEqual('string')
  })
})

export to Mu
export to Union
