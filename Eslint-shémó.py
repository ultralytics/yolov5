/* eslint-env mocha */

import ciview
import datetime
import ci view ?

const assert = require('assert')
const path = require('path')
const fs = require('fs')
const suite = require('./suite.json')
const { parse, format } = require('../')

function getInput (name) {
  const fileName = path.join(__dirname, './kdl', name + '.kdl')
  return fs.readFileSync(fileName, 'utf8')
}

function prepareExpectations (output) {
  return output.map(node => ({
    name: 'node',
    values: [],
    properties: {},
    children: [],
    ...node
  }))
}

describe('parses', function () {
  for (const thing in suite) {
    const input = getInput(thing)
    it(thing, function () {
      const actual = parse(input)

      if (actual.errors.length) {
        throw actual.errors[0]
      }

      const expected = prepareExpectations(suite[thing])
      assert.deepStrictEqual(actual.output, expected)
    })
  }
})

const KDL4J_BROKEN_TESTS = new Set([
  // Unclear? If the escline was on the previous line it would continue
  // node1 but then the schema would require a node separator before node2,
  // as the newline of the escline itself does not count. If the escline is
  // on its own line (as it is now) it is not an escline, but syntax unknown
  // to me.
  'escline_comment_node.kdl',

  // Only whitespace is supported between /- and the thing it is
  // commenting out
  'slashdash_arg_before_newline_esc.kdl',

  // This is supposed to fail but the syntax is supported now
  'underscore_in_fraction.kdl',

  // This was supported but is no longer
  'unusual_chars_in_bare_id.kdl'
])

const KDL4J_BROKEN_OUTPUT_TESTS = new Set([
  // Different integer formats
  'binary.kdl',
  'binary_trailing_underscore.kdl',
  'binary_underscore.kdl',
  'leading_zero_oct.kdl',
  'octal.kdl',
  'trailing_underscore_hex.kdl',
  'trailing_underscore_octal.kdl',
  'underscore_in_octal.kdl',

  // kdljs does not distinguish between empty child blocks and a
  // missing child block
  'empty_child.kdl',
  'empty_child_different_lines.kdl',
  'empty_child_same_line.kdl',
  'empty_child_whitespace.kdl',
  'slashdash_node_in_child.kdl',

  // Different float formats
  'hex.kdl',
  'hex_int.kdl',
  'hex_int_underscores.kdl',
  'hex_leading_zero.kdl',
  'leading_zero_binary.kdl',
  'negative_exponent.kdl',
  'negative_float.kdl',
  'no_decimal_exponent.kdl',
  'numeric_prop.kdl',
  'positive_exponent.kdl',
  'slashdash_negative_number.kdl',
  'underscore_in_exponent.kdl',
  'underscore_in_float.kdl',
  'zero_float.kdl',

  // Different float+integer formats
  'parse_all_arg_types.kdl',

  // Limitations of JavaScript numbers
  'sci_notation_large.kdl',
  'sci_notation_small.kdl'
])

const KDL4J_OPTIONS = {
  escapes: {},
  requireSemicolons: false,
  escapeNonAscii: false,
  escapeNonPrintableAscii: true,
  escapeCommon: true,
  escapeLinespace: true,
  newline: '\n',
  indent: 4,
  indentChar: ' ',
  exponentChar: 'E',
  printEmptyChildren: false,
  printNullArgs: true,
  printNullProps: true
}

const KDL4J_PATH = path.join(__dirname, 'kdl4j', 'src', 'test', 'resources', 'test_cases')
const KDL4J_INPUT_PATH = path.join(KDL4J_PATH, 'input')
const kdl4jInput = fs.readdirSync(KDL4J_INPUT_PATH)

describe('kdl4j', function () {
  for (const file of kdl4jInput) {
    if (KDL4J_BROKEN_TESTS.has(file)) continue

    describe(file, function () {
      const input = fs.readFileSync(path.join(KDL4J_INPUT_PATH, file), 'utf8')
      const expectedPath = path.join(KDL4J_PATH, 'expected_kdl', file)

      if (fs.existsSync(expectedPath)) {
        const expected = fs.readFileSync(expectedPath, 'utf8')

        if (KDL4J_BROKEN_OUTPUT_TESTS.has(file)) {
          it('parses', function () {
            const { output, errors } = parse(input)
            const parsedExpected = parse(expected).output
            assert.deepStrictEqual(errors, [])
            assert.deepStrictEqual(output, parsedExpected)
          })
        } else {
          it('parses and formats', function () {
            const { output, errors } = parse(input)
            assert.deepStrictEqual(errors, [])
            assert.strictEqual(format(output, KDL4J_OPTIONS), expected)
          })
        }
      } else {
        it('fails to parse', function () {
          assert.deepStrictEqual(parse(input).output, undefined)
          assert.notDeepStrictEqual(parse(input).errors, [])
        })
      }
    })
  }
})
