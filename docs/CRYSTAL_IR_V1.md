# Crystal IR v1

Crystal IR is a closed JSON expression language for pure, deterministic transformations. A program
is data, not source code.

## Program envelope

```json
{
  "version": 1,
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "expression": {"op": "literal", "value": null}
}
```

`version` and `expression` are required. The two schemas are optional but strongly recommended.
Unknown envelope keys, schema keywords, expression operators, and operator fields are rejected.

The supported JSON Schema subset is:

- `type`: `null`, `boolean`, `integer`, `number`, `string`, `array`, or `object`
- `properties`, `required`, and `additionalProperties`
- `items` and `enum`
- `minimum` and `maximum`
- `minLength`, `maxLength`, `minItems`, `maxItems`, `minProperties`, and `maxProperties`

## Value expressions

| Form | Result |
|---|---|
| `{"op":"literal","value":VALUE}` | A JSON literal |
| `{"op":"input"}` | The complete input value |
| `{"op":"input","name":"field"}` | A top-level input object field |
| `{"op":"var","name":"item"}` | A lexically scoped collection variable |
| `{"op":"object","fields":{"key":EXPR}}` | An object |
| `{"op":"array","items":[EXPR,...]}` | An array |
| `{"op":"get","target":EXPR,"key":EXPR}` | Object field, array index, or string index |
| `{"op":"get","target":EXPR,"key":EXPR,"default":EXPR}` | Lookup with lazy default |

Array and string indexes may be negative. Object keys must be strings. A missing lookup without a
default is an execution error.

## Operators using `args`

These forms are `{"op":"OP","args":[EXPR,...]}`.

| Operator | Arity | Behavior |
|---|---:|---|
| `add`, `mul` | 2+ | Numeric addition/product |
| `sub`, `div`, `mod`, `pow` | 2 | Numeric operation |
| `round` | 1 or 2 | Round a number, optionally to an integer number of digits |
| `min`, `max` | 1+ | Numeric minimum/maximum |
| `abs` | 1 | Numeric absolute value |
| `concat` | 0+ | Concatenate strings |
| `lower`, `upper`, `strip` | 1 | String transformation |
| `replace` | 3 | Source, old text, new text |
| `split` | 2 | Source and nonempty separator |
| `join` | 2 | Separator and array of strings |
| `length` | 1 | String, array, or object length |
| `slice` | 2 or 3 | Target, start, optional end |
| `eq`, `ne`, `lt`, `lte`, `gt`, `gte` | 2 | Equality or ordered comparison |
| `and`, `or` | 1+ | Lazy Boolean logic; operands must be Booleans |
| `not` | 1 | Boolean negation |
| `if` | 3 | Lazy condition, true branch, false branch |
| `coalesce` | 1+ | First non-null value, evaluated lazily |

Ordered comparisons accept two numbers or two strings. Numbers must be finite. Integer results are
limited in size, powers are bounded, and divide/modulo by zero is an error. `round` uses
round-half-to-even. Use integer minor units (for example, cents) for financial rules.

## Collection operators

Map:

```json
{
  "op": "map",
  "collection": {"op": "input", "name": "items"},
  "var": "item",
  "index": "i",
  "body": {"op": "var", "name": "item"}
}
```

`index` is optional. `filter` has the same shape and requires its body to return a Boolean. `sort`
has the same shape; its body returns a null, Boolean, number, or string key, and it accepts optional
`"descending": true`.

Sum an evaluated numeric array with:

```json
{"op":"sum","collection":EXPR}
```

Variables exist only in the collection body. Recursion and unbounded loops do not exist.

## Default resource limits

| Resource | Limit |
|---|---:|
| Program AST nodes | 1,000 |
| Program/value depth | 32 |
| Items in any collection | 1,000 |
| Dynamic expression evaluations | 100,000 |
| Canonical input | 65,536 bytes |
| Canonical output, generated string, or aggregate intermediate | 262,144 bytes |
| Integer magnitude | 4,096 bits |
| Absolute power exponent | 1,000 |
| Absolute rounding digits | 100 |

Limits are enforced during static validation and execution. A limit failure is deterministic and
does not partially return a value.

## Tests

`crystallize` accepts an array:

```json
[
  {
    "name": "basic",
    "input": {"x": 2, "y": 3},
    "expected": {"total": 5}
  }
]
```

Every test is executed through the same schema validator and interpreter used by
`execute_crystal`. Expected and actual outputs are compared as canonical JSON. At least one and at
most 100 tests are accepted. Include boundary, empty, zero, negative, and invalid-domain cases as
appropriate; invalid inputs should be excluded by `input_schema`.

## Determinism notes

- JSON is parsed strictly: duplicate keys and non-finite constants are rejected.
- Object keys are ordered during canonical serialization.
- `-0.0` canonicalizes as `0`; finite float spelling is normalized.
- There is no clock, random source, implicit timezone, locale, I/O, or external state. Supply such
  values explicitly as inputs when they are part of a rule.
- Python/Unicode behavior is part of `engine_version`. An engine-version mismatch fails closed.

See [`invoice_total.program.json`](../examples/invoice_total.program.json) and
[`sla_classification.program.json`](../examples/sla_classification.program.json).
