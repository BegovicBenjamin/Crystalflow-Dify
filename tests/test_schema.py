from __future__ import annotations

import unittest

from crystalflow.errors import (
    InputValidationError,
    OutputValidationError,
    SchemaDefinitionError,
)
from crystalflow.schema import (
    is_valid,
    validate_instance,
    validate_output,
    validate_schema,
)


class SchemaDefinitionTests(unittest.TestCase):
    def test_useful_object_and_array_subset(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 10},
                "scores": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 10},
                    "minItems": 1,
                    "maxItems": 3,
                },
            },
            "required": ["name", "scores"],
            "additionalProperties": False,
        }
        validate_schema(schema)
        validate_instance({"name": "Ada", "scores": [8, 9.5]}, schema)

    def test_unknown_keywords_are_not_silently_ignored(self) -> None:
        with self.assertRaises(SchemaDefinitionError) as caught:
            validate_schema({"type": "string", "minLenght": 2})
        self.assertEqual(caught.exception.code, "UNKNOWN_SCHEMA_KEY")

    def test_incompatible_and_inverted_bounds_are_rejected(self) -> None:
        with self.assertRaises(SchemaDefinitionError):
            validate_schema({"type": "number", "minLength": 1})
        with self.assertRaises(SchemaDefinitionError):
            validate_schema({"type": "array", "minItems": 3, "maxItems": 2})

    def test_duplicate_required_and_enum_are_rejected(self) -> None:
        with self.assertRaises(SchemaDefinitionError) as caught:
            validate_schema({"type": "object", "required": ["x", "x"]})
        self.assertEqual(caught.exception.code, "DUPLICATE_REQUIRED")
        with self.assertRaises(SchemaDefinitionError) as caught:
            validate_schema({"enum": [1, 1.0]})
        self.assertEqual(caught.exception.code, "DUPLICATE_ENUM_VALUE")


class InstanceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "type": "object",
            "properties": {
                "kind": {"enum": ["a", "b"]},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["kind", "count"],
            "additionalProperties": False,
        }

    def test_required_additional_enum_and_type(self) -> None:
        validate_instance({"kind": "a", "count": 2}, self.schema)
        cases = [
            ({"kind": "a"}, "MISSING_REQUIRED"),
            ({"kind": "c", "count": 2}, "ENUM_MISMATCH"),
            ({"kind": "a", "count": True}, "TYPE_MISMATCH"),
            ({"kind": "a", "count": 2, "extra": 1}, "ADDITIONAL_PROPERTY"),
        ]
        for value, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(InputValidationError) as caught:
                    validate_instance(value, self.schema)
                self.assertEqual(caught.exception.code, code)

    def test_additional_properties_can_have_a_schema(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string", "maxLength": 3},
        }
        validate_instance({"x": "yes"}, schema)
        with self.assertRaises(InputValidationError):
            validate_instance({"x": "long"}, schema)

    def test_output_uses_a_distinct_error_type(self) -> None:
        with self.assertRaises(OutputValidationError):
            validate_output("not a number", {"type": "number"})

    def test_is_valid_is_a_convenience_predicate(self) -> None:
        self.assertTrue(is_valid({"kind": "b", "count": 1}, self.schema))
        self.assertFalse(is_valid({"kind": "b", "count": 0}, self.schema))


if __name__ == "__main__":
    unittest.main()
