# Architecture

## Design goal

CrystalFlow crystallizes proven pure subroutines, not conversations. A warm workflow hit must be
able to complete without a model invocation inside the plugin and without any external side effect.

## Components

```text
Dify tool adapter
  ├── strict JSON input parsing
  ├── progressive runner
  │     ├── active-crystal lookup
  │     ├── selected Dify LLM fallback
  │     ├── opt-in bounded example retention
  │     ├── duplicate/conflict detection
  │     └── model proposal -> exact engine tests
  ├── CrystalFlow engine
  │     ├── closed IR validator
  │     ├── input/output schema validator
  │     ├── gas/depth/size limits
  │     └── deterministic evaluator
  └── registry
        ├── namespace catalog
        ├── immutable version records
        ├── mutable active-version alias
        └── best-effort aggregate telemetry
```

The engine has no Dify dependency. The registry uses Dify's documented KV operations:
`exist(key)`, `get(key)`, `set(key, bytes)`, and `delete(key)`. It also accepts simpler embedded
stores whose `get` returns `None` for a missing key.

`ProgressiveService` is framework-neutral: the Dify adapter injects fallback and builder callables.
That boundary lets the core prove that a warm hit returns before either callable is invoked.

## Trust boundaries

The program, description, tests, crystal name, task description, model output, retained examples,
and execution input are untrusted. Validation occurs before storage and again before execution. A
stored record is not trusted merely because it came from Dify KV; hashes and structure are checked
when read.

The plugin process and the Dify-provided storage/session object are trusted. Crystal programs never
receive either object.

## Storage layout

Keys are namespaced and versioned. Version records are immutable and content-addressed. An index and
active alias are mutable because Dify KV does not provide enumeration.

Progressive learning adds one bounded canonical state record per namespace and task key. It stores
deduplicated input/output examples, their hashes, candidate state, build-attempt counters, and a
quarantine flag. No progressive record is created while learning is disabled.

The Dify adapter binds its internal crystal name to a hash of the configured task key and task
description. Editing task semantics therefore creates a separate state/candidate identity instead
of silently executing an older crystal. The original description is sent to the selected model but
is not retained solely for this binding.

The registry and progressive service serialize mutations inside one Python worker.
Writers in different plugin worker processes can still race while updating the index, aliases, or
aggregate counters because the documented storage API has no transaction or compare-and-swap
primitive. Execution remains deterministic, and immutable version records avoid in-place program
corruption, but HA administration requires a transactional registry.

## Failure contract

Execution fails closed:

- Missing active version: `miss`
- Input schema rejection: `invalid_input`
- Retired crystal: `disabled`
- Corrupt record, engine mismatch, or unexpected runtime failure: `error`

Each non-hit sets `fallback_required: true`. The surrounding Dify workflow owns the fallback policy.
The low-level `execute_crystal` tool never invokes a model. `progressive_run` instead owns an
explicit cold path: it invokes only the model selected on that node, returns the model's JSON result,
and optionally observes it for learning. Model or builder failures fail closed and never activate
unvalidated code.

## Receipts

A successful execution receipt hashes a canonical object containing:

- engine version
- program hash
- crystal name and version
- canonical input
- canonical output

Receipts prove replay equivalence; they are not signatures and do not authenticate a caller.
