# Architecture

## Design goal

CrystalFlow removes repeated model work without caching stale answers. It supports two deliberately
separate artifacts:

- A compute crystal is a proven pure JSON subroutine.
- A route crystal is an exact request fingerprint mapped to one administrator-approved, read-only,
  answer-ready tool call.

A warm compute hit completes without a model or external side effect. A warm route hit completes
without the Agent model but intentionally invokes its live tool.

## Components

```text
Dify integrations
  ├── Adaptive Agent Strategy
  │     ├── exact route lookup before the LLM
  │     ├── normal Function Calling cold path
  │     ├── approved tool-contract validation
  │     ├── successful-route observation and conflict quarantine
  │     └── direct live tool invocation on a hit
  └── tool adapters
        ├── strict JSON input parsing
        └── progressive compute runner
              ├── active-compute-crystal lookup
              ├── selected Dify LLM fallback
              ├── opt-in bounded example retention
              └── model proposal -> exact engine tests

Framework-neutral core
  ├── adaptive route store
  │     ├── normalized request fingerprints
  │     ├── app/instruction/tool contract binding
  │     ├── bounded frequency and conflict state
  │     └── aggregate route telemetry
  ├── CrystalFlow engine
  │     ├── closed IR validator
  │     ├── input/output schema validator
  │     ├── gas/depth/size limits
  │     └── deterministic evaluator
  └── compute registry
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

`AdaptiveRouteStore` is also framework-neutral. It validates and returns a tool plan but never
invokes it. The Agent Strategy resolves that plan against the currently configured Dify tool,
merges only current runtime/form parameters, invokes it, and records a hit only after successful
completion.

## Trust boundaries

The program, description, tests, crystal name, task description, model output, retained examples,
route observations, tool arguments, tool contracts, and execution input are untrusted. Validation
occurs before storage and again before execution. A stored record is not trusted merely because it
came from Dify KV; hashes and structure are checked when read.

The plugin process and the Dify-provided storage/session object are trusted. Crystal programs never
receive either object. Route crystals cannot invoke tools themselves; only the strategy adapter can.

## Storage layout

Keys are namespaced and versioned. Version records are immutable and content-addressed. An index and
active alias are mutable because Dify KV does not provide enumeration.

Progressive learning adds one bounded canonical state record per namespace and task key. It stores
deduplicated input/output examples, their hashes, candidate state, build-attempt counters, and a
quarantine flag. No progressive record is created while learning is disabled.

Adaptive routing uses a separate versioned keyspace. Its request key hashes the normalized query,
explicit routing context, app/instruction scope, and route schema version. The record stores the
approved tool identity and arguments, its contract hash, a repetition counter, conflict/quarantine
state, and aggregate hit estimates. The original query, conversation history, tool output,
retrieved documents, credentials, and runtime/form parameters are not stored as separate route
fields. Model-supplied tool arguments are necessarily replayable and can themselves contain query
text, so only administrator-approved arguments suitable for workspace-local persistence are
eligible.

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

The Adaptive Agent treats missing, changed, conflicting, corrupt, or failing routes as cold-path
fallbacks in the same request. It observes only successful calls to the explicitly selected safe
tools. A warm path must return the answer-ready tool output directly; otherwise a post-tool model
would still be required and the route would not be a zero-Agent-LLM hit.

## Receipts

A successful execution receipt hashes a canonical object containing:

- engine version
- program hash
- crystal name and version
- canonical input
- canonical output

Receipts prove replay equivalence; they are not signatures and do not authenticate a caller.
