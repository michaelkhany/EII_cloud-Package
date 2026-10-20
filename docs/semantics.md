# Semantics reference

## Time

Every dataset has a strictly increasing event-time axis. Internally all temporal reasoning
runs on float seconds (datetime axes, any pandas resolution) or on the raw numeric value
(e.g. "time window" counters). Windows are given in the same representation
(`"5min"`, `pd.Timedelta`, or a number). JSON output uses the user's own labels
(ISO-8601 with offset, or numbers).

## Three-valued truth

| value | meaning |
|---|---|
| `TRUE` (1) | sufficient evidence satisfies the condition |
| `FALSE` (0) | sufficient *valid* evidence contradicts the condition |
| `UNKNOWN` (-1) | required evidence is unavailable or insufficient |

Kleene tables: `A AND B` is FALSE if either is FALSE, TRUE if both are TRUE, otherwise
UNKNOWN. `A OR B` is TRUE if either is TRUE, FALSE if both are FALSE, otherwise UNKNOWN.
`NOT UNKNOWN` is UNKNOWN.

Sources of UNKNOWN:

* a measurement is NaN (`signal`, `outside_envelope`, detectors);
* an envelope has no support at that sample (insufficient reference data, unknown regime);
* an event source is outside its declared coverage (`EventLog.coverage`);
* a windowed operator's window reaches before the first or after the last sample and the
  window contains no TRUE sample;
* a regime cannot be determined because its context signal is missing.

## Leaf conditions

| YAML | meaning |
|---|---|
| `{signal: s, op: ">=", value: v}` | numeric or categorical comparison; `op` in `> >= < <= == != in "not in"` |
| `{outside_envelope: {parameter: p, envelope: e}}` | NOT contained in envelope `e` |
| `{missing_when_expected: p}` | `p` is NaN and the acquisition schedule expects it (`dataset.expected`) |
| `{event: {event_type: t, source: s, tolerance: d}}` | an event of type `t` maps to the nearest sample within `tolerance` (default half the median step) |
| `{context: regime}` | regime active (from `ContextDefinition`) |
| `{finding: {detector: id}}` / `{finding: {category: c}}` | truth series of a detector output |
| `{rule: id}` | truth series of another rule (dependencies are topologically ordered; cycles raise `RuleCycleError`) |
| `{contradicts_context: {parameter: p, expectation: id}}` | a contextual expectation is violated |

## Temporal operators

With `w` a window and `t` the current sample time:

| operator | TRUE at `t` iff |
|---|---|
| `before {condition: A, next: B, window: w}` | `A(t)` and `B` TRUE at some `s ∈ [t, t+w]` |
| `after {condition: A, prior: B, window: w}` | `A(t)` and `B` TRUE at some `s ∈ [t-w, t]` |
| `within {condition: A, other: B, window: w}` | `A(t)` and `B` TRUE at some `s ∈ [t-w, t+w]` |
| `during {condition: A, interval: B}` | the whole maximal TRUE run of `A` containing `t` lies in TRUE samples of `B` |
| `overlaps {condition: A, interval: B}` | the maximal TRUE run of `A` containing `t` has at least one TRUE sample of `B` |
| `persists_for {condition: A, window: w}` | `A` TRUE at every sample in `[t-w, t]` and the window lies inside the axis (any FALSE in the window gives FALSE) |
| `count_within {condition: A, window: w, min_count: k}` | at least `k` rising edges of `A` in `[t-w, t]` |

Interval boundaries are inclusive. `before`/`within` need lookahead `w`; `during`/`overlaps`
need the complete run. In streaming mode a finding is *provisional* until the event-time
watermark passes `end_time + lookahead`.

## Findings

A finding is a maximal TRUE run of a rule or detector truth series (at least `min_duration`
samples). Its `finding_id` is a SHA-256 prefix of
`(asset_id, parameter_ids, category, detector_id, rule_id, start_time, end_time)` and is
therefore identical across batch/streaming runs and across machines.

Categories: A1 `value_deviation`, A2 `missing_when_expected`, A3 `abnormal_flatline`,
A4 `context_contradiction`, A5 `temporal_instability`, A6 `cross_signal_inconsistency`,
A7 `composite_evidence`.

## Envelope metadata

`result.envelopes[name]` contains `method`, `params`, `reference` (population, hashes,
fitted tables), `reference_id`, `support_fraction`, `warnings` and `coverage` (overall and
per regime). The arrays `lower/center/upper/support` are included so a chart can be redrawn
from the JSON alone.

## Annotation JSON

`annotations_document(result, dataset, parameter)` returns schema `1.0` with, per
annotation: `annotation_id` (= `ann-<finding_id>`), category, severity, truth status,
start/end, anchor time/value, text, rule/detector, compact evidence (matched event and
finding ids, observed values), context snapshot, envelope metadata and versions. The same
ids are set as `gid` on the Matplotlib artists and therefore appear as `id="..."` in SVG.
