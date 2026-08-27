# Origins

The project began as a discrete invariant-discovery benchmark over a known transformation
chain. That prototype extracted byte, nibble, entropy, parity, endpoint, difference, and
representation features, then ranked hand-written hypotheses by stability, separation, and
strength.

Its deepest contribution was the pressure loop:

```text
trajectory → observables → collision → discriminator → invariant test → repaired representation
```

The prototype was useful but guided: each named invariant was registered manually. The
foundational engine in this repository preserves the loop while replacing named hypotheses
with an automatic expression grammar, independent collision holdouts, transformation suites,
group transfer, dimensional safeguards, iterative repair, and explicit unresolved-pair
reporting.

The conceptual continuity is intentional. The implementation is new and domain-neutral.
