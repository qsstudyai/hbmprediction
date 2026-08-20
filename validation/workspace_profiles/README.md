# Workspace profiles

Profiler-derived kernel workspace entries are content-addressed and keyed by
family, fusion variant, local shape, parallel layout, MindSpore/CANN version,
and hardware. No profile is reused across a runtime major version. Until a key
is present, the evaluator reports the graph formula as a fallback and keeps
the capability experimental.
