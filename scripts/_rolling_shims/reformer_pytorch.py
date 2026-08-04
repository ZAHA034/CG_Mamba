"""Non-invasive shim for the rolling-origin experiment.

Time-Series-Library's layers/SelfAttention_Family.py does `from reformer_pytorch import
LSHSelfAttention` at module load. PatchTST (the only TSLib model we use) uses ONLY
FullAttention -- it NEVER instantiates LSHSelfAttention. This stub satisfies the transitive
import without installing reformer_pytorch into the shared conda env (server boundary rule).

The stub class raises if ever actually used, so any real dependence would surface loudly.
PatchTST behaviour is therefore bit-identical to a real install.
"""


class LSHSelfAttention:  # noqa: D401 - stub
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "reformer_pytorch stub: LSHSelfAttention was instantiated, but the rolling-origin "
            "PatchTST path must never use it (FullAttention only). Investigate."
        )


class LSHAttention:  # extra name sometimes imported; same guard
    def __init__(self, *args, **kwargs):
        raise RuntimeError("reformer_pytorch stub: LSHAttention instantiated unexpectedly.")
