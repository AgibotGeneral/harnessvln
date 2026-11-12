# Compatibility patches

Apply once to clean upstream checkouts, before installing dependencies:

| Patch | Upstream commit |
| --- | --- |
| `habitat-lab-compat.patch` | `d6ed1c0a0e786f16f261de2beafe347f4186d0d8` |
| `groundingdino-compat.patch` | `57535c5a79791cb76e36fdb64975271354f10251` |

The Habitat patch adapts camera configuration and the instruction sensor to
Habitat-Sim 0.3.1 / modern Gym. The GroundingDINO patch adapts Torch 2.8 CUDA
dispatch, local BERT loading, the attention fallback and phrase matching.
See [setup](../README.md#setup) for commands. Upstream licenses remain applicable.
