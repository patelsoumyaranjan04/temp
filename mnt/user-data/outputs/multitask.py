"""Root-level re-export so `from multitask import MultiTaskPerceptionModel` works
for the autograder, while the real implementation lives in models/multitask.py.
"""

from models.multitask import MultiTaskPerceptionModel

__all__ = ["MultiTaskPerceptionModel"]
