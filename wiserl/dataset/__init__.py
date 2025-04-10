from functools import lru_cache
# Register dataset classes here
# from .robomimic_dataset import RobomimicDataset # Awaiting Numba Release
from wiserl.dataset.cliff_walking_dataset import (
    CliffWalkingComparisonDataset,
    CliffWalkingOfflineDataset,
)
from wiserl.dataset.d4rl_dataset import D4RLOfflineDataset
from wiserl.dataset.ipl_dataset import IPLComparisonOfflineDataset
from wiserl.dataset.metaworld_dataset import MetaworldComparisonOfflineDataset
from wiserl.dataset.metaworld_offline_dataset import (
    MetaworldComparisonDataset,
    MetaworldOfflineDataset,
)
from wiserl.dataset.mismatched_mujoco_dataset import (
    MismatchedComparisonDataset,
    MismatchedOfflineDataset,
)

from .replay_buffer import ReplayBuffer

try:
    from .robomimic_dataset import RobomimicDataset
except ImportError:
    print("Warning: Could not import RobomimicDataset")


@lru_cache(maxsize=None)
def load_dataset(**kwargs):
    """Load dataset with caching support.
    
    Args:
        **kwargs: Additional arguments including 'class' parameter
    Returns:
        Dataset instance
    """
    cls = kwargs.pop("class")
    ds = globals()[cls](
        # we dont need observation and action space for the dataset
        observation_space=None,
        action_space=None,
        **kwargs
    )
    return ds