"""adversaria: calibration-tracked multi-agent debate.

Agents carry persistent calibration scores (Brier) from their track record.
Votes are weighted by calibration, not volume. A novelty critic scores each
proposal on argument novelty (embedding dissimilarity to prior proposals) so
repeated angles are discounted even when correct.
"""

from .agents import AgentProfile, SimulatedAgent, Proposal, Task
from .calibration import CalibrationTracker
from .novelty import NoveltyCritic, novelty_scores
from .debate import Debate, DebateResult

__all__ = [
    "AgentProfile",
    "SimulatedAgent",
    "Proposal",
    "Task",
    "CalibrationTracker",
    "NoveltyCritic",
    "novelty_scores",
    "Debate",
    "DebateResult",
]

__version__ = "0.1.0"
