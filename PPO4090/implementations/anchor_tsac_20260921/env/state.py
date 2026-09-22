"""分配账本的最小可重算记录。状态编号是持久化契约。"""
from dataclasses import dataclass, asdict

IDLE, PENDING, ALLOCATED, SKIPPED = range(4)
STATUS_NAMES = {IDLE: "idle", PENDING: "pending", ALLOCATED: "allocated", SKIPPED: "skipped"}


@dataclass(frozen=True)
class Allocation:
    beam_id: int
    status: str
    start: int = -1
    length: int = 0
    power_total_w: float = 0.0

    def to_dict(self):
        return asdict(self)

