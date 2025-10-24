from dataclasses import dataclass


@dataclass
class Label:
    class_id: int
    center_y: float
    center_x: float
    height: float
    width: float
