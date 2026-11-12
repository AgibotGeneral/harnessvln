"""Immutable coordinate types with explicit unit conversions.

Coordinates distinguish normalized image positions, pixels, metres and map
cells. Tuple conversion methods support simulator and planner interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class NormalizedPixel:
    """Pixel in [0, 1] image coordinates, origin top-left."""

    u: float
    v: float

    def __post_init__(self):
        if not (0.0 <= float(self.u) <= 1.0 and 0.0 <= float(self.v) <= 1.0):
            raise ValueError(f"NormalizedPixel out of range: u={self.u} v={self.v}")

    def to_image(self, width: int, height: int) -> "ImagePixel":
        return ImagePixel(
            x=int(round(self.u * (int(width) - 1))),
            y=int(round(self.v * (int(height) - 1))),
            width=int(width),
            height=int(height),
        )

    def as_tuple(self) -> Tuple[float, float]:
        return (float(self.u), float(self.v))


@dataclass(frozen=True)
class ImagePixel:
    """Integer pixel with the image size it was measured in.

    Carrying width/height is the point: a bbox from a 1000x1000 VLM canvas and
    a pixel in a 480x640 depth frame are not the same coordinate, and today
    nothing in the signature says which one you hold.
    """

    x: int
    y: int
    width: int
    height: int

    def to_normalized(self) -> NormalizedPixel:
        return NormalizedPixel(
            u=float(self.x) / float(self.width - 1),
            v=float(self.y) / float(self.height - 1),
        )

    def rescaled_to(self, width: int, height: int) -> "ImagePixel":
        """Same relative location expressed in a differently sized image."""
        return self.to_normalized().to_image(width, height)

    def clamped(self) -> "ImagePixel":
        return ImagePixel(
            x=min(max(int(self.x), 0), int(self.width) - 1),
            y=min(max(int(self.y), 0), int(self.height) - 1),
            width=int(self.width),
            height=int(self.height),
        )

    @property
    def in_bounds(self) -> bool:
        return 0 <= self.x < self.width and 0 <= self.y < self.height

    def as_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))


@dataclass(frozen=True)
class CameraPoint:
    """Metres in the camera frame."""

    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))


@dataclass(frozen=True)
class WorldPoint:
    """Metres in the world frame, ground plane only (habitat x/z)."""

    x: float
    z: float

    def as_tuple(self) -> Tuple[float, float]:
        return (float(self.x), float(self.z))

    @classmethod
    def from_tuple(cls, xz) -> "WorldPoint":
        x, z = xz
        return cls(x=float(x), z=float(z))


@dataclass(frozen=True)
class MapCell:
    """Semantic-map cell index plus the resolution that defines it."""

    row: int
    col: int
    resolution_cm: int

    def as_tuple(self) -> Tuple[int, int]:
        return (int(self.row), int(self.col))

    @classmethod
    def from_tuple(cls, rowcol, resolution_cm: int) -> "MapCell":
        row, col = rowcol
        return cls(row=int(row), col=int(col), resolution_cm=int(resolution_cm))

    def clamped(self, shape) -> "MapCell":
        """Clamp into ``shape`` = (rows, cols)."""
        rows, cols = int(shape[0]), int(shape[1])
        return MapCell(
            row=min(max(int(self.row), 0), rows - 1),
            col=min(max(int(self.col), 0), cols - 1),
            resolution_cm=int(self.resolution_cm),
        )


@dataclass(frozen=True)
class RelativeYaw:
    """Yaw offset in degrees, normalised to [0, 360)."""

    degrees: float

    @property
    def normalized(self) -> float:
        return float(self.degrees) % 360.0

    def quadrant(self) -> int:
        """Index of the nearest 90-degree view: 0=front, 1=left, 2=behind, 3=right.

        Matches the legacy ``target_angle // 90`` on the exact multiples the
        legacy code produces (0/90/180/270) and rounds to the nearest view for
        anything in between, instead of truncating.
        """
        return int(round(self.normalized / 90.0)) % 4

    def as_float(self) -> float:
        return float(self.degrees)
