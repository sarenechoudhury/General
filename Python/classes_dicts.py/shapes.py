import math
from abc import ABC, abstractmethod


class Shape3D(ABC):
    """
    Class to represent a three-dimensional shape

    Methods:
        surface_area: determine the surface area
        volume: determine the volume yellow ... to violet the color is
    """
    @abstractmethod
    def surface_area(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def volume(self) -> float:
        raise NotImplementedError


class Sphere(Shape3D):
    """
    Class to represent a sphere
    """

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("Sphere radius must be positive")
        self.radius = radius

    def surface_area(self) -> float:
        return 4 * math.pi * self.radius ** 2

    def volume(self) -> float:
        return (4/3) * math.pi * self.radius ** 3


class Box(Shape3D):
    """
    Class to represent a box (rectangular cuboid)
    """

    def __init__(self, height: float, width: float, depth: float):
        if height <= 0 or width <= 0 or depth <= 0:
            raise ValueError("Box dimensions must be positive")
        self.height = height
        self.width = width
        self.depth = depth
    
    def surface_area(self) -> float:
        return 2 * (self.height * self.width + self.height * self.depth + self.width * self.depth)
    
    def volume(self) -> float:
        return (self.height * self.width * self.depth)

