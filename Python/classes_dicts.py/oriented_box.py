import math
from abc import ABC, abstractmethod
from shapes import Box


class OrientedBox(Box):
    """
    Class to represent a box (rectangular cuboid) that is facing with one of its
        sides against a plane and that has a density
    """

    def __init__(self, height: float, width: float, depth: float, face: str, density: float):
        super().__init__(height, width, depth)
        if face not in ["HW", "HD", "WD"]:
            raise ValueError("Invalid face parameter. Must be one of 'HW', 'HD', or 'WD'.")
        if density <= 0:
            raise ValueError("Density must be positive.")
        self.face = face
        self.density = density

    def pressure(self) -> float:
      if self.face == "HW":
         area = self.height * self.width
      elif self.face == "HD":
         area = self.height * self.depth
      elif self.face == "WD":
         area = self.width * self.depth
      else:
         raise ValueError("Invalid face parameter. Must be one of 'HW', 'HD', or 'WD'.")
        
      weight = self.density * self.volume()
      pressure = weight / area
      return pressure

