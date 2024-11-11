from tree import TreeNode


def get_paths(t: TreeNode, weight: int) -> list[list[int]]:
   """
   Creates a list of all paths in a given tree that sum to
   a target weight

   Inputs:
      t [TreeNode]: a tree
      weight [int]: target weight
   
   Returns [List[List[int]]]: list of all paths in the tree
   that sum to the given weight
   """
   assert weight >= 0
   summed_paths: list[list[int]] = []
   def count_paths_helper(node: TreeNode, current_weight: int, current_path: list[int]) -> None:
      """
      Helper function to recursively count paths
      with weights summing to the target weight.

      Inputs:
         node [TreeNode]: The current node being processed.
         current_weight [int]: The current path weight.

      Returns:
         List[int]: paths with weights
         summing to the target weight.
      """
      if not node:
         return

      current_weight += node.value
      current_path.append(node.value)

      if current_weight == weight:
         summed_paths.append(list(current_path))
      
      for child in node.children:
         count_paths_helper(child, current_weight, current_path.copy())

      current_path.pop()

   count_paths_helper(t, 0, [])

   return summed_paths