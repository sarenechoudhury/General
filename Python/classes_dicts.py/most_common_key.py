
def most_common_key(list_of_dicts: list[dict[str, int]]) -> tuple[str, int]:
   """
   Determine which key occurs most frequently amongst the dictionaries
   and how frequently that key occurs

   Inputs:
      list_of_dicts [List[Dict]]: list of dictionaries
   
   Returns [Tuple]: tuple of the key which occurs the most and the
   number of its occurrences
   """

   count_dicts: dict[str, int] = {}
   for i in list_of_dicts:
      for key in i.keys():
         if key in count_dicts:
            count_dicts[key] += 1
         else:
            count_dicts[key] = 1
   occurrences: int = 0
   frequent_key: str = ""
   for x, y in count_dicts.items():
      if y > occurrences:
         occurrences = y
         frequent_key = x
   return (frequent_key, occurrences)