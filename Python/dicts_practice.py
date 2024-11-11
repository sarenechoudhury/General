import csv
import math

RADIUS = 6371000.0


def sighting_to_string(sighting):
    """
    Create a string description of a UFO sighting in the form:
        Sighting [ID]: [shape] object, [duration] seconds

    Input:
        sighting [Dict]: a UFO sighting

    Returns [str]: A description of the sighting in the format above.
    """

    return f"Sighting {sighting['ID']}: {sighting['Shape']} \
        object, {sighting['Duration']} seconds"    



def create_year_dict(sightings):
    """
    Given a list of UFO sightings, create a dictionary that maps each UFO 
        sighting ID to the year in which that sighting occurred.

    Inputs:
        sightings [List[Dict]]: list of UFO sightings

    Returns [Dict]: Dictionary that maps UFO sighting IDs to years.  
    """
    year_dict = {}
    for sighting in sightings:
        sighting_id = sighting['ID']
        year = sighting['Year']
        year_dict[sighting_id] = year
    return year_dict


def find_long_sightings(sightings, min_length):
    """
    Given a list of UFO sightings, create a list of sighting IDs for 
        sighting that lasted at least as long as some minimum number of seconds. 

    Inputs:
        sightings [List[Dict]]: list of UFO sightings
        min_length [int]: the minimum length of a sighting in seconds

    Returns [List[str]]: A list of sighting IDs that lasted at least as long
        as min_length.  
    """
    long_stgs = []
    for i in sightings:
        if i['Duration'] >= min_length:
            long_stgs.append(i['ID'])
    return long_stgs


def count_shapes(sightings):
    """
    Given a list of UFO sightings, create a dictionary that maps a UFO 
        sighting shape to the number of sightings of that shape. 

    Inputs:
        sightings [List[Dict]]: list of UFO sightings

    Returns [Dict]: Dictionary that maps UFO sighting shapes to the number of 
        sightings of that shape.  
    """
    count = {}
    for i in sightings:
        shape = i['Shape']
        if shape in count:
            count[shape] += 1
        else:
            count[shape] = 1
    return count


def most_common_shape(sightings):
    """
    Given a list of UFO sightings, determine the most commonly occurring 
        UFO sighting shape and the number of times it occurred. 
    
    Inputs:
        sightings [List[Dict]]: list of UFO sightings

    Returns [Tuple[str, int]]: The most commonly occuring shape and
        the number of times it occurred. 
    """
    count = {}
    for i in sightings:
        shape = i['Shape']
        if shape in count:
            count[shape] += 1
        else:
            count[shape] = 1
    
    most_common = max(count, key=count.get)
    count = count[most_common]
    
    return most_common, count


def sightings_by_year(sightings):
    """
    Given a list of UFO sightings, create a dictionary that maps a year 
        to a list of the UFO sighting dictionaries from that year.
    
    Inputs:
        sightings [List[Dict]]: list of UFO sightings

    Returns [Dict]: A dictionary that maps years to a list of UFO 
        sighting dictionaries.
    """
    by_year = {}
    for i in sightings:
        year = i['Year']
        if year in by_year:
            by_year[year].append(i)
        else:
            by_year[year] = [i]
    return by_year

def haversine(lat1, lon1, lat2, lon2):
    """
    Given two locations of latitude and longitude, calculates
    the distance between them using the Haversine Formula

    Inputs:
        lat1 [float]: latitude in degrees
        lon1 [float]: longitute in degrees
        lat2 [float]: latitude in degrees
        lon2 [float]: longitute in degrees

    Returns [float]: the distance between the two 
    locations in meters
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * \
        math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance_meters = RADIUS * c
    
    return distance_meters

def close_sightings(sightings, max_distance):
    """
    Given a list of UFO sightings, find the sightings that occurred within 
        a given distance of each other. 

    Inputs:
        sightings [List[Dict]]: list of UFO sightings
        max_distance [float]: the maximum distance between two locations

    Returns [List[Tuple[str, str]]]: A list of UFO sighting ID tuples where
        the sightings are within max_distance of each other. 
    """
    close_pairs = []
    for i, sighting1 in enumerate(sightings):
        for j, sighting2 in enumerate(sightings[i + 1:], start=i + 1):
            id1, id2 = sighting1['ID'], sighting2['ID']
            lat1, lon1 = math.radians(sighting1['Latitude']), \
                math.radians(sighting1['Longitude'])
            lat2, lon2 = math.radians(sighting2['Latitude']), \
                math.radians(sighting2['Longitude'])
            distance = haversine(lat1, lon1, lat2, lon2)
            if distance <= max_distance:
                # Ensure each pair is added only once
                close_pairs.append((min(id1, id2), max(id1, id2)))
    return close_pairs
    