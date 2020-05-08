import numpy as np
import os
import copy
import pickle
import heapq

"""Common Function"""

def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None

"""For Final Preprocessing Step"""
def get_grade_map():
    """
    Defines a mapping of Fontainebleau grades to integer values
    """
    grade_map = {
        '6A': 0,
        '6A+': 1,
        '6B': 2,
        '6B+': 3,
        '6C': 4,
        '6C+': 5,
        '7A': 6,
        '7A+': 7,
        '7B': 8,
        '7B+': 9,
        '7C': 10,
        '7C+': 11,
        '8A': 12,
        '8A+': 13,
        '8B': 14,
        '8B+': 15,
    }
    return grade_map

def get_grade_map_new():
    """
    Defines a mapping of Fontainebleau grades to integer values
    """
    grade_map = {
        '6B': 0,
        '6B+': 1,
        '6C': 2,
        '6C+': 3,
        '7A': 4,
        '7A+': 5,
        '7B': 6,
        '7B+': 7,
        '7C': 8,
        '7C+': 9,
        '8A': 10,
        '8A+': 11,
        '8B': 12,
        '8B+': 13,
    }
    return grade_map