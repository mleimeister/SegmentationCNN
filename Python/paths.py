# encoding: utf-8
"""
    Define path locations and helpful functions
"""

import os

audio_path = '../Audio'
beats_path = '../Audio/beats'
mls_path = '../Audio/mls'
annotations_path =  '../Data/salami-data-public/annotations/'

def remove_suffix(filename):
     return os.path.splitext(os.path.basename(filename))[0]

def mls_path(audio_filename):
     return os.path.join(audio_path, remove_suffix(audio_filename) + '.mls.npy')

