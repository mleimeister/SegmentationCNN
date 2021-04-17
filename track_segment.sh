#!/bin/bash

cd Python
mkdir -p  ~/src/salami-data-public/annotations/$1/parsed
python ./track_segmentation.py ~/src/salami-audio/$1.* ~/src/salami-data-public/annotations/$1/parsed/predicted.txt
