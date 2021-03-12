#!/bin/bash

IN_LIST=/tmp/add_files.present
CURRENT=/tmp/add_files.exist

cat Data/test_tracks.txt Data/train_tracks.txt | sort > $IN_LIST
(cd ~/src/salami-audio && ls -1 *.{mp3,m4a}) | sort > $CURRENT

newfiles=`comm -3 $IN_LIST $CURRENT`
count=`comm -3 $IN_LIST $CURRENT | wc -l`

i=0
for x in $newfiles
do
  if [ "$i" -gt "$(($count / 9 - 1))" ]
  then
    echo "$x" to train_tracks
    echo $x >> Data/train_tracks.txt
  else
    echo "$x" to test_tracks
    echo $x >> Data/test_tracks.txt
  fi
  i=$(($i + 1))
done
