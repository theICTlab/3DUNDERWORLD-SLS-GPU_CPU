#!/bin/sh

# Rename all jpgs in folder given by $1
a=0
for i in $1/*.JPG 
do
    new=$(printf "$1/%04d.jpg" "$a")
    cp "$i" ${new}
    let a=a+1;
done

