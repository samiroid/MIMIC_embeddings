#!/bin/bash
N="$1"
delay="$2"
EXEC="$3"

if [ -z $1 ]; then
        echo "Missing N"
        exit 0
fi

if [ -z $2 ]; then
        echo "Missing delay"
        exit 0
fi

if [ -z "$3" ]; then
        echo "Missing exec"
        exit 0
fi

while [ $N -gt 0 ] 
do
    $EXEC
    echo "$N runs left"
    sleep $delay
    clear
    N=$(( $N - 1 ))
done