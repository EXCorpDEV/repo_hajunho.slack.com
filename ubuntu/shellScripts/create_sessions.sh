#!/bin/bash

# Define an array of session names
session_names=("apt" "rich" "proxy" "kernel" "tensorflow" "blockchain" "logs" "yocto" "mysql" "sodoc" "logs2" "logs3" "ncdu" "fastapi")

# Loop through the array and create a screen session for each name
for name in "${session_names[@]}"
do
  screen -dmS "$name"
done

