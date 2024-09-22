#!/bin/bash

# Check if the samples directory exists
if [ -d "./samples" ]; then
    # Remove all items under the samples directory
    rm -rf ./samples/*
    echo "All items under ./samples have been removed."
else
    echo "Directory ./samples does not exist."
fi