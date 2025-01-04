#!/bin/bash

DIRECTORY="./data"
TRAIN="train_data.csv"
TEST="test_data.csv"

if [ -e "$DIRECTORY/$TRAIN" ] && [ -e "$DIRECTORY/$TEST" ]; then
  ./build/{EXECUTABLE}
else
  python prepare.py
  ./build/{EXECUTABLE}
fi