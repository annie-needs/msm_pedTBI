#!/bin/bash

SOURCE_DIR="scenario29"

for i in {30..56}; do 
    NEW_DIR="scenario${i}"
    echo "creating $NEW_DIR"
    cp -r "$SOURCE_DIR" "$NEW_DIR"
done

echo "Done copying scenarios 30 thru 56"