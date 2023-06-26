#!/bin/bash

if [[ "$1" = "" ]]; then
  echo Usage: $0 FILE
  exit 1
fi

if [[ ! -e "$1" ]]; then
  echo Error: File $1 does not exist.
  exit 2
fi

xpdf -remote filewatch "openFile($1)" &
XPDFPID=$!

while [[ -e /proc/$XPDFPID ]]; do
  inotifywait `dirname $1` | grep "MODIFY $1"
  [[ $? = 0 ]] && sleep 0.5 && xpdf -remote filewatch reload
done
