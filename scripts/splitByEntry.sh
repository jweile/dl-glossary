#!/usr/bin/env bash

DOCFILE="${1:-README.md}"

#make sure entries folders exists and is empty
mkdir -p entries
rm entries/entry*


#split file by entries (which start with '##' headers)
RX='^## '
NUMENTRIES=$(grep -c "$RX" README.md)
csplit -f entries/entry README.md "/$RX/" "{$((NUMENTRIES-1))}"

#derive names for entries according to their headings and rename the files accordingly
for ENTRY in entries/*; do
	HEADER=$(head -1 "$ENTRY")
	if [[ "$HEADER" == "# "* ]]; then
		# echo "Intro!"
		mv -v "$ENTRY" entries/_intro.md
	elif [[ "$HEADER" == "## Table of Contents" ]]; then
		# echo "TOC!"
		mv -v "$ENTRY" entries/_toc.md
	elif [[ "$HEADER" == "## Resources" ]]; then
		# echo "Outro!"
		mv -v "$ENTRY" entries/_outro.md
	else
		NAME="${HEADER:3}"
		TAG=${NAME,,}
		TAG=${TAG// /-}
		# echo "Section: $NAME -> $TAG"
		mv -v "$ENTRY" "entries/${TAG}.md"
	fi
done

