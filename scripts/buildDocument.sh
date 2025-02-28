#!/usr/bin/env bash

DOCFILE="${1:-glossary.md}"
TOCFILE=entries/_toc.md

echo "Building table of contents"

printf "## Table of contents\n\n">"$TOCFILE"
for ENTRY in entries/*; do
	#exclude intro, outro and toc file from indexing
	if [[ "$ENTRY" != entries/_* ]]; then
		# echo "$ENTRY"
		HEADER=$(head -1 "$ENTRY")
		NAME="${HEADER:3}"
		TAG=${NAME,,}
		TAG=${TAG// /-}

		LONGNAME=$(grep "==> " "$ENTRY")
		if [[ -n "$LONGNAME" ]]; then
			LONGNAME="${LONGNAME:4}"
			echo " * [${LONGNAME} (${NAME})](#${TAG})" >> "$TOCFILE"
		else
			echo " * [${NAME}](#${TAG})" >> "$TOCFILE"
		fi
	fi
done

echo "Building document"

cat entries/_intro.md entries/_toc.md>"$DOCFILE"
printf "\n***\n\n">>"$DOCFILE"
for ENTRY in entries/*; do
	#exclude intro, outro and toc file from indexing
	if [[ "$ENTRY" != entries/_* ]]; then
		#splice out the TOC fullname reference before writing to document
		grep -v "==> " "$ENTRY">>"$DOCFILE"
	fi
done
printf "\n***\n\n">>"$DOCFILE"
cat entries/_outro.md>>"$DOCFILE"

