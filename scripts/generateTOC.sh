#!/usr/bin/env bash
#Find all headings and convert to TOC entries
while IFS= read -r LINE; do
  CROP="${LINE:3}"
  LOWER="${CROP,,}"
  DASHED="${LOWER// /-}"
  echo " * [${CROP}](#${DASHED})"
done < <(grep "## " README.md|sort)
