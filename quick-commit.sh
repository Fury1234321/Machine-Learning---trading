#!/bin/bash
# Quick commit script - run with a commit message
# Usage: ./quick-commit.sh "Your commit message"

if [ $# -eq 0 ]; then
  echo "Please provide a commit message."
  echo "Usage: ./quick-commit.sh \"Your commit message\""
  exit 1
fi

git add .
git commit -m "$1"
git push origin main

echo "Changes committed and pushed successfully!" 