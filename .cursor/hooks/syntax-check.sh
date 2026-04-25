#!/bin/bash
# Syntax-check any edited Python file in p300_analysis/
# Receives hook JSON on stdin; returns allow with optional context.

input=$(cat)

# Extract the file path from the hook input
file_path=$(echo "$input" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    # postToolUse / afterFileEdit provides tool_input
    path = (d.get('tool_input') or {}).get('path', '')
    print(path)
except Exception:
    print('')
" 2>/dev/null)

if [[ -z "$file_path" ]]; then
    echo '{"additional_context": ""}'
    exit 0
fi

# Only check .py files
if [[ "$file_path" != *.py ]]; then
    echo '{"additional_context": ""}'
    exit 0
fi

# Run syntax check
result=$(python3 -c "
import ast, sys
try:
    with open('$file_path', 'r') as f:
        src = f.read()
    ast.parse(src)
    print('OK')
except SyntaxError as e:
    print(f'SYNTAX ERROR line {e.lineno}: {e.msg}')
except FileNotFoundError:
    print('FILE NOT FOUND')
" 2>&1)

if [[ "$result" == "OK" ]]; then
    echo '{"additional_context": ""}'
else
    echo "{\"additional_context\": \"Syntax check failed for $file_path: $result — fix before running the app.\"}"
fi

exit 0
