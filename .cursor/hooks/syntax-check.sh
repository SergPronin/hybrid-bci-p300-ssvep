#!/bin/bash
# afterFileEdit hook:
#   1. Syntax-check edited Python files in p300_analysis/
#   2. Trigger incremental graph update in background (CodeGraph mark-dirty + GitNexus)

input=$(cat)

file_path=$(echo "$input" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    path = (d.get('tool_input') or {}).get('path', '')
    print(path)
except Exception:
    print('')
" 2>/dev/null)

context_parts=()

# --- Syntax check (only .py files) ---
if [[ "$file_path" == *.py ]]; then
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

    if [[ "$result" != "OK" ]]; then
        context_parts+=("⚠ Syntax: $file_path — $result")
    fi
fi

# --- Graph update in background (non-blocking) ---
if [[ "$file_path" == *.py ]]; then
    # CodeGraph: mark dirty immediately (< 1ms), sync will happen lazily
    codegraph mark-dirty . > /dev/null 2>&1 &

    # GitNexus: incremental analyze in background (~0.8s, non-blocking for Cursor)
    npx gitnexus analyze --skip-git > /dev/null 2>&1 &
fi

# --- Output ---
if [[ ${#context_parts[@]} -gt 0 ]]; then
    msg=$(printf '%s\n' "${context_parts[@]}")
    echo "{\"additional_context\": \"$msg\"}"
else
    echo '{"additional_context": ""}'
fi

exit 0
