"""Entry point: python3 main.py

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input/), forecasts each
with model.forecast, and writes <id>.out JSON to $MAKEFUN_OUTPUT_DIR
(else /output/). Pure standard library, no network.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import forecast  # noqa: E402


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith('.in'))
    if not names:
        print('no .in files in', in_dir, file=sys.stderr)
        return 1
    for name in names:
        stem = name[:-3]
        with open(os.path.join(in_dir, name)) as f:
            case = json.load(f)
        rows = forecast(case)
        doc = {'hours': [{'hour': i + 2, 'q': [round(v, 2) for v in rows[i]]}
                         for i in range(23)]}
        out_path = os.path.join(out_dir, stem + '.out')
        with open(out_path, 'w') as f:
            json.dump(doc, f)
        size = os.path.getsize(out_path)
        if size > 65536:
            print(f'{stem}: WARNING output {size} bytes exceeds 65536',
                  file=sys.stderr)
            return 1
        print(f'{stem}: wrote {len(doc["hours"])} hours ({size} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
