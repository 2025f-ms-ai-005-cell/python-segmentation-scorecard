"""Offline binary segmentation evaluation on supplied reference/prediction pairs."""
import argparse
import html
import json
from pathlib import Path


def mask_shape(mask):
    if not isinstance(mask, list) or not mask or not isinstance(mask[0], list) or not mask[0]:
        raise ValueError('nonempty rectangular masks required')
    w = len(mask[0])
    if w * len(mask) > 65536:
        raise ValueError('mask exceeds 65536 pixels')
    if any(not isinstance(row, list) or len(row) != w for row in mask):
        raise ValueError('ragged mask')
    if any(type(v) is not int or v not in (0, 1) for row in mask for v in row):
        raise ValueError('binary integer pixels required')
    return w, len(mask)


def metrics(counts):
    tp, fp, fn, tn = (counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    def ratio(a, b):
        return a / b if b else None
    return dict(iou=ratio(tp, tp + fp + fn), dice=ratio(2 * tp, 2 * tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                accuracy=ratio(tp + tn, tp + fp + fn + tn))


def compare(reference, prediction):
    shape = mask_shape(reference)
    if mask_shape(prediction) != shape:
        raise ValueError('reference and prediction shapes differ')
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    labels = []
    for rr, pp in zip(reference, prediction):
        row = []
        for r, p in zip(rr, pp):
            key = ('tp' if p else 'fn') if r else ('fp' if p else 'tn')
            counts[key] += 1
            row.append(key)
        labels.append(row)
    return dict(width=shape[0], height=shape[1], counts=counts, metrics=metrics(counts), labels=labels)


def evaluate(data):
    if not isinstance(data, dict) or not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('input needs a nonempty cases array')
    if len(data['cases']) > 100:
        raise ValueError('at most 100 cases supported')
    cases, names = [], set()
    totals = dict(tp=0, fp=0, fn=0, tn=0)
    for c in data['cases']:
        if not isinstance(c, dict) or not {'name', 'reference', 'prediction'} <= c.keys():
            raise ValueError('each case needs name, reference and prediction')
        name = c['name']
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or name in names:
            raise ValueError('case names must be unique nonempty strings, up to 80 characters')
        names.add(name)
        result = compare(c['reference'], c['prediction'])
        cases.append(dict(name=name, **result))
        for k in totals:
            totals[k] += result['counts'][k]
    macro = {}
    for k in cases[0]['metrics']:
        values = [c['metrics'][k] for c in cases if c['metrics'][k] is not None]
        macro[k] = dict(value=sum(values) / len(values) if values else None, defined_cases=len(values))
    return dict(cases=cases, pooled_counts=totals, micro=metrics(totals), macro=macro,
                undefined_policy='zero denominator -> null; excluded from macro with defined case count')


def graphic(case):
    palette = dict(tp='#34d399', fp='#fb7185', fn='#fbbf24', tn='#1e293b')
    size = min(60, 600 / max(case['width'], case['height']))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">',
             '<rect width="1600" height="1000" fill="#091426"/>',
             '<g font-family="Arial,sans-serif" fill="#e2e8f0">',
             '<text x="70" y="85" font-size="44" font-weight="bold">SEGMENTATION SCORECARD</text>',
             '<text x="70" y="135" font-size="23">Computed pixel errors • Synthetic example • No trained-model claim</text>',
             f'<text x="70" y="185" font-size="24">{html.escape(case["name"])}</text>']
    for y, row in enumerate(case['labels']):
        for x, label in enumerate(row):
            parts.append(f'<rect x="{70+x*size}" y="{240+y*size}" width="{size-1}" height="{size-1}" fill="{palette[label]}"/>')
    for i, (key, color) in enumerate(palette.items()):
        y = 245 + 60 * i
        parts.append(f'<rect x="820" y="{y}" width="25" height="25" fill="{color}"/><text x="865" y="{y+22}" font-size="24">{key.upper()}: {case["counts"][key]} pixels</text>')
    for i, (key, val) in enumerate(case['metrics'].items()):
        value = f'{val:.4f}' if val is not None else 'undefined'
        parts.append(f'<text x="820" y="{560+i*48}" font-size="24">{key}: {value}</text>')
    parts.append('<text x="70" y="920" font-size="22">TP = correct foreground • FP = extra foreground • FN = missed foreground • TN = background</text></g></svg>')
    return ''.join(parts)


def write_report(result, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    rows = []
    for i, case in enumerate(result['cases']):
        filename = f'case-{i+1:03d}.svg'
        (out / filename).write_text(graphic(case), encoding='utf-8')
        rows.append(f'<section><h2>{html.escape(case["name"])}</h2><img width="100%" src="{filename}" alt="Computed confusion map"></section>')
    page = '<!doctype html><meta charset="utf-8"><title>Segmentation Scorecard</title><style>body{max-width:1100px;margin:40px auto;font:18px Arial;background:#091426;color:#eee}section{margin:32px 0}</style><h1>Segmentation Scorecard</h1><p>Supplied masks only. Undefined denominators are null in JSON. No model inference.</p>' + ''.join(rows)
    (out / 'index.html').write_text(page, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.input.stat().st_size > 2000000:
            raise ValueError('input exceeds 2 MB')
        result = evaluate(json.loads(args.input.read_text(encoding='utf-8')))
        write_report(result, args.out)
        print(f'{len(result["cases"])} cases evaluated; micro IoU={result["micro"]["iou"]}; {args.out}')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
