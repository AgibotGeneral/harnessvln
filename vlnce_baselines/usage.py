"""Single-model usage schema shared by workers and reporting tools."""
COUNTERS = ('calls', 'input_tokens', 'output_tokens', 'total_tokens')


def empty_usage():
    return dict.fromkeys(COUNTERS, 0)


def merge_usage(records):
    totals = empty_usage()
    names = set()
    for record in records:
        if not isinstance(record.get('model'), dict):
            raise ValueError('Unsupported usage schema: expected a model counter dictionary')
        for key in COUNTERS:
            totals[key] += record['model'].get(key, 0) or 0
        if record.get('model_name'):
            names.add(record['model_name'])
    if len(names) > 1:
        raise ValueError('Cannot merge workers using different models: ' + ', '.join(sorted(names)))
    return dict(model_name=next(iter(names), None), model=totals,
                total_calls=totals['calls'], total_tokens=totals['total_tokens'])
