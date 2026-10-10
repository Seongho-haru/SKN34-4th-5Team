"""Five frozen datasets: strict ingestion, DeepEval contract metric, local output scoring.
No model/tool calls in validate/score. Semantic prose quality remains human-reviewed.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import run as base

HERE = Path(__file__).parent
GROUPS = ('first_action', 'clarification', 'delegation', 'schedule', 'final_answer')
TOOLS = base.KNOWN_TOOLS


def obj(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError('Missing or unknown object keys')


def text(value, limit=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit: raise ValueError('Invalid string')


def boolean(value):
    if type(value) is not bool: raise ValueError('Expected boolean')


def strings(value, domain=None, maximum=100):
    if not isinstance(value, list) or len(value) > maximum: raise ValueError('Invalid list length')
    for item in value:
        text(item)
        if domain is not None and item not in domain: raise ValueError('Unknown enum/tool')
    if len(value) != len(set(value)): raise ValueError('Duplicate strings')


NOTICES = {'demo', 'partial_candidates', 'lookup_failed', 'partial_results', 'stale_saved', 'unconfirmed', 'conflicting_sources'}
SCHEDULE_KEYS = {'start_date', 'end_date', 'upcoming_only', 'status', 'availability', 'stale', 'total_count', 'has_more', 'offset'}
CLAIM_KEYS = {'evidence_id', 'date', 'home', 'away', 'status', 'figure', 'source_url', 'demo'}


def schedule(value):
    obj(value, SCHEDULE_KEYS)
    for key in ('start_date', 'end_date'): datetime.strptime(value[key], '%Y-%m-%d')
    if value['start_date'] > value['end_date']: raise ValueError('Reversed dates')
    for key in ('upcoming_only', 'stale', 'has_more'): boolean(value[key])
    for key in ('total_count', 'offset'):
        if type(value[key]) is not int or not 0 <= value[key] <= 1000000: raise ValueError('Invalid count')
    if value['status'] not in ('all', 'upcoming', 'finished', 'canceled'): raise ValueError('Unknown status')
    if value['availability'] not in ('available', 'empty', 'unknown', 'error'): raise ValueError('Unknown availability')


def claims(value, evidence_ids):
    if not isinstance(value, list) or len(value) > 100: raise ValueError('Invalid claims')
    for claim in value:
        obj(claim, CLAIM_KEYS)
        for key in CLAIM_KEYS - {'figure', 'demo'}: text(claim[key], 200)
        if claim['evidence_id'] not in evidence_ids: raise ValueError('Unknown evidence ID')
        datetime.strptime(claim['date'], '%Y-%m-%d')
        if claim['status'] not in ('scheduled', 'canceled', 'finished', 'unknown'): raise ValueError('Unknown game status')
        if claim['figure'] is not None: text(claim['figure'], 80)
        if not claim['source_url'].startswith('https://example.invalid/'): raise ValueError('Synthetic source must be labeled')
        boolean(claim['demo'])
    if len({json.dumps(c, sort_keys=True) for c in value}) != len(value): raise ValueError('Duplicate claims')


def expected(group, value, evidence_ids):
    if group == 'first_action':
        obj(value, {'allowed_first', 'no_call'})
        strings(value['allowed_first'], TOOLS)
        boolean(value['no_call'])
        if value['no_call'] != (not value['allowed_first']): raise ValueError('Inconsistent no-call')
    elif group in ('clarification', 'delegation'):
        keys = {'required_events', 'order', 'forbidden_tools', 'ui', 'notices'} if group == 'clarification' else {'required_events', 'order', 'parallel', 'exact_tools', 'one_goal', 'main_questions', 'no_nested'}
        obj(value, keys)
        events = value['required_events']
        strings(events, TOOLS | {'lookup', 'ui', 'user_answer', 'followup_lookup', 'final'})
        if not isinstance(value['order'], list): raise ValueError('Expected order pairs')
        for pair in value['order']:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(x not in events for x in pair) or pair[0] == pair[1]: raise ValueError('Invalid partial order')
        if group == 'clarification':
            strings(value['forbidden_tools'], TOOLS)
            strings(value['notices'], NOTICES)
            ui = value['ui']
            obj(ui, {'required', 'choices', 'offer_writer', 'free_input'})
            for key in ('required', 'offer_writer', 'free_input'): boolean(ui[key])
            strings(ui['choices'], maximum=10)
            if len(ui['choices']) == 1 or any(len(c) > 80 for c in ui['choices']): raise ValueError('Invalid UI choices')
        else:
            strings(value['exact_tools'], TOOLS)
            for key in ('one_goal', 'main_questions', 'no_nested'): boolean(value[key])
            if not isinstance(value['parallel'], list): raise ValueError('Expected parallel batches')
            for batch in value['parallel']:
                strings(batch, TOOLS)
                if len(batch) < 2 or not set(batch) <= set(events): raise ValueError('Invalid parallel tools')
    elif group == 'schedule':
        obj(value, {'schedule', 'notices', 'strict_future'})
        schedule(value['schedule'])
        strings(value['notices'], NOTICES)
        boolean(value['strict_future'])
    else:
        obj(value, {'claims', 'notices'})
        claims(value['claims'], evidence_ids)
        strings(value['notices'], NOTICES)


def load():
    rows = []
    evidence_ids = set()
    for group in GROUPS:
        data = json.loads((HERE / f'{group}.json').read_text())
        obj(data, {'schema_version', 'group', 'cases'})
        if type(data['schema_version']) is not int or data['schema_version'] != 1 or data['group'] != group: raise ValueError('Unknown schema/group')
        if not isinstance(data['cases'], list) or len(data['cases']) < 12: raise ValueError('Expected at least twelve cases')
        for row in data['cases']:
            obj(row, {'id', 'family', 'input', 'provenance', 'review_status', 'synthetic', 'as_of', 'timezone', 'context', 'observations', 'rubric', 'expected'}, {'decision'})
            for key in ('id', 'family', 'input', 'rubric'): text(row[key])
            if row['provenance'] != 'hand-authored' or row['review_status'] != 'author-reviewed' or row['synthetic'] is not True: raise ValueError('Unreviewed/unlabeled fixture')
            if row['as_of'] != '2026-10-10T18:00:00+09:00' or row['timezone'] != 'Asia/Seoul': raise ValueError('Clock must be frozen')
            if not isinstance(row['context'], dict) or not isinstance(row['observations'], list): raise ValueError('Invalid context/observations')
            ids = set()
            for observation in row['observations']:
                obj(observation, {'evidence_id', 'tool', 'data'})
                text(observation['evidence_id'])
                if observation['evidence_id'] in evidence_ids: raise ValueError('Duplicate evidence ID')
                evidence_ids.add(observation['evidence_id']); ids.add(observation['evidence_id'])
                if observation['tool'] not in TOOLS or not isinstance(observation['data'], dict): raise ValueError('Invalid frozen tool observation')
            expected(group, row['expected'], ids)
            if group == 'schedule':
                frozen = row['observations'][0]['data']
                if {k: frozen[k] for k in SCHEDULE_KEYS} != row['expected']['schedule']: raise ValueError('Observation/oracle conflict')
                projected = [{k: g[k] for k in ('game_date', 'game_time', 'status_code')} for g in frozen['games']]
                if not contract({**row, 'group': group}, {'schedule': row['expected']['schedule'], 'notices': row['expected']['notices'], 'games': projected, 'claims_no_games': False}): raise ValueError('Invalid schedule observation')
                s = row['expected']['schedule']
                if s['has_more'] != (s['offset'] + len(projected) < s['total_count']): raise ValueError('Inconsistent pagination')
            if group == 'first_action': base.labels(row.get('decision'))
            elif 'decision' in row: raise ValueError('Unexpected decision')
            rows.append({**row, 'group': group})
    if len({r['id'] for r in rows}) != len(rows) or len({r['input'] for r in rows}) != len(rows): raise ValueError('Duplicate IDs/inputs')
    manifest = json.loads((HERE / 'remaining_manifest.json').read_text())
    obj(manifest, {'schema_version', 'split_basis', 'dev', 'heldout'})
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1: raise ValueError('Unknown manifest schema')
    text(manifest['split_basis'])
    for key in ('dev', 'heldout'): strings(manifest[key])
    dev, held = set(manifest['dev']), set(manifest['heldout'])
    if not dev or not held or dev & held or dev | held != {r['id'] for r in rows}: raise ValueError('Invalid split')
    families = {}
    for row in rows: families.setdefault(row['family'], set()).add('dev' if row['id'] in dev else 'heldout')
    if any(len(parts) != 1 for parts in families.values()): raise ValueError('Sibling paraphrases leak across splits')
    return rows, manifest


@base.local_only
def dataset():
    rows, _ = load()
    return base.EvaluationDataset(goldens=[base.Golden(input=r['input'], expected_output=json.dumps(r['expected'], ensure_ascii=False), additional_metadata=r) for r in rows])


def contract(row, actual):
    """Only supplied output contracts; never generates or replays a model answer."""
    e, group = row['expected'], row['group']
    if group == 'first_action':
        obj(actual, {'selected_tools', 'invalid_tool_calls'})
        strings(actual['selected_tools'], TOOLS)
        boolean(actual['invalid_tool_calls'])
        return not actual['invalid_tool_calls'] and (not actual['selected_tools'] if e['no_call'] else bool(actual['selected_tools']) and set(actual['selected_tools']) <= set(e['allowed_first']))
    if group in ('clarification', 'delegation'):
        keys = {'events', 'tools', 'notices', 'ui', 'main_questions'} if group == 'clarification' else {'events', 'tools', 'parallel', 'goals', 'main_questions', 'nested_delegation'}
        obj(actual, keys)
        events = actual['events']
        strings(events, TOOLS | {'lookup', 'ui', 'user_answer', 'followup_lookup', 'final'})
        strings(actual['tools'], TOOLS)
        boolean(actual['main_questions'])
        if not actual['main_questions'] or not set(e['required_events']) <= set(events): return False
        if any(events.index(a) >= events.index(b) for a, b in e['order']): return False
        if group == 'delegation':
            boolean(actual['nested_delegation'])
            if not isinstance(actual['goals'], dict) or set(actual['goals']) != {t for t in e['exact_tools'] if t.startswith('ask_')}: return False
            for goals in actual['goals'].values(): strings(goals)
            if actual['nested_delegation'] or set(actual['tools']) != set(e['exact_tools']) or any(len(v) != 1 for v in actual['goals'].values()): return False
            if not isinstance(actual['parallel'], list): return False
            for batch in actual['parallel']: strings(batch, TOOLS)
            return sorted(map(sorted, actual['parallel'])) == sorted(map(sorted, e['parallel']))
        strings(actual['notices'], NOTICES)
        if set(actual['tools']) & set(e['forbidden_tools']) or not set(e['notices']) <= set(actual['notices']): return False
        ui = actual['ui']
        if not e['ui']['required']: return ui is None and 'ui' not in events
        obj(ui, {'question', 'choices', 'offer_writer', 'free_input'})
        text(ui['question'], 160)
        strings(ui['choices'], maximum=10)
        if any(len(c) > 80 for c in ui['choices']) or len(ui['choices']) == 1: return False
        boolean(ui['offer_writer']); boolean(ui['free_input'])
        if any(ui[k] != e['ui'][k] for k in ('choices', 'offer_writer', 'free_input')): return False
        candidates = [c for o in row['observations'] for c in o['data'].get('candidates', [])]
        return set(ui['choices']) <= set(candidates) and (not ui['choices'] or events.index('lookup') < events.index('ui'))
    if group == 'schedule':
        obj(actual, {'schedule', 'notices', 'games', 'claims_no_games'})
        schedule(actual['schedule']); strings(actual['notices'], NOTICES); boolean(actual['claims_no_games'])
        if actual['schedule'] != e['schedule'] or not set(e['notices']) <= set(actual['notices']): return False
        if actual['claims_no_games'] and e['schedule']['availability'] != 'empty': return False
        if not isinstance(actual['games'], list): return False
        frozen_games = row['observations'][0]['data']['games']
        projected = [{k: game[k] for k in ('game_date', 'game_time', 'status_code')} for game in frozen_games]
        if actual['games'] != projected: return False
        for game in actual['games']:
            obj(game, {'game_date', 'game_time', 'status_code'})
            text(game['status_code'])
            start = datetime.fromisoformat(f"{game['game_date']}T{game['game_time']}+09:00")
            if not e['schedule']['start_date'] <= game['game_date'] <= e['schedule']['end_date']: return False
            if e['strict_future'] and (start <= datetime.fromisoformat(row['as_of']) or game['status_code'] not in ('PREV', 'READY', 'scheduled')): return False
        return True
    obj(actual, {'claims', 'notices'})
    claims(actual['claims'], {o['evidence_id'] for o in row['observations']})
    strings(actual['notices'], NOTICES)
    # Full tuples, not a bag of dates/team names/numbers: reassociation must fail.
    return actual['claims'] == e['claims'] and set(e['notices']) <= set(actual['notices'])


class StructuredContract(base.BaseMetric):
    threshold = 1
    async_mode = False
    def __init__(self, row): self.row = row
    @property
    def __name__(self): return 'Structured output contract (not semantic prose truth)'
    def measure(self, case, **kwargs):
        expected(self.row['group'], self.row['expected'], {o['evidence_id'] for o in self.row['observations']})
        try: ok = contract(self.row, json.loads(case.actual_output))
        except (ValueError, TypeError, KeyError, IndexError): ok = False
        self.score, self.success = float(ok), bool(ok)
        self.reason = 'Structured contract satisfied; semantic review pending' if ok else 'Structured contract violation'
        return self.score
    async def a_measure(self, case, **kwargs): return self.measure(case)
    def is_successful(self): return self.success


def supervisor_rows(rows):
    """Compatible with existing real supervisor adapter; classifier oracle independent.
    No invocation here. Invoke base.supervisor on an explicitly authorized <=5 selection.
    """
    return [{'id': r['id'], 'input': r['input'], **r['decision'], 'context': deepcopy(r['context']),
             'selected_tools': r['expected']['allowed_first'], 'selection_mode': 'exact' if r['expected']['no_call'] else 'subset'}
            for r in rows if r['group'] == 'first_action']


@base.local_only
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('validate', 'score', 'supervisor-input'))
    parser.add_argument('--actual', type=Path, help='JSON object mapping case IDs to supplied structured outputs; complete suite required')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = dataset()
    rows, manifest = load()
    if args.command == 'validate':
        result = {'counts': {g: sum(r['group'] == g for r in rows) for g in GROUPS}, 'total': len(rows), 'dev': len(manifest['dev']), 'heldout': len(manifest['heldout']), 'goldens': len(data.goldens)}
    elif args.command == 'supervisor-input': result = supervisor_rows(rows)
    else:
        if args.actual is None: parser.error('score requires --actual; no scripted quality replay')
        outputs = json.loads(args.actual.read_text())
        obj(outputs, {r['id'] for r in rows})
        results = []
        for row in rows:
            case = base.LLMTestCase(input=row['input'], actual_output=json.dumps(outputs[row['id']]), expected_output=json.dumps(row['expected']))
            metric = StructuredContract(row)
            score = metric.measure(case)
            results.append({'id': row['id'], 'split': 'dev' if row['id'] in manifest['dev'] else 'heldout', 'contract_score': score, 'reason': metric.reason, 'semantic_correctness': 'not_measured'})
        result = {'scope': 'supplied structured outputs only, not production/live quality', 'results': results}
    if args.output:
        with args.output.open('x') as target: target.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    else: print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(args.command == 'score' and any(r['contract_score'] != 1 for r in result['results']))


if __name__ == '__main__': raise SystemExit(main())
