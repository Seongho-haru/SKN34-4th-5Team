"""Offline scorer self-tests, not replayed production/model quality scores."""
from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import remaining as suite


def positive(row):
    """Scorer unit-test input deliberately derived from oracle, never model evidence."""
    e, group = row['expected'], row['group']
    if group == 'first_action': return {'selected_tools': e['allowed_first'][:1], 'invalid_tool_calls': False}
    if group == 'clarification':
        return {'events': e['required_events'], 'tools': [] if not e['ui']['required'] else ['present_planning_questions'],
                'notices': e['notices'], 'main_questions': True,
                'ui': None if not e['ui']['required'] else {'question': '합성 선택 질문', **{k: e['ui'][k] for k in ('choices', 'offer_writer', 'free_input')}}}
    if group == 'delegation':
        return {'events': e['required_events'], 'tools': e['exact_tools'], 'parallel': e['parallel'],
                'goals': {t: ['합성 단일 조회 목표'] for t in e['exact_tools'] if t.startswith('ask_')},
                'main_questions': True, 'nested_delegation': False}
    if group == 'schedule':
        return {'schedule': e['schedule'], 'notices': e['notices'], 'games': [{k: g[k] for k in ('game_date', 'game_time', 'status_code')} for g in row['observations'][0]['data']['games']], 'claims_no_games': False}
    return deepcopy(e)


class RemainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rows, cls.manifest = suite.load()
    def score(self, row, actual):
        case = suite.base.LLMTestCase(input=row['input'], actual_output=json.dumps(actual), expected_output=json.dumps(row['expected']))
        return suite.StructuredContract(row).measure(case)
    def row(self, id): return next(r for r in self.rows if r['id'] == id)
    def test_sixty_goldens_and_frozen_family_split(self):
        self.assertEqual(len(suite.dataset().goldens), 60)
        self.assertEqual({g: sum(r['group'] == g for r in self.rows) for g in suite.GROUPS}, {g: 12 for g in suite.GROUPS})
        self.assertEqual(len({r['id'] for r in self.rows}), 60)
        self.assertEqual(len({r['input'] for r in self.rows}), 60)
        self.assertEqual((len(self.manifest['dev']), len(self.manifest['heldout'])), (46, 14))
    def test_every_scorer_positive_and_bad_counterexample(self):
        for row in self.rows:
            with self.subTest(id=row['id']):
                actual = positive(row)
                self.assertEqual(self.score(row, actual), 1)
                bad = deepcopy(actual)
                if row['group'] == 'first_action': bad['invalid_tool_calls'] = True
                elif row['group'] in ('clarification', 'delegation'): bad['events'] = []
                elif row['group'] == 'schedule': bad['schedule']['total_count'] += 1
                else: bad['notices'] = []
                self.assertEqual(self.score(row, bad), 0)
                self.assertEqual(self.score(row, {**actual, 'unknown': True}), 0)
    def test_exact_real_queries_and_adapter_selection(self):
        rows = suite.supervisor_rows(self.rows)
        self.assertEqual(rows[0]['input'], '경기일정 조회 해줘')
        self.assertEqual(rows[1]['input'], '음 그냥 가고싶은데 주변에서 뭘해야해?')
        self.assertEqual(len(rows), 12)
        for row in rows:
            suite.base.labels({k: row[k] for k in suite.base.LABEL_KEYS})
            good = suite.base.AIMessage('', tool_calls=[{'name': t, 'args': {}, 'id': str(i)} for i, t in enumerate(row['selected_tools'][:1])])
            self.assertTrue(suite.base.selection_ok(row, good))
            bad = suite.base.AIMessage('', tool_calls=[{'name': 'present_planning_questions', 'args': {}, 'id': 'bad'}])
            self.assertFalse(suite.base.selection_ok(row, bad))
    def test_grounded_ui_order_and_choices(self):
        row = self.row('clarification-stadium')
        for mutation in ('order', 'invent', 'eleven', 'free', 'writer', 'owner'):
            actual = deepcopy(positive(row))
            if mutation == 'order': actual['events'][0:2] = ['ui', 'lookup']
            if mutation == 'invent': actual['ui']['choices'] = ['가짜 구장', '사직']
            if mutation == 'eleven': actual['ui']['choices'] = [str(i) for i in range(11)]
            if mutation == 'free': actual['ui']['free_input'] = False
            if mutation == 'writer': actual['ui']['offer_writer'] = True
            if mutation == 'owner': actual['main_questions'] = False
            self.assertEqual(self.score(row, actual), 0)
        known = self.row('clarification-known-screen')
        actual = positive(known); actual['tools'] = ['present_planning_questions']
        self.assertEqual(self.score(known, actual), 0)
    def test_independent_parallel_and_narrow_goal(self):
        row = self.row('delegation-independent')
        for mutation in ('sequential', 'goals', 'nested', 'owner'):
            actual = deepcopy(positive(row))
            if mutation == 'sequential': actual['parallel'] = []
            if mutation == 'goals': actual['goals']['ask_baseball'] = ['교통', '음식']
            if mutation == 'nested': actual['nested_delegation'] = True
            if mutation == 'owner': actual['main_questions'] = False
            self.assertEqual(self.score(row, actual), 0)
    def test_schedule_cutoff_status_stale_and_pagination(self):
        row = self.row('schedule-at-cutoff')
        for time, status, wanted in [('18:00:00', 'READY', 0), ('18:01:00', 'END', 0), ('18:01:00', 'READY', 1)]:
            actual = deepcopy(positive(row)); actual['games'] = [{'game_date': '2026-10-10', 'game_time': time, 'status_code': status}]
            self.assertEqual(self.score(row, actual), wanted)
        for id in ('schedule-stale-empty', 'schedule-error', 'schedule-stale-saved'):
            row = self.row(id); actual = deepcopy(positive(row)); actual['claims_no_games'] = True
            self.assertEqual(self.score(row, actual), 0)
        row = self.row('schedule-page-one'); actual = deepcopy(positive(row)); actual['schedule']['has_more'] = False
        self.assertEqual(self.score(row, actual), 0)
    def test_final_answer_association_not_numberbag(self):
        row = self.row('final_answer-two-games')
        actual = positive(row)
        original_dates = [c['date'] for c in actual['claims']]
        actual['claims'][0]['date'], actual['claims'][1]['date'] = actual['claims'][1]['date'], actual['claims'][0]['date']
        self.assertCountEqual(original_dates, [c['date'] for c in actual['claims']])
        self.assertEqual(self.score(row, actual), 0, 'same date/team bags, unsupported association')
        for key, value in [('home', '합성 원정 B'), ('figure', '999명'), ('source_url', 'https://example.invalid/invented'), ('demo', False)]:
            actual = positive(row); actual['claims'][0][key] = value
            self.assertEqual(self.score(row, actual), 0)
    def test_schema_type_enum_length_unknown_key_rejection(self):
        for row in self.rows:
            e = deepcopy(row['expected']); e['unknown'] = True
            with self.assertRaises(ValueError): suite.expected(row['group'], e, {o['evidence_id'] for o in row['observations']})
        for value in (['imaginary_tool'], ['get_games', 'get_games'], 'get_games', [1]):
            with self.assertRaises(ValueError): suite.strings(value, suite.TOOLS)
        for value in (1, 'false', None):
            with self.assertRaises(ValueError): suite.boolean(value)
        row = self.row('clarification-stadium'); e = deepcopy(row['expected']); e['ui']['choices'] = ['a'] * 11
        with self.assertRaises(ValueError): suite.expected(row['group'], e, set())
        row = self.row('schedule-today'); e = deepcopy(row['expected']); e['schedule']['status'] = 'FAKE'
        with self.assertRaises(ValueError): suite.expected(row['group'], e, set())
    def test_duplicate_ids_unknown_keys_and_split_leaks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group in suite.GROUPS: (root / f'{group}.json').write_bytes((suite.HERE / f'{group}.json').read_bytes())
            (root / 'remaining_manifest.json').write_bytes((suite.HERE / 'remaining_manifest.json').read_bytes())
            p = root / 'first_action.json'; original = json.loads(p.read_text())
            for mutation in ('duplicate', 'unknown', 'type'):
                data = deepcopy(original)
                if mutation == 'duplicate': data['cases'][1]['id'] = data['cases'][0]['id']
                elif mutation == 'unknown': data['cases'][0]['unknown'] = True
                else: data['cases'][0]['expected']['no_call'] = 0
                p.write_text(json.dumps(data))
                with patch.object(suite, 'HERE', root), self.assertRaises(ValueError): suite.load()
            p.write_text(json.dumps(original))
            manifest = deepcopy(self.manifest)
            id = 'first_action-schedule-paraphrase'
            manifest['dev'].remove(id); manifest['heldout'].append(id)
            (root / 'remaining_manifest.json').write_text(json.dumps(manifest))
            with patch.object(suite, 'HERE', root), self.assertRaises(ValueError): suite.load()
    def test_actual_compiled_graph_scripted_contracts_not_live_quality(self):
        methods = ['test_grounded_candidates_delegate_before_main_question',
                   'test_complex_calls_jev_once_and_specialists_run_in_parallel',
                   'test_tool_loop_ends_with_final_answer_not_recursion_error']
        for method in methods:
            result = suite.base.deterministic({'id': method, 'check': 'graph_regression', 'test_method': method})
            self.assertEqual(result['status'], 'passed')
            self.assertIn('scripted', result['evidence'])


if __name__ == '__main__': unittest.main()
