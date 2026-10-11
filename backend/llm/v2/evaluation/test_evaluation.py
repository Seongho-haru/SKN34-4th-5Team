"""Focused offline checks; no provider credentials or DB tables required."""
import importlib.util
from pathlib import Path
import unittest
import json
import os
import subprocess
import sys
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('jev_evaluation', Path(__file__).with_name('run.py'))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTest(unittest.TestCase):
    def test_reviewed_seed_and_deterministic_contracts(self):
        data = evaluation.dataset()
        self.assertEqual(len(data.goldens), 28)
        for golden in data.goldens:
            row = golden.additional_metadata
            self.assertEqual(row['review_status'], 'implementation-reviewed')
            if row['part'] == 'deterministic':
                with self.subTest(id=row['id']):
                    self.assertEqual(evaluation.deterministic(row)['status'], 'passed')

    def test_expanded_dataset_and_frozen_split(self):
        rows = [g.additional_metadata for g in evaluation.dataset(Path(__file__).with_name('reviewed_boundary.json')).goldens]
        manifest = json.loads(Path(__file__).with_name('reviewed_boundary_manifest.json').read_text())
        semantic = [r for r in rows if r['part'] == 'semantic']
        contracts = [r for r in rows if r['part'] == 'deterministic']
        self.assertEqual(len(semantic), 44)
        self.assertEqual(len(contracts), 26)
        self.assertEqual(sum(r['guard'] == 'PASS' for r in semantic), 24)
        self.assertEqual(len({r['input'] for r in semantic}), 44)
        self.assertFalse(set(manifest['dev']) & set(manifest['heldout']))
        self.assertEqual(set(manifest['dev']) | set(manifest['heldout']), {r['id'] for r in rows})
        self.assertTrue(all('selected_tools' not in r for r in semantic))

    def test_dimension_report_does_not_credit_fixed_tools(self):
        row = {'id': 'boundary', 'status': 'failed',
               'expected': {'guard': 'PASS', 'capabilities': ['schedule'], 'course_request': 'NONE'},
               'actual': {'guard': 'PASS', 'capabilities': ['weather'], 'course_request': 'NONE'}}
        report = evaluation.dimension_report([row])
        self.assertEqual(report['guard']['exact'], 1)
        self.assertEqual(report['capability']['exact'], 0)
        self.assertEqual(report['capability']['missing'], 1)
        self.assertEqual(report['capability']['overextra'], 1)
        self.assertEqual(report['capability']['precision'], 0)
        self.assertFalse(report['fixed_tools_semantic_credit'])

    def test_preserved_seed_requires_selection_oracle(self):
        row = deepcopy(json.loads(evaluation.SEED.read_text())[0])
        del row['selected_tools']
        with self.assertRaises(ValueError): evaluation.validate_row(row)
        evaluation.validate_row(row, semantic_only=True)

    def test_actual_production_fixed_six_exposed_and_execution_gated(self):
        from llm.v2.agent import chain, sub_agents
        fixed = (*sub_agents.SPECIALISTS, 'present_planning_questions')
        self.assertEqual(len(fixed), 6)
        registry = [*chain.TOOLS, *fixed, 'jev_read_body']
        middleware = evaluation.DynamicToolMiddleware(registry, evaluation.CAPABILITY_TOOLS, fixed)
        state = {'decision': {'allowed': True, 'capabilities': ['unknown']}, 'messages': []}
        self.assertEqual(middleware.allowed(state), frozenset(fixed))
        for allowed in (False, 1, 'PASS', None):
            state['decision']['allowed'] = allowed
            self.assertEqual(middleware.allowed(state), frozenset())
            seen = []
            for name in fixed:
                request = evaluation.NS(state=state, tool_call={'name': name, 'id': 'x', 'args': {}})
                result = middleware.wrap_tool_call(request, lambda r: seen.append(r.tool_call['name']))
                self.assertEqual(result.status, 'error')
            self.assertFalse(seen)

    def test_exact_metric_rejects_guard_and_capability_counterexamples(self):
        expected = '{"guard":"PASS","capabilities":["standings"],"course_request":"NONE"}'
        for actual, score in [(expected, 1),
                ('{"guard":"PASS","capabilities":[],"course_request":"NONE"}', 0),
                ('{"guard":"NON_PASS","capabilities":[],"course_request":"NONE"}', 0)]:
            case = evaluation.LLMTestCase(input='현재 KBO 순위', actual_output=actual, expected_output=expected)
            self.assertEqual(evaluation.ExactLabels().measure(case), score)

    def test_malformed_labels(self):
        expected = {'guard': 'PASS', 'capabilities': ['standings'], 'course_request': 'NONE'}
        bad = [{}, [], {**expected, 'capabilities': {'standings': False}},
               {k: v for k, v in expected.items() if k != 'capabilities'},
               {**expected, 'guard': 'MAYBE'}, {**expected, 'course_request': 'BROKEN'},
               {**expected, 'capabilities': ['unknown']}, {**expected, 'capabilities': ['standings', 'standings']}]
        for value in bad:
            with self.subTest(value=value):
                case = evaluation.LLMTestCase(input='q', actual_output=json.dumps(value), expected_output=json.dumps(expected))
                self.assertEqual(evaluation.ExactLabels().measure(case), 0)
                case.expected_output = json.dumps(value)
                with self.assertRaises(ValueError): evaluation.ExactLabels().measure(case)
        self.assertEqual(evaluation.ExactLabels().measure(evaluation.LLMTestCase(input='q', actual_output='{', expected_output=json.dumps(expected))), 0)

    def test_dataset_boundary(self):
        rows = json.loads(evaluation.SEED.read_text())
        for change in [{'review_status': 'candidate/unreviewed'}, {'provenance': 'generated'},
                       {'part': 'mystery'}, {'guard': 'MAYBE'}, {'course_request': 'BROKEN'},
                       {'capabilities': ['unknown']}, {'capabilities': 'standings'},
                       {'selected_tools': ['missing']}, {'unexpected': True}, {'history': [{'role': 'system', 'content': 'q'}]}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluation.validate_row({**rows[0], **change})
        row = deepcopy(rows[0]); del row['guard']
        with self.assertRaises(ValueError): evaluation.validate_row(row)
        with patch.object(evaluation.Path, 'read_text', return_value=json.dumps([rows[0]] * 28)):
            with self.assertRaises(ValueError): evaluation.dataset()

    def test_invalid_tool_calls_fail(self):
        row = {'selected_tools': []}
        response = evaluation.AIMessage('', invalid_tool_calls=[{'name': 'get_games', 'args': '{', 'id': 'bad', 'error': 'parse'}])
        self.assertFalse(evaluation.selection_ok(row, response))
        self.assertFalse(evaluation.selection_ok({'selected_tools': ['get_games'], 'selection_mode': 'subset'}, evaluation.AIMessage('')))

    def test_supervisor_rejects_unsupported_model_call_budget_before_model_creation(self):
        from llm.v2.agent import chain, common
        for budget in (1, 0, -1):
            with self.subTest(budget=budget), patch.object(chain, 'MAIN_MODEL_CALL_BUDGET', budget), \
                    patch.object(common, 'llm') as model:
                with self.assertRaisesRegex(ValueError, 'MAIN_MODEL_CALL_BUDGET > 1'):
                    evaluation.supervisor([])
                model.assert_not_called()

    def test_supervisor_matches_production_first_input(self):
        from llm.v2.agent import chain, common
        from llm.v2.tests.test_chain import ScriptedModel, fake_tools
        from langchain.agents.middleware import ModelRequest
        for mode, question, selected in [('NEW', '잠실 새 코스 짜줘', False), ('EDIT', '카페만 바꿔줘', False),
                                         ('NONE', '카페 바꿔줘', False), ('NEW', '이 장소 다음에 카페 추가해줘', True)]:
            course = {'places': [{'name': 'old', 'category': 'CE7', 'visitId': 'v1', 'label': 'A', 'phase': 'before'}]}
            if selected: course['selectedPlace'] = course['places'][0]
            row = {'input': question, 'guard': 'PASS', 'capabilities': [], 'course_request': mode,
                   'history': [{'role': 'human', 'content': '이전 부산 코스'}, {'role': 'ai', 'content': '과거'}],
                   'context': {'stadium': 'JAMSIL', 'currentCourse': course}, 'course_memory': {'conditions': ['old']}}
            original = deepcopy(row)
            state = evaluation.supervisor_state(row)
            adapted = []
            request = ModelRequest(model=ScriptedModel(script=[], calls=[]), messages=state['messages'], state=state, tools=[], runtime=None)
            evaluation.jev.JevGuidelineMiddleware(chain.MAIN_RULES, True).wrap_model_call(request, lambda r: adapted.extend([r.system_message, *r.messages]))
            calls = []
            model = ScriptedModel(script=[evaluation.AIMessage('done')], calls=calls)
            decision = {'allowed': True, 'capabilities': [], 'course_request': mode}
            with patch.object(evaluation.jev, 'classify', return_value=decision):
                chain.build_graph(model, fake_tools([])).invoke({'messages': [*evaluation.messages(row), evaluation.HumanMessage(question)], 'context': deepcopy(row['context']), 'course_memory': deepcopy(row['course_memory'])})
            # Clock text can cross a second; all other bytes and message content must match.
            import re
            normalize = lambda text: re.sub(r'현재 한국 시각은 .*? 이다', '현재 한국 시각은 CLOCK 이다', text)
            self.assertEqual([normalize(m.content) for m in adapted], [normalize(m.content) for m in calls[0]['messages']])
            registry = [*chain.TOOLS, *chain.sub_agents.SPECIALISTS, 'present_planning_questions', 'jev_read_body']
            dynamic = evaluation.DynamicToolMiddleware(registry, evaluation.CAPABILITY_TOOLS,
                                                       (*chain.sub_agents.SPECIALISTS, 'present_planning_questions'))
            self.assertEqual(dynamic.allowed(state), frozenset(calls[0]['tools']))
            self.assertEqual(row, original)
            if state['decision'].get('course_request') == 'EDIT':
                self.assertNotIn('<current_course_destination>', adapted[-1].content)
            else:
                self.assertIn('<current_course_destination>', adapted[-1].content)

    def test_empty_selection_cli(self):
        with patch.object(sys, 'argv', ['run.py', 'offline', '--ids', 'greeting', '--output', '/Users/yunseongho/.hermes/cache/scratch/never-written-empty.json']):
            with self.assertRaises(SystemExit) as raised: evaluation.main()
            self.assertEqual(raised.exception.code, 2)

    def test_generation_cardinality(self):
        from deepeval.synthesizer import Synthesizer
        from openai import OpenAI
        for data in [[], [{'input': '하나'}, {'input': '둘'}]]:
            response = evaluation.NS(output_text=json.dumps({'data': data}), usage=evaluation.NS(model_dump=lambda: {}))
            with patch.object(OpenAI, '__init__', return_value=None), patch.object(OpenAI, 'responses', create=True) as responses:
                responses.create.return_value = response
                result = evaluation.generate(1)
            self.assertEqual(result['status'], 'generation_error')
        with patch.object(OpenAI, '__init__', return_value=None), patch.object(Synthesizer, 'generate_goldens_from_contexts', return_value=[]):
            self.assertEqual(evaluation.generate(1)['status'], 'generation_error')

    def test_import_isolation_and_explicit_credentials(self):
        code = '''import importlib.util, json, os, sys
from pathlib import Path
blocked=[]
authorized=False
fixture=Path(sys.argv[2]).resolve()
def deny(event, args):
    if event.startswith('socket.') and event != 'socket.__new__':
        blocked.append(event)
        raise AssertionError('Network attempted: '+event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        source=Path(os.fsdecode(args[0]))
        if source.name == '.env' or source.name.startswith('.env.') or source.name in ('.deepeval', '.deepeval_confident', 'credentials', 'credentials.json'):
            if authorized and source.resolve() == fixture: return
            blocked.append('credential-file-open')
            raise AssertionError('Implicit credential source attempted')
sys.addaudithook(deny)
# Prove the guard rejects both safe fixture sources before importing anything.
for source in (fixture, fixture.parent / '.deepeval' / '.deepeval'):
    try: source.open()
    except AssertionError: pass
    else: raise AssertionError('Audit guard did not reject source')
blocked.clear()
spec=importlib.util.spec_from_file_location('evaluation',sys.argv[1])
e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
assert os.environ['DEEPEVAL_DISABLE_DOTENV']=='1'
assert os.environ['DEEPEVAL_DISABLE_LEGACY_KEYFILE']=='1'
assert not any(k in os.environ for k in ('OPENAI_API_KEY','OPENROUTER_API_KEY','UNRELATED_FIXTURE'))
# Exercise the real CLI selector, permitting only this nonsecret scratch file.
authorized=True
sys.argv=[sys.argv[1],'offline','--credentials',str(fixture),'--output',str(fixture.parent / 'offline.json')]
assert e.main()==0
assert os.environ['OPENAI_API_KEY']=='scratch-openai'
assert os.environ['OPENAI_BASE_URL']=='https://example.invalid/v1'
assert os.environ['OPENROUTER_API_KEY']=='scratch-router'
assert os.environ['LLM_MODEL']=='scratch-model'
assert 'UNRELATED_FIXTURE' not in os.environ
# A dependency may probe a local bind; every socket operation is denied.
assert not [event for event in blocked if event != 'socket.bind'], blocked
print(json.dumps({'blocked_operations':blocked,'explicit_selection':'four keys only'}))
'''
        with TemporaryDirectory(prefix='jev-import-audit-') as scratch:
            root = Path(scratch)
            (root / '.env').write_text('OPENAI_API_KEY=scratch-openai\nOPENAI_BASE_URL=https://example.invalid/v1\nOPENROUTER_API_KEY=scratch-router\nLLM_MODEL=scratch-model\nUNRELATED_FIXTURE=must-not-load\n')
            (root / '.deepeval').mkdir()
            (root / '.deepeval' / '.deepeval').write_text('{"OPENAI_API_KEY":"scratch-legacy"}')
            env = {'PATH': os.environ.get('PATH', ''), 'HOME': scratch, 'TMPDIR': scratch,
                   'DEEPEVAL_DISABLE_DOTENV': '0', 'DEEPEVAL_DISABLE_LEGACY_KEYFILE': '0',
                   'DEEPEVAL_FILE_SYSTEM': 'READ_ONLY', 'DEEPEVAL_CACHE_FOLDER': str(root / 'cache'),
                   'PYTHONDONTWRITEBYTECODE': '1'}
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code,
                                     str(Path(__file__).with_name('run.py').resolve()), str(root / '.env')],
                                    cwd=scratch, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_inherited_and_contextual_tracing_disabled(self):
        code = '''import os, sys, importlib.util
os.environ.update(LANGSMITH_TRACING='true', LANGCHAIN_TRACING_V2='true', LANGSMITH_API_KEY='fake', LANGSMITH_ENDPOINT='https://example.invalid')
from langsmith import Client, tracing_context
from unittest.mock import patch
calls=[]
def deny(event, args):
    if event.startswith('socket.') and event not in ('socket.__new__',):
        raise AssertionError('Network attempted: '+event)
sys.addaudithook(deny)
with patch.object(Client,'create_run',side_effect=lambda *a,**k:calls.append('create')), patch.object(Client,'update_run',side_effect=lambda *a,**k:calls.append('update')), patch.object(Client,'batch_ingest_runs',side_effect=lambda *a,**k:calls.append('batch')):
    spec=importlib.util.spec_from_file_location('evaluation',sys.argv[1]); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    assert os.environ['LANGSMITH_TRACING']=='false' and os.environ['LANGCHAIN_TRACING_V2']=='false'
    with tracing_context(enabled=True): e.deterministic({'id':'concurrency','check':'concurrency'})
assert not calls, calls
print('zero upload callbacks; zero permitted network')
'''
        result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).with_name('run.py'))], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__': unittest.main()
