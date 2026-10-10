"""Local, no-upload JEV evaluation. Run from repo root; see README.md."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
from functools import wraps
from copy import deepcopy

os.environ.update(DEEPEVAL_TELEMETRY_OPT_OUT="1", PYTHON_DOTENV_DISABLED="1",
                  DEEPEVAL_DISABLE_DOTENV="1", DEEPEVAL_DISABLE_LEGACY_KEYFILE="1",
                  DJANGO_SETTINGS_MODULE="config.settings", WEB_RESEARCH_ENABLED="false",
                  LANGSMITH_TRACING="false", LANGCHAIN_TRACING="false", LANGCHAIN_TRACING_V2="false")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["PYTHONPATH"] = str(ROOT / "backend")
from langsmith import tracing_context
with tracing_context(enabled=False):
    from django.conf import settings
    settings.DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
    settings.DATABASE_ROUTERS = []
    import django
    django.setup()
    from deepeval.dataset import Golden, EvaluationDataset
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from llm.v2.middleware import jev_guidelines as jev
    from llm.v2.middleware.dynamic_tools import DynamicToolMiddleware, CAPABILITY_TOOLS

SEED = Path(__file__).with_name("seed.json")


def local_only(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with tracing_context(enabled=False):
            return fn(*args, **kwargs)
    return wrapped


LABEL_KEYS = {'guard', 'capabilities', 'course_request'}
CHECKS = {'graph_regression', 'allocation', 'threshold', 'invalid_guard', 'invalid_course', 'provider_error',
          'hidden_call', 'attachment', 'suppression', 'specialist', 'course_transition', 'concurrency'}
KNOWN_TOOLS = {n for names in CAPABILITY_TOOLS.values() for n in names} | {'jev_read_body', 'present_planning_questions'}


def string_list(value, domain):
    if (not isinstance(value, list) or any(not isinstance(v, str) or v not in domain for v in value)
            or len(set(value)) != len(value)):
        raise ValueError('Expected unique strings in the known domain')
    return value


def labels(value):
    if not isinstance(value, dict) or set(value) != LABEL_KEYS:
        raise ValueError('Expected exactly guard, capabilities, course_request')
    if value['guard'] not in ('PASS', 'NON_PASS') or value['course_request'] not in ('NONE', 'NEW', 'EDIT'):
        raise ValueError('Unknown guard/course label')
    return {**value, 'capabilities': sorted(string_list(value['capabilities'], jev.CAPABILITIES))}


def validate_row(row, semantic_only=False):
    if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError('Seed needs a nonempty string ID')
    if row.get('review_status') != 'implementation-reviewed' or row.get('provenance') != 'hand-authored':
        raise ValueError('Only explicitly implementation-reviewed hand-authored seeds are accepted')
    if row.get('part') not in ('semantic', 'deterministic'):
        raise ValueError('Unknown seed part')
    allowed_keys = {'id', 'review_status', 'provenance', 'part', 'history', 'context', 'course_memory'}
    allowed_keys |= ({'input', 'guard', 'capabilities', 'course_request', 'selected_tools', 'selection_mode'}
                     if row['part'] == 'semantic' else {'check', 'role_tools', 'capabilities', 'expected_tools', 'tool_group_ids', 'test_method'})
    if set(row) - allowed_keys: raise ValueError('Unknown seed fields')
    if 'course_memory' in row and not isinstance(row['course_memory'], dict): raise ValueError('Memory must be an object')
    messages(row)
    if 'context' in row and not isinstance(row['context'], dict):
        raise ValueError('Context must be an object')
    if row['part'] == 'semantic':
        if not isinstance(row.get('input'), str) or not row['input'].strip():
            raise ValueError('Semantic seed needs a question')
        labels({k: row[k] for k in LABEL_KEYS if k in row})
        if not semantic_only or 'selected_tools' in row:
            string_list(row.get('selected_tools'), KNOWN_TOOLS)
        if row.get('selection_mode', 'exact') not in ('exact', 'subset'):
            raise ValueError('Unknown selection mode')
        if row.get('selection_mode') == 'subset' and not row['selected_tools']:
            raise ValueError('Subset expectation must be nonempty')
    else:
        if row.get('check') not in CHECKS:
            raise ValueError('Unknown deterministic check')
        if row['check'] == 'graph_regression' and (not isinstance(row.get('test_method'), str) or not row['test_method'].startswith('test_')):
            raise ValueError('Graph scenario needs a test method')
        if row['check'] == 'allocation':
            string_list(row.get('role_tools'), KNOWN_TOOLS)
            string_list(row.get('expected_tools'), KNOWN_TOOLS)
            # Deliberate fail-closed counterexample, not a semantic oracle capability.
            domain = (*jev.CAPABILITIES, 'unknown') if row['id'] == 'unknown-group' else jev.CAPABILITIES
            string_list(row.get('capabilities'), domain)
            string_list(row.get('tool_group_ids', []), jev.CAPABILITIES)


def dataset(path=None):
    source = path or SEED
    rows = json.loads(source.read_text())
    if not isinstance(rows, list) or not rows or (source == SEED and not 20 <= len(rows) <= 30):
        raise ValueError('Expected 20–30 seed rows')
    for row in rows: validate_row(row, semantic_only=source != SEED)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate seed ID')
    return EvaluationDataset(goldens=[Golden(input=r.get('input', r['id']),
        expected_output=json.dumps({k: r[k] for k in LABEL_KEYS}, sort_keys=True) if r['part'] == 'semantic' else None,
        additional_metadata={**r, 'scope': 'no-rag'}) for r in rows])


class ExactLabels(BaseMetric):
    threshold = 1
    async_mode = False
    @property
    def __name__(self):
        return "Exact JEV labels (no judge)"
    def measure(self, case, **kwargs):
        expected = labels(json.loads(case.expected_output))  # Malformed oracle is a configuration error.
        try:
            actual = labels(json.loads(case.actual_output))
        except (ValueError, TypeError):
            self.score, self.success, self.reason = 0.0, False, 'Malformed actual labels'
            return self.score
        self.score = float(actual == expected)
        self.success = self.score == 1
        self.reason = 'Exact match' if self.success else 'Guard/capability/course label mismatch'
        return self.score
    async def a_measure(self, case, **kwargs):
        return self.measure(case)
    def is_successful(self):
        return self.success


def dimension_report(results):
    pairs = [r for r in results if 'actual' in r and 'expected' in r]
    tp = fp = fn = 0
    details = []
    for row in pairs:
        expected = set(row['expected']['capabilities'])
        actual = set(row['actual']['capabilities'])
        tp += len(actual & expected)
        fp += len(actual - expected)
        fn += len(expected - actual)
        details.append({'id': row['id'], 'overextra': sorted(actual - expected), 'missing': sorted(expected - actual)})
    guard_tp = sum(r['actual']['guard'] == r['expected']['guard'] == 'PASS' for r in pairs)
    guard_fp = sum(r['actual']['guard'] == 'PASS' and r['expected']['guard'] != 'PASS' for r in pairs)
    guard_fn = sum(r['actual']['guard'] != 'PASS' and r['expected']['guard'] == 'PASS' for r in pairs)
    return {'semantic_count': len(pairs), 'execution_errors': sum(r['status'] == 'execution_error' for r in results),
            'guard': {'exact': sum(r['actual']['guard'] == r['expected']['guard'] for r in pairs),
                      'false_allow': guard_fp, 'false_deny': guard_fn,
                      'precision': guard_tp / (guard_tp + guard_fp) if guard_tp + guard_fp else None,
                      'recall': guard_tp / (guard_tp + guard_fn) if guard_tp + guard_fn else None},
            'course': {'exact': sum(r['actual']['course_request'] == r['expected']['course_request'] for r in pairs)},
            'capability': {'exact': sum(set(r['actual']['capabilities']) == set(r['expected']['capabilities']) for r in pairs),
                           'true_positive': tp, 'overextra': fp, 'missing': fn,
                           'precision': tp / (tp + fp) if tp + fp else None,
                           'recall': tp / (tp + fn) if tp + fn else None, 'details': details},
            'fixed_tools_semantic_credit': False,
            'contract': {'passed': sum(r['status'] == 'passed' for r in results if 'evidence' in r),
                         'scope': 'actual production middleware/compiled graph; scripted model and safe tool handlers, not live backend execution'}}


def messages(row):
    history = row.get('history', [])
    if not isinstance(history, list) or any(not isinstance(m, dict) or set(m) != {'role', 'content'}
            or m['role'] not in ('human', 'ai') or not isinstance(m['content'], str) for m in history):
        raise ValueError('History requires human/ai roles and string content')
    return [HumanMessage(m['content']) if m['role'] == 'human' else AIMessage(m['content']) for m in history]


@local_only
def deterministic(row):
    from llm.v2.agent.sub_agents import SPECIALISTS
    check = row['check']
    middleware = DynamicToolMiddleware(['get_standings', 'get_games', 'jev_read_body', *SPECIALISTS,
                                         'search_documents_tool'], CAPABILITY_TOOLS, SPECIALISTS)
    state = {'decision': {'allowed': True, 'capabilities': ['standings']}, 'messages': []}
    if check == 'graph_regression':
        from llm.v2.tests.test_chain import ChainTest, ClassifierTest
        method = row['test_method']
        owner = next((cls for cls in (ChainTest, ClassifierTest) if method.startswith('test_') and callable(getattr(cls, method, None))), None)
        if owner is None:
            raise ValueError('Unknown production graph regression')
        case = owner(method)
        case.setUp()
        try:
            getattr(case, method)()
            return {'id': row['id'], 'status': 'passed', 'evidence': 'production graph/middleware with scripted model; no live tool backend',
                    'source_test': f'llm.v2.tests.test_chain.{owner.__name__}.{method}',
                    'exposed_per_model_call': [c['tools'] for c in getattr(case, 'model_calls', [])],
                    'invoked_safe_handlers': getattr(case, 'executed', [])}
        finally:
            case.tearDown()
            case.doCleanups()
    elif check == 'allocation':
        m = DynamicToolMiddleware(row['role_tools'], CAPABILITY_TOOLS, SPECIALISTS)
        state['decision']['capabilities'] = row['capabilities']
        state['tool_group_ids'] = row.get('tool_group_ids', [])
        # Preserve the original capability oracle; add production's registered fixed delegations.
        expected = frozenset(row['expected_tools']) | (frozenset(row['role_tools']) & SPECIALISTS.keys())
        assert m.allowed(state) == expected
        state['decision']['allowed'] = False
        assert not m.allowed(state)
    elif check in ('threshold', 'invalid_guard', 'invalid_course', 'provider_error'):
        response = NS(choices={'guard': NS(choice='PASS'), 'course_request': NS(choice='NONE')},
                      nouls={c: NS(noul=0) for c in jev.CAPABILITIES}, usage=None)
        response.nouls['standings'].noul = .5
        response.nouls['schedule'].noul = .49999
        if check == 'invalid_guard': response.choices['guard'].choice = 'MAYBE'
        if check == 'invalid_course': response.choices['course_request'].choice = 'INVALID'
        with patch.object(jev, '_client') as client:
            client.return_value.invoke.return_value = response
            if check == 'provider_error': client.return_value.invoke.side_effect = TimeoutError('synthetic timeout')
            if check == 'threshold':
                assert jev.classify('q')['capabilities'] == ['standings']
                response.choices['guard'].choice = 'NON_PASS'
                assert jev.classify('q')['capabilities'] == []
            else:
                try: jev.classify('q')
                except (ValueError, TimeoutError): pass
                else: raise AssertionError('classifier did not fail closed')
    elif check in ('hidden_call', 'attachment', 'suppression'):
        seen = []
        handler = lambda req: seen.append(req.tool_call['name']) or 'executed'
        def call(name, id='a', args=None):
            return middleware.wrap_tool_call(NS(state=state, tool_call={'name': name, 'id': id, 'args': args or {}}), handler)
        if check == 'hidden_call':
            assert call('get_games').status == 'error' and not seen
        elif check == 'attachment':
            state['attachment_web_call_id'] = 'a'
            assert call('jev_read_body', 'wrong').status == 'error'
            assert call('jev_read_body', args={'url': 'https://example.org'}).status == 'error'
            assert call('jev_read_body') == 'executed'
            state['attachment_web_done'] = True
            assert call('jev_read_body').status == 'error'
            assert seen == ['jev_read_body']
        else:
            state['decision']['capabilities'] = ['day_plan', 'nearby_places']
            state['messages'] = [HumanMessage('코스'), ToolMessage('완료', name='ask_course', tool_call_id='c')]
            for n in ['ask_course', 'ask_travel_research', 'search_documents_tool']:
                assert call(n).status == 'error'
            captured = []
            request = NS(state=state, tools=[NS(name=n) for n in middleware.role_tools])
            request.override = lambda **kw: NS(**kw)
            middleware.wrap_model_call(request, lambda req: captured.extend(t.name for t in req.tools))
            assert not set(captured) & {'ask_course', 'ask_travel_research', 'search_documents_tool'}
            assert not seen
    elif check == 'specialist':
        with patch.object(jev, 'classify', side_effect=AssertionError('reclassification')):
            assert jev.JevGuidelineMiddleware('', False).before_agent(state, None) is None
            assert DynamicToolMiddleware(['get_games']).allowed(state) == frozenset(['get_games'])
    elif check == 'course_transition':
        from langgraph.graph.message import add_messages
        from llm.v1.rag.course.memory import empty
        current = {'stadium': 'JAMSIL', 'origin': {'lat': 37}, 'routePath': [1],
                   'currentCourse': {'places': [{'name': 'old', 'category': 'CE7', 'visitId': 'v1'}], 'writerState': {'origin': {'lat': 38}}}}
        for mode in ['NEW', 'EDIT']:
            state = {'messages': [HumanMessage('이전 조건', id='old'), AIMessage('이전 코스', id='answer'),
                                  HumanMessage('코스 짜줘', id='latest')],
                     'context': deepcopy(current), 'course_memory': {'conditions': ['old']}}
            original = deepcopy(state)
            with patch.object(jev, 'classify', return_value={'allowed': True, 'capabilities': [], 'course_request': mode}):
                out = jev.JevGuidelineMiddleware('', True).before_agent(state, None)
            assert out['decision']['capabilities'] == ['day_plan']
            assert state == original
            if mode == 'NEW':
                assert out['context'] == {'stadium': 'JAMSIL', 'origin': {'lat': 38}, 'routePath': [1]}
                assert out['course_memory'] == empty()
                assert add_messages(state['messages'], out['messages']) == [state['messages'][-1]]
            else:
                assert not {'messages', 'context', 'course_memory'} & out.keys()
                assert state['messages'] == original['messages'] and state['course_memory'] == original['course_memory']
    elif check == 'concurrency':
        # Reuse the existing actual compiled graph concurrency regression, not a duplicate graph harness.
        from llm.v2.tests.test_chain import ChainTest
        ChainTest('test_concurrent_requests_see_only_their_own_tools').test_concurrent_requests_see_only_their_own_tools()
    else: raise ValueError(check)
    return {'id': row['id'], 'status': 'passed', 'evidence': 'deterministic only; not model quality'}


def credentials(path):
    if path:
        from dotenv import dotenv_values
        from io import StringIO
        keys = ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENROUTER_API_KEY', 'LLM_MODEL')
        with path.open() as source:
            selected = ''.join(line for line in source if line.strip().split('=', 1)[0].removeprefix('export ').strip() in keys)
        values = dotenv_values(stream=StringIO(selected))
        for key in keys:
            if values.get(key): os.environ.setdefault(key, values[key])
        # Never display values or serialize environment; unrelated configuration is not applied.


def timeout(seconds):
    def expired(*_): raise TimeoutError('evaluation wall-clock limit')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)


@local_only
def semantic(rows):
    from langchain_typesafe import TypeSafeClassifier
    client = TypeSafeClassifier(base_url='https://openrouter.ai/api', api_key=os.environ['OPENROUTER_API_KEY'], model='jev-1.13', timeout=25)
    results = []
    original_invoke = TypeSafeClassifier.invoke
    for row in rows[:5]:
        raw = []
        def capture(instance, *args, **kwargs):
            response = original_invoke(instance, *args, **kwargs)
            raw.append(response.model_dump(mode='json'))
            return response
        usage = []
        try:
            timeout(35)
            with patch.object(TypeSafeClassifier, 'invoke', capture), patch.object(jev, '_client', return_value=client), patch('llm.service.usage.record_external', side_effect=lambda a,b: usage.append({'input_tokens': a, 'output_tokens': b})):
                decision = jev.classify(row['input'], messages(row), row.get('context'))
            actual = {'guard': 'PASS' if decision['allowed'] else 'NON_PASS', 'capabilities': decision['capabilities'], 'course_request': decision['course_request']}
            expected = {k: row[k] for k in actual}
            score = ExactLabels().measure(LLMTestCase(input=row['input'], actual_output=json.dumps(actual), expected_output=json.dumps(expected)))
            results.append({'id': row['id'], 'status': 'passed' if score else 'failed', 'actual': actual, 'expected': expected, 'usage': usage, 'raw_response': raw})
        except Exception as exc:
            results.append({'id': row['id'], 'status': 'execution_error', 'error_type': type(exc).__name__, 'usage': usage, 'raw_response': raw})
        finally: signal.alarm(0)
    return results


def supervisor_state(row):
    from langgraph.graph.message import add_messages
    state = {'messages': [*messages(row), HumanMessage(row['input'])],
             'context': deepcopy(row.get('context')), 'course_memory': deepcopy(row.get('course_memory', {}))}
    decision = {'allowed': row['guard'] == 'PASS', 'capabilities': list(row['capabilities']), 'course_request': row['course_request']}
    with patch.object(jev, 'classify', return_value=decision):
        update = jev.JevGuidelineMiddleware('', True).before_agent(state, None)
    if 'messages' in update:
        update['messages'] = add_messages(state['messages'], update['messages'])
    return {**state, **update}


def selection_ok(row, response):
    if response.invalid_tool_calls:
        return False
    selected = [c['name'] for c in response.tool_calls]
    expected = row['selected_tools']
    return (bool(selected) and set(selected) <= set(expected) if row.get('selection_mode') == 'subset'
            else sorted(selected) == sorted(expected))


@local_only
def supervisor(rows):
    # Real production registry/schema, model, MAIN_RULES and dynamic middleware; stop before tool execution.
    from llm.v2.agent import chain, common, sub_agents
    if chain.MAIN_MODEL_CALL_BUDGET <= 1:
        raise ValueError('Supervisor evaluation requires MAIN_MODEL_CALL_BUDGET > 1 (ORCHESTRATOR_MODEL_CALL_BUDGET); budget-one production starts without tools')
    from llm.tools import create_default_tools
    from llm.tools.knowledge import create_knowledge_tools
    from llm.tools.assistant import build_specialized_tools
    from llm.v2.agent.chat_ui import present_planning_questions
    from llm.v2.middleware.attachment_context import jev_read_body
    registry = {t.name: t for t in (*create_default_tools(), *create_knowledge_tools())}
    for t in build_specialized_tools(): registry.setdefault(t.name, t)
    tools = [*(registry[n] for n in chain.TOOLS), *sub_agents.build(common.llm(), registry), present_planning_questions, jev_read_body]
    dynamic = DynamicToolMiddleware([t.name for t in tools], CAPABILITY_TOOLS,
                                    (*sub_agents.SPECIALISTS, present_planning_questions.name))
    results = []
    for row in rows[:5]:
        state = supervisor_state(row)
        decision = state['decision']
        response = AIMessage('')
        exposed = []
        usage = None
        try:
            timeout(35)
            if not decision['allowed']:
                selected = []  # Actual guard short circuit, no supervisor call.
            else:
                from langchain.agents.middleware import ModelRequest
                request = ModelRequest(model=common.llm(), messages=state['messages'], state=state, tools=tools, runtime=None)
                def invoke(req):
                    exposed.extend(t.name for t in req.tools)
                    response = req.model.bind_tools(req.tools).invoke([req.system_message, *req.messages])
                    return response
                response = jev.JevGuidelineMiddleware(chain.MAIN_RULES, True).wrap_model_call(request, lambda req: dynamic.wrap_model_call(req, invoke))
                selected = [c['name'] for c in response.tool_calls]
                usage = response.usage_metadata
            expected = row['selected_tools']
            ok = selection_ok(row, response)
            results.append({'id': row['id'], 'status': 'passed' if ok else 'failed', 'selected': selected, 'expected': expected, 'exposed': exposed, 'usage': usage, 'raw_response': response.model_dump(mode='json'), 'scope': 'first selection only; no tools executed; independently seeded decision'})
        except Exception as exc:
            results.append({'id': row['id'], 'status': 'execution_error', 'error_type': type(exc).__name__})
        finally: signal.alarm(0)
    return results


@local_only
def generate(context_count):
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import EvolutionConfig, FiltrationConfig, StylingConfig
    from openai import OpenAI
    calls = []
    class Provider(DeepEvalBaseLLM):
        def load_model(self): return OpenAI(timeout=25, max_retries=0)
        def get_model_name(self): return self.name
        def generate(self, prompt, schema=None):
            if len(calls) >= 12: raise RuntimeError('generation call budget exhausted')
            calls.append({'status': 'started'})
            response = self.model.responses.create(model=self.name, input=prompt, max_output_tokens=1800,
                **({'text': {'format': {'type': 'json_schema', 'name': 'candidate', 'schema': schema.model_json_schema(), 'strict': False}}} if schema else {}))
            text = response.output_text
            calls[-1] = {'status': 'completed', 'output': text, 'usage': response.usage.model_dump()}
            parsed = schema.model_validate_json(text) if schema else text
            if schema and hasattr(parsed, 'data') and len(parsed.data) != 1:
                raise ValueError('Expected exactly one generated input')
            return parsed
        async def a_generate(self, prompt, schema=None): return self.generate(prompt, schema)
    contexts = [[jev.GUARD_INSTRUCTIONS, json.dumps(jev.GUARD_CRITERIA, ensure_ascii=False)],
                [json.dumps(jev.CAPABILITY_INSTRUCTIONS, ensure_ascii=False)],
                [jev.COURSE_REQUEST_INSTRUCTIONS, json.dumps(jev.COURSE_REQUEST_CRITERIA, ensure_ascii=False)]]
    model = Provider(os.getenv('LLM_MODEL') or 'gpt-6-luna')
    synth = Synthesizer(model=model, async_mode=False, evolution_config=EvolutionConfig(num_evolutions=0),
        filtration_config=FiltrationConfig(max_quality_retries=0, critic_model=model),
        styling_config=StylingConfig(input_format='한국어 사용자 질문 하나. 규칙의 예시를 복사하지 말고 반례도 포함.', task='KBO guard 및 도구 capability 평가용 질문 생성. 답변이나 RAG 검색이 아님.'))
    try:
        timeout(180)
        goldens = synth.generate_goldens_from_contexts(contexts[:context_count], include_expected_output=False, max_goldens_per_context=1, _send_data=False)
        if len(goldens) != context_count or any(not g.input.strip() or g.expected_output is not None for g in goldens):
            raise ValueError('Expected one unlabeled nonempty candidate per context')
        return {'status': 'generated', 'review_status': 'candidate/unreviewed', 'goldens': [g.model_dump(mode='json') for g in goldens], 'calls': calls}
    except Exception as exc:
        return {'status': 'generation_error', 'error_type': type(exc).__name__, 'calls': calls, 'review_status': 'candidate/unreviewed'}
    finally: signal.alarm(0)


def sanitize(value):
    if isinstance(value, dict): return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list): return [sanitize(v) for v in value]
    if isinstance(value, str):
        for key in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY'):
            secret = os.getenv(key)
            if secret: value = value.replace(secret, '[REDACTED]')
    return value


@local_only
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['offline', 'jev', 'supervisor', 'generate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--dataset', type=Path, help='Separate reviewed dataset; defaults to preserved seed28')
    parser.add_argument('--contexts', type=int, choices=[1,2,3], default=1)
    parser.add_argument('--ids', help='Comma-separated semantic seed IDs, maximum five')
    args = parser.parse_args()
    if args.output.resolve() == SEED.resolve() or args.output.exists(): parser.error('output must be a new file; never overwrite seed or frozen runs')
    if args.part == 'jev' and args.ids is None:
        parser.error('Live JEV requires an explicit --ids selection, maximum five requests')
    credentials(args.credentials)
    data = dataset(args.dataset)
    rows = [g.additional_metadata for g in data.goldens]
    if args.part == 'supervisor' and any(r['part'] == 'semantic' and 'selected_tools' not in r for r in rows):
        parser.error('Dataset has no first-selection oracle; use jev or offline')
    if args.ids is not None:
        if args.part not in ('jev', 'supervisor'): parser.error('--ids is only valid for live semantic parts')
        ids = args.ids.split(',')
        if len(set(ids)) != len(ids) or len(ids) > 5 or not set(ids) <= {r['id'] for r in rows if r['part'] == 'semantic'}:
            parser.error('--ids must select at most five existing semantic seeds')
        rows = [r for r in rows if r['id'] in ids]
    try:
        if args.part == 'offline':
            results = []
            for row in rows:
                if row['part'] != 'deterministic': continue
                try: results.append(deterministic(row))
                except Exception as exc: results.append({'id': row['id'], 'status': 'failed', 'error_type': type(exc).__name__, 'detail': str(exc)})
            result = {'part': 'offline', 'seed_count': len(rows), 'results': results}
        elif args.part == 'generate': result = generate(args.contexts)
        else: result = {'part': args.part, 'results': (semantic if args.part == 'jev' else supervisor)([r for r in rows if r['part'] == 'semantic'])}
    except Exception as exc:
        result = {'part': args.part, 'status': 'blocked', 'error_type': type(exc).__name__,
                  'reason': 'Provider initialization or credential unavailable; no quality evidence',
                  'review_status': 'candidate/unreviewed' if args.part == 'generate' else 'reviewed-seed'}
    if args.part != 'generate':
        result['selected_count'] = sum(r['part'] == ('deterministic' if args.part == 'offline' else 'semantic') for r in rows)
        if args.part != 'offline': result['selected_count'] = min(5, result['selected_count'])
        result['executed_count'] = len(result.get('results', []))
        if not result['executed_count'] and result.get('status') != 'blocked': result['status'] = 'empty_selection'
    result['dimensions'] = dimension_report(result.get('results', []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(sanitize(result), ensure_ascii=False, indent=2))
    print(f"{args.part}: results saved to {args.output}")
    return int(result.get('status') in ('blocked', 'generation_error', 'empty_selection') or any(r['status'] != 'passed' for r in result.get('results', [])))


if __name__ == '__main__': raise SystemExit(main())
