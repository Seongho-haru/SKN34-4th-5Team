"""Hand-authored five-group fixtures; refuse replacement, freeze split before scoring."""
import json
from pathlib import Path

HERE = Path(__file__).parent
AS_OF = '2026-10-10T18:00:00+09:00'


def build():
    groups = {name: [] for name in ('first_action', 'clarification', 'delegation', 'schedule', 'final_answer')}
    def add(group, slug, query, rubric, expected, observations=None, context=None, decision=None, family=None):
        row = dict(id=f'{group}-{slug}', family=family or f'{group}-{slug}', input=query,
                   provenance='hand-authored', review_status='author-reviewed', synthetic=True,
                   as_of=AS_OF, timezone='Asia/Seoul', context=context or {},
                   observations=observations or [], rubric=rubric, expected=expected)
        if decision: row['decision'] = decision
        groups[group].append(row)
    def first(slug, query, tools, caps=(), context=None, course='NONE', family=None):
        add('first_action', slug, query, 'Independent first-call oracle; prerequisite subset or no call, not classifier quality.',
            {'allowed_first': tools, 'no_call': not tools}, context=context,
            decision={'guard': 'PASS', 'capabilities': list(caps), 'course_request': course}, family=family)
    first('exact-schedule', '경기일정 조회 해줘', ['get_games'], ['schedule'], family='schedule-unspecified')
    first('exact-nearby', '음 그냥 가고싶은데 주변에서 뭘해야해?', ['ask_baseball'], family='nearby-unresolved')
    first('schedule-paraphrase', '팀은 상관없고 야구 일정 보여줘', ['get_games'], ['schedule'], family='schedule-unspecified')
    first('nearby-paraphrase', '야구장 주변에 가고 싶은데 어디인지 아직 못 정했어', ['ask_baseball'], family='nearby-unresolved')
    first('greeting', '안녕 야구 보러 왔어', [])
    first('thanks', '고마워 도움이 됐어', [])
    first('standings', '현재 저장된 순위를 보여줘', ['get_standings'], ['standings'])
    first('player', 'LG 김현수 선수 기록 찾아줘', ['search_players'], ['players'])
    first('stadium-list', '구장 목록을 보여줘', ['get_stadiums'], ['stadium_info'])
    first('course-known', '잠실 직관 코스 짜줘', ['ask_course'], ['day_plan'], course='NEW', context={'stadium': 'JAMSIL'})
    first('course-edit', '현재 코스에서 카페만 빼줘', ['ask_course'], ['day_plan'], course='EDIT', context={'stadium': 'JAMSIL', 'currentCourse': {'places': [{'name': '합성 카페', 'visitId': 'v1'}]}})
    first('seat-prerequisite', '잠실 좌석 시야 확인해줘', ['get_stadium'], ['stadium_info'])

    candidates = ['잠실', '사직', '고척']
    def obs(id, tool, data): return {'evidence_id': id, 'tool': tool, 'data': data}
    for slug, query, kind, choices, context in [
        ('stadium', '직관할 구장 후보 중 하나를 고르고 싶어', 'enumerable', candidates, {}),
        ('teams', '응원팀을 정해서 구장 안내받고 싶어', 'enumerable', ['LG', '두산', '롯데'], {}),
        ('players', '동명이인 합성 선수 중 누구인지 고를게', 'enumerable', ['합성 선수 A', '합성 선수 B'], {}),
        ('places', '검색된 카페 중 하나를 고를게', 'enumerable', ['합성 카페 A', '합성 카페 B'], {}),
        ('ten', '열 개 구장 후보를 모두 보여줘', 'enumerable', [f'합성 구장 {i}' for i in range(10)], {}),
        ('eleven', '많은 구장 후보 중에서 고르고 싶어', 'partial', [f'합성 구장 {i}' for i in range(10)], {}),
        ('known-screen', '선택한 잠실 일정 보여줘', 'known', [], {'stadium': 'JAMSIL'}),
        ('known-history', '아까 선택한 사직으로 이어가자', 'known', [], {'confirmed_stadium': '사직'}),
        ('free-preference', '동행에게 바라는 분위기를 직접 적고 싶어', 'open', [], {}),
        ('lookup-failed', '구장 후보 조회가 실패했는데 직접 적을게', 'failure', [], {}),
        ('followup', '앞서 고른 잠실로 최종 안내해줘', 'known', [], {'confirmed_stadium': '잠실'}),
        ('course-known', '고척 코스는 날짜 없어도 우선 만들어줘', 'known', [], {'stadium': 'GOCHEOK'}),
    ]:
        lookup = 'search_players' if slug == 'players' else 'search_places' if slug == 'places' else 'ask_baseball'
        events = ['lookup', 'ui', 'user_answer', 'followup_lookup', 'final'] if kind in ('enumerable', 'partial') else ['ui', 'user_answer', 'final'] if kind in ('open', 'failure') else ['final']
        data = {'candidates': choices + (['합성 구장 10'] if kind == 'partial' else []), 'status': 'error' if kind == 'failure' else 'success'}
        notices = ['partial_candidates'] if kind == 'partial' else ['lookup_failed'] if kind == 'failure' else []
        add('clarification', slug, query, 'Frozen observations; main-owned UI then wait for user, free input available; never re-ask known context. Final after follow-up on next turn.',
            {'required_events': events, 'order': list(zip(events, events[1:])), 'forbidden_tools': ['present_planning_questions'] if kind == 'known' else [],
             'ui': {'required': kind != 'known', 'choices': choices, 'offer_writer': False, 'free_input': True}, 'notices': notices},
            [obs(f'clarify-{slug}-candidates', lookup, data)] if kind not in ('known', 'open') else [], context)

    delegation = [
        ('direct', '저장된 순위 하나만 조회해줘', ['get_standings'], [], []),
        ('batch', '팀 상관없이 이번 주 일정을 한 번에 조회해줘', ['get_games'], [], []),
        ('short-sequence', '잠실 ID 확인 후 교통 조회해줘', ['get_stadium', 'get_transport'], [('get_stadium', 'get_transport')], []),
        ('missing-primitive', '일정 도구가 없으면 야구 전문 도구로 일정만 확인해줘', ['ask_baseball'], [], []),
        ('candidate-narrow', '구장 목록 도구가 없으면 구장 후보만 가져와줘', ['ask_baseball'], [], []),
        ('independent', '독립적으로 잠실 교통과 공개 코스 조사해줘', ['ask_baseball', 'ask_place_data'], [], [['ask_baseball', 'ask_place_data']]),
        ('independent-web', '공개 코스와 별도 웹 공지를 각각 확인해줘', ['ask_place_data', 'ask_web_research'], [], [['ask_place_data', 'ask_web_research']]),
        ('dependent', '구장 확인 결과를 바탕으로 주변 후보 조사해줘', ['ask_baseball', 'ask_travel_research'], [('ask_baseball', 'ask_travel_research')], []),
        ('internal-sequence', '전문 에이전트에서 구장 ID와 좌석을 순서대로 확인해줘', ['ask_baseball'], [], []),
        ('main-question', '전문 도구가 확인한 후보로 메인이 질문해줘', ['ask_baseball', 'present_planning_questions'], [('ask_baseball', 'present_planning_questions')], []),
        ('course-priority', '잠실 코스에 후기 좋은 카페 넣어줘', ['ask_course'], [], []),
        ('attachment', '첨부 원문부터 읽고 요약해줘', ['jev_read_body'], [], []),
    ]
    for slug, query, tools, order, parallel in delegation:
        add('delegation', slug, query, 'Direct when exposed; missing primitive narrow delegation; one goal per specialist, no nested delegation, main asks/finalizes. Parallel only independent goals; specialist internal sequence allowed.',
            {'required_events': tools + ['final'], 'order': order + [(t, 'final') for t in tools], 'parallel': parallel,
             'exact_tools': tools, 'one_goal': True, 'main_questions': True, 'no_nested': True},
            [obs(f'delegation-{slug}', tools[0], {'status': 'success', 'synthetic_note': 'frozen safe observation; not actual provider output'})])

    schedule_specs = [
        ('today', '오늘 경기 전체 상태 보여줘', '2026-10-10', '2026-10-10', False, 'available', False, 3, False, []),
        ('next', '지금 이후 다음 예정 경기 보여줘', '2026-10-10', '2027-10-11', True, 'available', False, 1, False, []),
        ('at-cutoff', '18시 정각 경기는 다음 경기에서 제외해줘', '2026-10-10', '2027-10-11', True, 'available', False, 1, False, []),
        ('exact-date', '10월 12일 경기만 보여줘', '2026-10-12', '2026-10-12', False, 'available', False, 1, False, []),
        ('explicit-range', '10월 11일부터 13일까지 조회해줘', '2026-10-11', '2026-10-13', False, 'available', False, 2, False, []),
        ('default-seven', '팀 미정 경기 일정 전체 조회해줘', '2026-10-10', '2026-10-16', False, 'available', False, 3, False, []),
        ('page-one', '일정 첫 두 건과 더 있는지 알려줘', '2026-10-10', '2026-10-16', False, 'available', False, 3, True, ['partial_results']),
        ('page-two', '다음 페이지 이어서 보여줘', '2026-10-10', '2026-10-16', False, 'available', False, 3, False, []),
        ('stale-saved', '최신 확인 안 돼도 저장 일정 안내해줘', '2026-10-10', '2026-10-16', False, 'available', True, 3, False, ['stale_saved']),
        ('stale-empty', '갱신 실패에 빈 조회면 일정 확정하지 마', '2026-10-10', '2026-10-16', False, 'unknown', True, 0, False, ['unconfirmed']),
        ('error', '조회 오류면 경기 없다고 하지 마', '2026-10-10', '2026-10-16', False, 'error', False, 0, False, ['lookup_failed']),
        ('demo', '합성 데모 일정은 실제와 구분해줘', '2026-10-10', '2026-10-16', False, 'available', False, 3, False, ['demo']),
    ]
    for slug, query, start, end, future, availability, stale, total, more, notices in schedule_specs:
        schedule = dict(start_date=start, end_date=end, upcoming_only=future, status='upcoming' if future else 'all',
                        availability=availability, stale=stale, total_count=total, has_more=more, offset=2 if slug == 'page-two' else 0)
        games = [{'game_date': '2026-10-10', 'game_time': '18:00:00', 'home_team': '합성 홈 A', 'away_team': '합성 원정 B', 'status_code': 'READY'},
                 {'game_date': '2026-10-10', 'game_time': '18:01:00', 'home_team': '합성 홈 C', 'away_team': '합성 원정 D', 'status_code': 'PREV'}]
        games = [dict(games[i % 2], game_date=start, game_time=f'18:0{i + 1}:00' if future else f'18:0{i}:00', status_code='READY' if future else ('END', 'READY', 'CANCEL')[i % 3]) for i in range(total)]
        games = games[schedule['offset']:2] if slug == 'page-one' else games[schedule['offset']:]
        add('schedule', slug, query, 'KST fixed clock; all states for today/default7, strictly future scheduled only for next. Range/pagination/freshness explicit; error/unknown are not no-game. Synthetic fixtures never real game claims.',
            {'schedule': schedule, 'notices': notices, 'strict_future': future},
            [obs(f'schedule-{slug}', 'get_games', {**schedule, 'games': games if total else [], 'demo': slug == 'demo'})])

    for slug, query, notice in [
        ('date', '경기 날짜를 출처대로 안내해줘', None), ('teams', '홈과 원정 팀을 바꾸지 마', None),
        ('status', '취소 경기는 예정으로 말하지 마', None), ('score', '확인된 점수만 안내해줘', None),
        ('link', '각 경기의 실제 관찰 출처를 인용해줘', None), ('two-games', '서로 다른 날짜 대진을 연결해서 보여줘', None),
        ('no-invented-figure', '관찰에 없는 관중 수는 만들지 마', None), ('demo', '데모 결과임을 밝히고 실제로 말하지 마', 'demo'),
        ('failure', '실패한 조회는 성공했다고 말하지 마', 'lookup_failed'), ('unknown', '미확인 시각은 불확실하다고 밝혀줘', 'unconfirmed'),
        ('stale', '저장 결과의 최신성 경고를 붙여줘', 'stale_saved'), ('conflict', '서로 충돌하는 자료는 확정하지 마', 'conflicting_sources'),
    ]:
        eid = f'answer-{slug}'
        claims = [{'evidence_id': eid, 'date': '2026-10-10', 'home': '합성 홈 A', 'away': '합성 원정 B', 'status': 'canceled' if slug == 'status' else 'scheduled',
                   'figure': '2:1' if slug == 'score' else None, 'source_url': f'https://example.invalid/evidence/{eid}', 'demo': True}]
        if slug == 'two-games': claims.append({**claims[0], 'date': '2026-10-11', 'home': '합성 홈 C', 'away': '합성 원정 D'})
        if slug in ('failure', 'unknown', 'conflict'): claims = []
        add('final_answer', slug, query, 'Claim tuples are evidence-associated structured extraction, NOT proof of prose truth. Human review must check unsupported additions/omissions, dates, teams, statuses, source links, uncertainty and demo-vs-real wording.',
            {'claims': claims, 'notices': list(dict.fromkeys(['demo'] + ([notice] if notice else [])))},
            [obs(eid, 'get_games', {'supported_claims': claims, 'status': 'error' if slug == 'failure' else 'unknown' if slug in ('unknown', 'conflict') else 'success'})])
    return groups


def main():
    groups = build()
    paths = [HERE / f'{g}.json' for g in groups] + [HERE / 'remaining_manifest.json']
    if any(p.exists() for p in paths): raise FileExistsError('Frozen datasets exist; never overwrite')
    manifest = {'schema_version': 1, 'split_basis': 'hand-authored family order, frozen before scoring; not blind or independent signoff', 'dev': [], 'heldout': []}
    for group, rows in groups.items():
        families = list(dict.fromkeys(r['family'] for r in rows))
        held = set(families[3::4])
        for r in rows: manifest['heldout' if r['family'] in held else 'dev'].append(r['id'])
        (HERE / f'{group}.json').write_text(json.dumps({'schema_version': 1, 'group': group, 'cases': rows}, ensure_ascii=False, indent=2) + '\n')
    paths[-1].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print({g: len(rows) for g, rows in groups.items()})


if __name__ == '__main__': main()
