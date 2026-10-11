"""Fictional public evidence; all provider calls are mocked, never write a real shop claim."""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import TestCase, override_settings
from django.utils import timezone

from llm.v1.rag.course import evidence_memory as evidence
from travel.place_knowledge_models import PlaceKnowledgeObservation, PlaceKnowledgeSource, PlaceEnrichmentAttempt


POLICY = [{"url_prefix": "https://example.com/", "attributes": ["menu", "cuisine", "review_feature"],
           "reference": "직접 작성한 가상 테스트 데이터", "days": 7}]
PLACE = {"placeId": "fixture:course-evidence", "name": "가상 메뉴 식당", "address": "서울 송파구 올림픽로 10",
         "lat": 37.51, "lng": 127.08, "category": "FOOD"}


def requirement(**values):
    return evidence.Requirement(term="돈까스", attribute="menu", intent="required", group="meal", **values)


def finding(**values):
    item = evidence.Finding.model_validate({
        "place_id": PLACE["placeId"], "term": "돈까스", "attribute": "menu", "name": PLACE["name"],
        "address": PLACE["address"], "url": "https://example.com/menu", "kind": "menu_listing",
        "polarity": "positive", "basis": "menu_listing", "body_read": True, "evidence": "돈까스 메뉴 표기",
        "observed_on": "", "general_context": True, "promotion": "unknown", "review_category": "food", **values})
    # This helper represents the output of the independently mocked page reader.
    item._body_verified = True
    return item


@override_settings(COURSE_WEB_VERIFICATION_ENABLED=True, PLACE_EVIDENCE_STORAGE_POLICIES=POLICY)
class CourseEvidenceTests(TestCase):
    def test_closing_time_fallback_reuses_verified_menu_within_turn_without_storage_permission(self):
        second = {**PLACE, 'placeId':'fixture:second'}
        proof = [finding(), finding(place_id=second['placeId'])]
        with self.settings(PLACE_EVIDENCE_STORAGE_POLICIES=[]), evidence.request_budget(), \
                patch.object(evidence, 'requirements', return_value=[requirement()]), \
                patch.object(evidence, 'search', return_value=(proof, {'https://example.com/menu'}, 1)) as search:
            first = evidence.enrich([PLACE, second], ['돈까스'])
            fallback = evidence.enrich([dict(second)], ['돈까스'])
            self.assertEqual(len(first), 2)
            self.assertEqual([p['placeId'] for p in fallback], [second['placeId']])
            self.assertEqual(PlaceKnowledgeObservation.objects.count(), 0)
            search.assert_called_once()
        self.assertIsNone(evidence._BUDGET.get())

    def setUp(self):
        self.now = timezone.now()
        self.requirement = requirement()
        self.place = evidence.ensure_place(PLACE, self.now)
        self.opened = {"https://example.com/menu"}

    def rows(self, findings=None):
        return evidence.checked_rows(PLACE, [self.requirement], findings or [finding()], self.opened, self.now)

    def test_cuisine_category_does_not_require_duplicate_web_proof_but_menu_still_does(self):
        from llm.v1.rag.course import agent
        raw = evidence.Requirements(items=[
            evidence.Requirement(term="중식", attribute="cuisine", intent="required", group="type"),
            evidence.Requirement(term="자장면", attribute="menu", intent="required", group="meal"),
        ])
        with patch.object(agent, "llm") as model, patch.object(evidence, "search", return_value=([finding(term="자장면")], self.opened, 1)) as search:
            model.return_value.with_structured_output.return_value.invoke.return_value = raw
            result = evidence.enrich([{**PLACE, "detail": "음식점 > 중식"}], ["중식", "자장면", "중식 자장면"])
        self.assertEqual([(r.attribute, r.term) for r in search.call_args.args[1]], [("menu", "자장면")])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["conditionChecks"][0]["status"], "match")
        self.assertEqual(result[0]["detail"], "음식점 > 중식")

    def test_catalog_cuisine_conversion_preserves_exclusion_groups_and_specific_traits(self):
        from llm.v1.rag.course import agent
        raw = evidence.Requirements(items=[
            evidence.Requirement(term="일식", attribute="cuisine", intent="exclude", group="avoid"),
            evidence.Requirement(term="비건", attribute="cuisine", intent="required", group="diet"),
            evidence.Requirement(term="짜장밥", attribute="menu", intent="required", group="meal"),
        ])
        with patch.object(agent, "llm") as model:
            model.return_value.with_structured_output.return_value.invoke.return_value = raw
            parsed = evidence.requirements(["일식 제외", "비건", "짜장밥"])
        self.assertEqual(parsed[0].model_dump(), {"term": "일식", "attribute": "catalog", "intent": "exclude", "group": "avoid"})
        self.assertEqual(parsed[1:], raw.items[1:])

    def test_verified_menu_alias_is_reused_without_inventing_a_new_fact(self):
        req = self.requirement.model_copy(update={"term": "짜장면"})
        rows = evidence.checked_rows(PLACE, [req], [finding(term="짜장면")], self.opened, self.now)
        evidence.store(PLACE, rows, self.now)
        requested = req.model_copy(update={"term": "자장면"})
        reused = evidence.read(PLACE, requested)
        self.assertEqual(evidence.verdict(reused, requested), "match")
        self.assertEqual(PlaceKnowledgeObservation.objects.get().term, "짜장면")
        unrelated = req.model_copy(update={"term": "짜장밥"})
        self.assertEqual(evidence.verdict(evidence.read(PLACE, unrelated), unrelated), "unknown")

    def test_corrected_menu_spelling_rechecks_legacy_misses_and_still_cools_new_failures(self):
        req = self.requirement.model_copy(update={"term": "자장면"})
        fields = dict(place=self.place, attribute="menu", term=req.term, status="no_evidence", reason_code="not_found",
                      started_at=self.now, finished_at=self.now, next_retry_at=self.now + timedelta(minutes=60))
        PlaceEnrichmentAttempt.objects.create(attempt_key=evidence.VERSION + ":legacy", **fields)
        self.assertEqual(evidence.cooling_terms(self.place, [req], self.now), set())
        PlaceEnrichmentAttempt.objects.create(attempt_key=evidence.attempt_prefix(req) + "corrected", **fields)
        self.assertEqual(evidence.cooling_terms(self.place, [req], self.now), {("menu", "자장면")})

    def test_live_verified_keyword_roundtrip_reuses_without_web_and_keeps_no_raw_body(self):
        with patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", return_value=([finding()], self.opened, 3)) as search:
            first = evidence.enrich([PLACE], ["돈까스 판매 메뉴"])
            second = evidence.enrich([PLACE], ["돈까스 판매 메뉴"])
        search.assert_called_once()
        self.assertEqual(first[0]["conditionChecks"][0]["status"], "match")
        self.assertEqual(second[0]["conditionChecks"][0]["status"], "match")
        self.assertEqual(PlaceKnowledgeObservation.objects.count(), 1)
        row = PlaceKnowledgeObservation.objects.get()
        self.assertEqual(row.summary, "menu:돈까스 [positive]")
        self.assertNotIn("메뉴 표기", row.summary)
        self.assertEqual(PlaceEnrichmentAttempt.objects.get().search_calls, 3)

    def test_unknown_wrong_branch_snippet_and_unrequested_keyword_cannot_be_stored(self):
        cases = [finding(address="서울 강남구 테헤란로 10"), finding(body_read=False), finding(term="피자"),
                 finding(general_context=False), finding(url="https://example.com/not-opened")]
        for f in cases:
            self.assertEqual(self.rows([f]), [])
        self.assertEqual(PlaceKnowledgeObservation.objects.count(), 0)

    def test_storage_policy_is_required_and_revocation_blocks_reuse(self):
        with self.settings(PLACE_EVIDENCE_STORAGE_POLICIES=[]):
            self.assertEqual(evidence.store(PLACE, self.rows(), self.now), 0)
        evidence.store(PLACE, self.rows(), self.now)
        source = PlaceKnowledgeSource.objects.get()
        source.storage_policy = "blocked"
        source.save()
        self.assertEqual(evidence.read(PLACE, self.requirement), [])

    def test_one_review_conflicts_and_absence_never_become_confirmed_traits(self):
        req = evidence.Requirement(term="조용함", attribute="review_feature", intent="required", group="quiet")
        row = {"attribute": "review_feature", "term": "조용함", "polarity": "positive",
               "experience_key": "one", "source": {"url": "https://example.com/review1"}}
        self.assertEqual(evidence.verdict([row, row], req), "unknown")
        another = {**row, "experience_key": "two"}
        self.assertEqual(evidence.verdict([row, another], req), "match")
        self.assertEqual(evidence.verdict([row, another, {**row, "polarity": "negative"}], req), "unknown")
        self.assertEqual(evidence.verdict([], req), "unknown")

    def test_optional_unknown_is_not_a_rejection_and_negative_required_is(self):
        for intent, count in (("optional", 1), ("required", 0), ("exclude", 0)):
            req = self.requirement.model_copy(update={"intent": intent})
            with patch.object(evidence, "requirements", return_value=[req]), patch.object(evidence, "search", return_value=([], set(), 1)):
                self.assertEqual(len(evidence.enrich([PLACE], ["메뉴 조건"])), count)

    def test_existing_cache_does_not_rank_before_newly_verified_candidate(self):
        evidence.store(PLACE, self.rows(), self.now)
        closer = {**PLACE, "placeId": "fixture:closer", "name": "가상 가까운 식당", "lat": 37.511}
        f = finding(place_id=closer["placeId"], name=closer["name"], url="https://example.com/closer")
        with patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", return_value=([f], {f.url}, 2)):
            result = evidence.enrich([closer, PLACE], ["돈까스 판매 메뉴"])
        self.assertEqual([p["placeId"] for p in result], [closer["placeId"], PLACE["placeId"]])

    def test_expired_evidence_and_explicit_negative_menu(self):
        evidence.store(PLACE, self.rows(), self.now)
        future = self.now + timedelta(days=8)
        with patch("travel.place_keyword_memory.timezone.now", return_value=future):
            self.assertEqual(evidence.read(PLACE, self.requirement), [])
        negative = finding(polarity="negative", basis="explicit_non_sale")
        rows = self.rows([negative])
        req = self.requirement.model_copy(update={"intent": "exclude"})
        self.assertEqual(evidence.verdict(rows, req), "match")
        self.assertEqual(evidence.verdict([], req), "unknown")

    def test_provider_failure_is_cooled_down_and_does_not_create_false_evidence(self):
        with patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", side_effect=TimeoutError("private-provider-error")) as search:
            self.assertEqual(evidence.enrich([PLACE], ["돈까스 메뉴"]), [])
            self.assertEqual(evidence.enrich([PLACE], ["돈까스 메뉴"]), [])
        search.assert_called_once()
        self.assertEqual(PlaceKnowledgeObservation.objects.count(), 0)
        self.assertEqual(PlaceEnrichmentAttempt.objects.get().reason_code, "provider_error")

    def test_policy_blocked_search_records_zero_provider_model_calls(self):
        from llm.v2.agent.browser_research import allow_new_source_search
        token = allow_new_source_search.set(False)
        try:
            with patch.object(evidence, "requirements", return_value=[self.requirement]), patch("openai.OpenAI") as provider:
                self.assertEqual(evidence.enrich([PLACE], ["돈까스 메뉴"]), [])
            provider.assert_not_called()
        finally:
            allow_new_source_search.reset(token)
        attempt = PlaceEnrichmentAttempt.objects.get()
        self.assertEqual((attempt.search_calls, attempt.model_calls), (0, 0))
        self.assertEqual((attempt.status, attempt.reason_code), ("api_error", "provider_error"))
        self.assertEqual(PlaceKnowledgeObservation.objects.count(), 0)

    def test_turn_budget_spans_nested_lookups_and_resets_for_next_turn(self):
        places = [{**PLACE, "placeId": f"fixture:budget-{i}"} for i in range(3)]
        with patch("llm.v1.rag.course.agent.llm") as model, \
                patch.object(evidence, "search", return_value=([], set(), 1)) as search:
            parse = model.return_value.with_structured_output.return_value.invoke
            parse.return_value = evidence.Requirements(items=[self.requirement])
            with evidence.request_budget():
                evidence.enrich([places[0]], ["돈까스 메뉴"])
                with evidence.request_budget():
                    evidence.enrich([places[1]], ["돈까스 메뉴"])
                evidence.enrich([places[2]], ["돈까스 메뉴"])
                self.assertEqual(search.call_count, evidence.MAX_PASSES)
                self.assertEqual(parse.call_count, 1)
            with evidence.request_budget():
                evidence.enrich([places[2]], ["돈까스 메뉴"])
            self.assertEqual(search.call_count, evidence.MAX_PASSES * 2)
            self.assertEqual(parse.call_count, 2)
        self.assertIsNone(evidence._BUDGET.get())

    def test_menu_hint_guides_investigation_but_never_substitutes_for_proof(self):
        generic = [{**PLACE, "placeId": f"fixture:generic-{i}", "name": "가상 삼계탕"} for i in range(4)]
        menu = {**PLACE, "placeId": "fixture:menu", "name": "가상 돈카츠집", "detail": "일식 > 돈까스"}
        with patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", return_value=([], set(), 1)) as search:
            self.assertEqual(evidence.enrich([*generic, menu], ["돈까스"]), [])
        self.assertEqual(search.call_args_list[0].args[0][0], menu)
        self.assertEqual(len(search.call_args_list[0].args[0]), 4)

    def test_cooling_first_four_candidates_does_not_hide_other_candidates(self):
        candidates = [{**PLACE, "placeId": f"fixture:cool-{i}"} for i in range(5)]
        with patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", return_value=([], set(), 1)) as search:
            with patch.object(evidence, "MAX_PASSES", 1):
                evidence.enrich(candidates[:4], ["돈까스"])
            evidence.enrich(candidates, ["돈까스"])
        self.assertEqual(search.call_count, 1 + evidence.MAX_PASSES)
        self.assertEqual(search.call_args.args[0], [candidates[4]])

    def test_actual_keyword_hit_is_investigated_before_name_hint_but_keeps_candidate_order(self):
        named = {**PLACE, "placeId": "fixture:named", "name": "가상 돈까스집"}
        keyword_hit = {**PLACE, "placeId": "fixture:keyword", "name": "가상 북쪽식당", "_menu_queries": ["돈까스"]}
        items = [named, keyword_hit]
        findings = [finding(place_id=p["placeId"], name=p["name"]) for p in items]
        with patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", return_value=(findings, self.opened, 1)) as search:
            result = evidence.enrich(items, ["돈까스 메뉴"])
        self.assertEqual(search.call_args.args[0], [keyword_hit, named])
        self.assertEqual([p["placeId"] for p in result], [p["placeId"] for p in items])

    def test_review_dates_advertising_and_limited_context_are_not_generic_traits(self):
        req = evidence.Requirement(term="조용함", attribute="review_feature", intent="required", group="quiet")
        valid = finding(term="조용함", attribute="review_feature", kind="customer_review", basis="customer_experience",
                        observed_on=self.now.date().isoformat(), promotion="not_disclosed")
        for changes in ({"observed_on": ""}, {"observed_on": (self.now + timedelta(days=1)).date().isoformat()},
                        {"observed_on": (self.now - timedelta(days=181)).date().isoformat()},
                        {"promotion": "disclosed"}, {"promotion": "unknown"}, {"general_context": False},
                        {"kind": "official"}):
            self.assertEqual(evidence.checked_rows(PLACE, [req], [valid.model_copy(update=changes)], self.opened, self.now), [])

    def test_and_or_exclusion_logic_does_not_invent_missing_evidence(self):
        udon = self.requirement.model_copy(update={"term": "우동", "group": "second"})
        evidence.store(PLACE, self.rows(), self.now)
        with self.settings(COURSE_WEB_VERIFICATION_ENABLED=False):
            for requests, count in (([self.requirement, udon], 0),
                                    ([self.requirement, udon.model_copy(update={"group": "meal"})], 1),
                                    ([self.requirement, udon.model_copy(update={"intent": "exclude"})], 0)):
                with patch.object(evidence, "requirements", return_value=requests):
                    self.assertEqual(len(evidence.enrich([PLACE], ["복합 조건"])), count)

    def test_search_counts_search_actions_only_and_rejects_failed_page_open(self):
        actions = [{"type": "web_search_call", "status": status, "action": {"type": kind, "url": url, "sources": None}}
                   for kind, status, url in (("search", "completed", ""), ("open_page", "failed", "https://example.com/failed"),
                       ("open_page", "completed", "https://example.com/menu"), ("find_in_page", "completed", "https://example.com/menu"))]
        response = SimpleNamespace(status="completed", usage=None, output_text='{"items":[]}',
                                   output=[Mock(model_dump=Mock(return_value=a)) for a in actions])
        with patch.object(evidence, "structured_search", return_value=response):
            _, opened, calls = evidence.search([PLACE], [self.requirement])
        self.assertEqual(calls, 1)
        self.assertEqual(opened, {"https://example.com/menu"})

    def test_unknown_optional_trait_has_no_confirmed_reason_or_citation(self):
        place = {"conditionChecks": [{"attribute": "menu", "term": "돈까스", "status": "match", "intent": "required"},
                                     {"attribute": "review_feature", "term": "조용함", "status": "unknown", "intent": "optional"}],
                 "verifiedFacts": [{"attribute": "menu", "term": "돈까스", "url": "https://example.com/menu"},
                                   {"attribute": "review_feature", "term": "조용함", "url": "https://example.com/review"}]}
        self.assertEqual(evidence.reason(place), "돈까스 근거 확인 · 조용함 미확인")
        self.assertIn("https://example.com/menu", evidence.citation_links(place))
        self.assertNotIn("review", evidence.citation_links(place))

    def test_verification_instruction_is_not_an_extra_shop_requirement(self):
        meta = self.requirement.model_copy(update={"term": "실제 이용 후기 근거", "attribute": "review_feature"})
        with patch("llm.v1.rag.course.agent.llm") as model:
            model.return_value.with_structured_output.return_value.invoke.return_value = evidence.Requirements(items=[self.requirement, meta])
            self.assertEqual(evidence.requirements(["돈까스 메뉴 근거를 확인"]), [self.requirement])

    def test_timeline_cites_only_verified_matching_keyword(self):
        from llm.v1.rag.course import timeline
        place = {**PLACE, "conditionChecks": [{"attribute": "menu", "term": "돈까스", "status": "match", "intent": "required"}],
                 "verifiedFacts": [{"attribute": "menu", "term": "돈까스", "url": "https://example.com/menu"}]}
        text = timeline.text_lines([{"key": "p", "phase": "BEFORE", "reason": evidence.reason(place)}], {"p": place},
                                   {"rows": [{"time": "12:00", "stayMin": 50}]})[0]
        self.assertIn("[메뉴 근거](https://example.com/menu)", text)

    def test_visiting_a_citation_does_not_replace_independent_body_verification(self):
        claim = finding()
        claim._body_verified = False
        self.assertEqual(self.rows([claim]), [])
        with patch.object(evidence, "_search_once", return_value=([claim], self.opened, 1, [])), \
                patch("llm.v1.rag.course.grounding.verify", return_value=[]) as verify:
            findings, _, calls = evidence.search([PLACE], [self.requirement])
        self.assertEqual(findings, [])
        self.assertEqual(calls, 1)
        verify.assert_called_once()

    def test_real_menu_body_can_supply_proof_when_search_model_omits_findings(self):
        url = "https://www.diningcode.com/profile.php?rid=fixture"
        page = {"body_read": True, "title": PLACE["name"] + " - 음식점", "status": "read",
                "body_text": PLACE["name"] + "\n" + PLACE["address"] + "\n메뉴정보\n돈까스\n12000 원\n메뉴 더보기"}
        with patch.object(evidence, "_search_once", return_value=([], {url}, 1, [])), \
                patch("llm.v1.rag.course.grounding.PublicReader") as reader:
            reader.return_value.read.return_value = page
            findings, opened, calls = evidence.search([PLACE], [self.requirement])
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0]._body_verified)
        self.assertEqual(findings[0].evidence, "돈까스")
        self.assertEqual(calls, 1)
        self.assertTrue(evidence.checked_rows(PLACE, [self.requirement], findings, opened, self.now))

    def test_empty_first_batch_moves_to_new_candidates_and_stops_on_verified_result(self):
        items = [{**PLACE, "placeId": f"fixture:batch-{i}"} for i in range(9)]
        hit = finding(place_id=items[5]["placeId"])
        with patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", side_effect=[([], set(), 1), ([hit], self.opened, 1)]) as search:
            result = evidence.enrich(items, ["돈까스"])
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args_list[0].args[0], items[:4])
        self.assertEqual(search.call_args_list[1].args[0], items[4:8])
        self.assertEqual([p["placeId"] for p in result], [items[5]["placeId"]])

    def test_prior_extractor_claims_are_not_reused_without_body_verification(self):
        evidence.store(PLACE, self.rows(), self.now)
        PlaceKnowledgeObservation.objects.update(extractor_version="course-evidence-v3")
        self.assertEqual(evidence.read(PLACE, self.requirement), [])

    def test_alternative_source_pass_can_recover_the_same_candidate(self):
        with patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", side_effect=[([], set(), 1), ([finding()], self.opened, 1)]) as search:
            result = evidence.enrich([PLACE], ["돈까스"])
        self.assertEqual(search.call_count, 2)
        self.assertEqual(result[0]["conditionChecks"][0]["status"], "match")

    def test_later_success_clears_earlier_failure_cooldown_without_storing_unapproved_facts(self):
        with self.settings(PLACE_EVIDENCE_STORAGE_POLICIES=[]), \
                patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search", side_effect=[([], set(), 1), ([finding()], self.opened, 1),
                                                              ([finding()], self.opened, 1)]) as search:
            self.assertEqual(len(evidence.enrich([PLACE], ["돈까스"])), 1)
            self.assertEqual(len(evidence.enrich([PLACE], ["돈까스"])), 1)
        self.assertEqual(search.call_count, 3)
        self.assertEqual(PlaceKnowledgeObservation.objects.count(), 0)

    def test_exhausted_time_budget_does_not_start_another_lookup(self):
        with evidence.request_budget(), patch.object(evidence, "requirements", return_value=[self.requirement]), \
                patch.object(evidence, "search") as search:
            evidence._BUDGET.get()["deadline"] = evidence.time.monotonic() - 1
            self.assertEqual(evidence.enrich([PLACE], ["돈까스"]), [])
        search.assert_not_called()
