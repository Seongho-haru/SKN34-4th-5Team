import uuid

from django.test import SimpleTestCase
from django.urls import resolve

from baseball.views import RESOURCE_VIEWSETS
from community.views import CommunityPostListCreateView
from llm.v1.views import ChatFinalizeView, GuestChatView
from travel.views import CourseListCreateView


class RootUrlResolutionTests(SimpleTestCase):
    def test_feature_routes_resolve_to_expected_views(self):
        turn_id = uuid.uuid4()
        cases = (
            ("/api/v1/courses/", CourseListCreateView),
            ("/api/v1/community/posts/", CommunityPostListCreateView),
            ("/api/v1/baseball/manage/teams/", RESOURCE_VIEWSETS["teams"]),
            (f"/api/v1/chat/turns/{turn_id}/finalize/", ChatFinalizeView),
            ("/api/v1/chat/guest/", GuestChatView),
        )

        for path, view_class in cases:
            with self.subTest(path=path):
                func = resolve(path).func
                self.assertIs(getattr(func, "view_class", getattr(func, "cls", None)), view_class)
