"""소유권 guard를 통과한 disposable PostgreSQL에서만 실행하는 통합 검사."""

import os
import unittest
from datetime import date, datetime, time, timezone
from decimal import Decimal

import django
import psycopg
from psycopg import sql as pg_sql

if os.environ.get("DJANGO_SETTINGS_MODULE") != "baseball.tests.integration_settings":
    raise RuntimeError("run through run_postgres_integration.py")
django.setup()

from django.conf import settings  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.core.management.base import CommandError  # noqa: E402

from baseball import models  # noqa: E402
from baseball.serializers import RESOURCE_MODELS  # noqa: E402
from baseball.query_repository import (  # noqa: E402
    BaseballQueryAccessDeniedError,
    BaseballQueryLockTimeoutError,
    BaseballQueryTimeoutError,
)
from baseball.query_service import BaseballQueryService  # noqa: E402
from llm.v1.tools import create_baseball_tools  # noqa: E402


class BaseballPostgresIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owner = settings.DATABASES["default"]
        reader = settings.DATABASES["baseball_readonly"]
        assert owner["USER"] != reader["USER"], "owner와 reader 계정은 달라야 합니다."
        cls.owner_dsn = {
            "dbname": owner["NAME"],
            "user": owner["USER"],
            "password": owner["PASSWORD"],
            "host": owner["HOST"],
            "port": owner["PORT"],
        }
        cls.reader_dsn = {
            "dbname": reader["NAME"],
            "user": reader["USER"],
            "password": reader["PASSWORD"],
            "host": reader["HOST"],
            "port": reader["PORT"],
        }
        cls._create_fixtures()
        cls.service = BaseballQueryService()

    @classmethod
    def _create_fixtures(cls):
        for model in reversed(RESOURCE_MODELS.values()):
            model.objects.all().delete()
        now = datetime(2026, 9, 14, tzinfo=timezone.utc)
        team1 = models.Team.objects.create(id=1, team_code="H", team_name_ko="홈")
        team2 = models.Team.objects.create(id=2, team_code="A", team_name_ko="원정")
        models.Team.objects.bulk_create(
            [models.Team(id=i, team_code=f"T{i}", team_name_ko=f"팀{i}") for i in range(3, 203)]
        )
        stadium = models.Stadium.objects.create(
            id=1,
            stadium_code="S",
            stadium_name_ko="구장",
            address="서울",
            longitude=Decimal("127.00000000"),
            latitude=Decimal("37.00000000"),
            geocode_source="test",
            collected_at=now,
        )
        home = models.HomeContext.objects.create(id=1, season=2026, team=team1, stadium=stadium)
        stage = models.PostseasonStage.objects.create(
            id=1,
            stage_code="KS",
            stage_name="한국시리즈",
            start_date=date(2026, 10, 1),
            end_date=date(2026, 10, 7),
            matchup_description="테스트",
            status_tag="scheduled",
            collected_at=now,
        )
        game = models.Game.objects.create(
            id=1,
            game_code="G1",
            home_team=team1,
            away_team=team2,
            stadium=stadium,
            postseason_stage=stage,
            game_date=date(2026, 9, 14),
            game_time=time(18, 30),
            home_score=3,
            away_score=2,
            status_code="final",
            game_type="regular",
            collected_at=now,
        )
        models.StandingHistory.objects.create(
            id=1, team=team1, snapshot_date=date(2026, 9, 14), rank=1,
            wins=70, losses=40, draws=2, games_behind=Decimal("0.00"), collected_at=now,
        )
        zone = models.SeatZone.objects.create(
            id=1, home_context=home, zone_code="Z1", zone_name_ko="1루",
            level="1", side="home", seat_type="chair", accessible=True,
        )
        models.TicketPrice.objects.create(
            id=1, seat_zone=zone, price_tier="normal", day_type="weekday",
            customer_type="adult", price_krw=10000, discount_condition="none", collected_at=now,
        )
        models.TicketPolicy.objects.create(
            id=1, policy_code="P", team=team1, game=game, policy_type="open",
            subtype="general", channel_no=1, booking_channel="web",
            channel_condition="none", collected_at=now,
        )
        seat_map = models.SeatMap.objects.create(id=1, home_context=home, map_title="좌석도", page_url="https://example.test")
        models.SeatMapAsset.objects.create(id=1, seat_map=seat_map, asset_no=1, asset_url="https://example.test/map.png", asset_role="main")
        scope = models.SeatScope.objects.create(id=1, home_context=home, scope_code="SC", scope_name="내야")
        models.SeatView.objects.create(id=1, seat_scope=scope, view_characteristic="좋음", roof_coverage="일부", evidence_scope="official")
        models.Transport.objects.create(id=1, stadium=stadium, access_code="SUB", mode="subway", title="지하철", details="도보", collected_at=now)
        store = models.FoodStore.objects.create(id=1, record_code="F", stadium=stadium, store_facility="매점", collected_at=now)
        models.FoodStoreLocation.objects.create(id=1, food_store=store, location_no=1, floor="1", zone_location="중앙")
        models.FoodStoreMenu.objects.create(id=1, food_store=store, menu_category_official="식사")
        models.StadiumContent.objects.create(id=1, record_code="C", stadium=stadium, content_type="tour", name="투어", floor="1", location="중앙", official_description="설명", operating_condition="경기일", collected_at=now)
        models.Facility.objects.create(id=1, record_code="FA", stadium=stadium, facility_type="화장실", floor="1", side="home", nearby_section="A", gate="1", gender="all", indoor_outdoor="indoor", location_detail="중앙", collected_at=now)

    def test_all_tables_and_domain_queries(self):
        schema = self.service.get_baseball_schema()
        names = [table["name"] for table in schema["tables"]]
        self.assertEqual(len(names), 19)
        for table in names:
            result = self.service.execute_baseball_select(f'SELECT * FROM "{table}"', {}, 1)
            self.assertEqual(len(result["rows"]), 1, table)

        queries = (
            'SELECT h.team_name_ko, a.team_name_ko, s.stadium_name_ko FROM "GAME" g JOIN "TEAM" h ON h.id=g.home_team_id JOIN "TEAM" a ON a.id=g.away_team_id JOIN "STADIUM" s ON s.id=g.stadium_id',
            'SELECT s.stadium_name_ko, z.zone_name_ko, p.price_krw FROM "STADIUM" s JOIN "HOME_CONTEXT" hc ON hc.stadium_id=s.id JOIN "SEAT_ZONE" z ON z.home_context_id=hc.id JOIN "TICKET_PRICE" p ON p.seat_zone_id=z.id',
            'SELECT s.stadium_name_ko, l.floor, m.menu_category_official FROM "STADIUM" s JOIN "FOOD_STORE" f ON f.stadium_id=s.id JOIN "FOOD_STORE_LOCATION" l ON l.food_store_id=f.id JOIN "FOOD_STORE_MENU" m ON m.food_store_id=f.id',
            'SELECT t.team_name_ko, COUNT(*) FROM "TEAM" t JOIN "GAME" g ON g.home_team_id=t.id GROUP BY t.id',
            'SELECT * FROM "TEAM" WHERE id IN (SELECT home_team_id FROM "GAME")',
            'WITH recent AS (SELECT * FROM "GAME") SELECT * FROM recent JOIN "TEAM" t ON t.id=recent.home_team_id',
        )
        for query in queries:
            self.assertTrue(self.service.execute_baseball_select(query, {}, 10)["rows"])

    def test_langchain_tools_use_restricted_database_and_deny_writes(self):
        schema_tool, select_tool = create_baseball_tools(self.service)
        self.assertEqual(len(schema_tool.invoke({})["tables"]), 19)
        result = select_tool.invoke(
            {
                "sql": (
                    'SELECT t.team_name_ko, COUNT(g.id) AS games FROM "TEAM" t '
                    'JOIN "GAME" g ON g.home_team_id=t.id GROUP BY t.id'
                ),
                "max_rows": 10,
            }
        )
        self.assertEqual(result["columns"], ["team_name_ko", "games"])
        self.assertEqual(result["rows"], [["홈", 1]])
        self.assertEqual(
            select_tool.invoke({"sql": 'DELETE FROM "TEAM"'}),
            "단일 SELECT 문만 허용됩니다.",
        )

    def test_parameters_truncation_timeout_and_recovery(self):
        injected = "홈' OR 1=1 --"
        result = self.service.execute_baseball_select(
            'SELECT id FROM "TEAM" WHERE team_name_ko=%(name)s', {"name": injected}, 10
        )
        self.assertEqual(result["rows"], [])
        result = self.service.execute_baseball_select('SELECT id FROM "TEAM" ORDER BY id', {}, 2)
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(result["truncated"])
        with self.assertRaises(BaseballQueryTimeoutError):
            self.service.execute_baseball_select(
                'SELECT COUNT(*) FROM "TEAM" a CROSS JOIN "TEAM" b CROSS JOIN "TEAM" c CROSS JOIN "TEAM" d',
                {},
                1,
            )
        self.assertEqual(
            self.service.execute_baseball_select('SELECT COUNT(*) FROM "TEAM"', {}, 1)["rows"],
            [[202]],
        )

    def test_literal_percent_modulo_and_named_parameters_share_one_query(self):
        self.assertEqual(
            self.service.execute_baseball_select(
                (
                    'SELECT \'%(literal)s\', id % 2 FROM "TEAM" '
                    'WHERE team_name_ko LIKE \'%홈%\''
                ),
                {},
                10,
            )["rows"],
            [["%(literal)s", 1]],
        )
        result = self.service.execute_baseball_select(
            (
                'SELECT \'%(literal)s\', id % 2, team_name_ko LIKE \'홈%\' '
                'FROM "TEAM" WHERE id = %(team_id)s'
            ),
            {"team_id": 1},
            10,
        )
        self.assertEqual(result["rows"], [["%(literal)s", 1, True]])

    def test_reader_role_cannot_read_apps_or_write(self):
        with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication,
                           rolbypassrls, rolinherit
                      FROM pg_roles WHERE rolname = %s
                    """,
                    [settings.DATABASES["baseball_readonly"]["USER"]],
                )
                self.assertEqual(cursor.fetchone(), (False, False, False, False, False, False))
                cursor.execute(
                    "SELECT rolconfig FROM pg_roles WHERE rolname = %s",
                    [settings.DATABASES["baseball_readonly"]["USER"]],
                )
                self.assertIn("default_transaction_read_only=on", cursor.fetchone()[0])
        with psycopg.connect(**self.reader_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_user")
                self.assertEqual(cursor.fetchone()[0], settings.DATABASES["baseball_readonly"]["USER"])
                cursor.execute("SET default_transaction_read_only = off")
                for sql in (
                    "SELECT * FROM accounts_customuser",
                    "SELECT * FROM llm_chatmessage",
                    "CREATE TEMP TABLE forbidden_temp (id integer)",
                    'INSERT INTO "TEAM" (id, team_code, team_name_ko) VALUES (999, \'X\', \'X\')',
                ):
                    with self.subTest(sql=sql), self.assertRaises(psycopg.Error):
                        cursor.execute(sql)
                cursor.execute('SELECT COUNT(*) FROM "TEAM"')
                self.assertEqual(cursor.fetchone()[0], 202)

    def test_prepared_database_permissions_are_scoped_and_idempotent(self):
        call_command("provision_baseball_reader", prepare_db_permissions=True)
        with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT a.grantee, a.privilege_type
                       FROM pg_database d
                       CROSS JOIN LATERAL aclexplode(d.datacl) a
                       WHERE d.datname = current_database()
                         AND a.privilege_type = 'TEMPORARY'
                         AND a.grantee IN (0, (SELECT oid FROM pg_roles WHERE rolname = %s))""",
                    [settings.DATABASES["default"]["USER"]],
                )
                grantees = {row[0] for row in cursor.fetchall()}
                cursor.execute(
                    "SELECT oid FROM pg_roles WHERE rolname = %s",
                    [settings.DATABASES["default"]["USER"]],
                )
                self.assertEqual(grantees, {cursor.fetchone()[0]})

    def test_database_denial_is_distinct_and_connection_recovers(self):
        with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    pg_sql.SQL('REVOKE SELECT ON "FACILITY" FROM {}').format(
                        pg_sql.Identifier(settings.DATABASES["baseball_readonly"]["USER"])
                    )
                )
        try:
            with self.assertRaises(BaseballQueryAccessDeniedError):
                self.service.execute_baseball_select('SELECT * FROM "FACILITY"', {}, 1)
            self.assertEqual(
                self.service.execute_baseball_select('SELECT COUNT(*) FROM "TEAM"', {}, 1)["rows"],
                [[202]],
            )
        finally:
            with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        pg_sql.SQL('GRANT SELECT ON "FACILITY" TO {}').format(
                            pg_sql.Identifier(settings.DATABASES["baseball_readonly"]["USER"])
                        )
                    )

    def test_lock_timeout_is_distinct_and_connection_recovers(self):
        with psycopg.connect(**self.owner_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute('LOCK TABLE "TEAM" IN ACCESS EXCLUSIVE MODE')
                with self.assertRaises(BaseballQueryLockTimeoutError):
                    self.service.execute_baseball_select('SELECT * FROM "TEAM"', {}, 1)
                connection.rollback()
        self.assertEqual(
            self.service.execute_baseball_select('SELECT COUNT(*) FROM "TEAM"', {}, 1)["rows"],
            [[202]],
        )

    def test_public_access_is_reported_without_automatic_revoke(self):
        with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("GRANT SELECT ON accounts_customuser TO PUBLIC")
                cursor.execute('GRANT UPDATE ON "TEAM" TO PUBLIC')
        try:
            with self.assertRaises(CommandError):
                call_command("provision_baseball_reader")
            with psycopg.connect(**self.reader_dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM accounts_customuser")
                    self.assertIsNotNone(cursor.fetchone())
                    cursor.execute("SET default_transaction_read_only = off")
                    cursor.execute('UPDATE "TEAM" SET team_name_ko=\'PUBLIC 경유\' WHERE id=1')
                    self.assertEqual(cursor.rowcount, 1)
        finally:
            with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("REVOKE SELECT ON accounts_customuser FROM PUBLIC")
                    cursor.execute('REVOKE UPDATE ON "TEAM" FROM PUBLIC')
                    cursor.execute('UPDATE "TEAM" SET team_name_ko=\'홈\' WHERE id=1')

    def test_unsafe_existing_role_is_rejected(self):
        unsafe = "baseball_unsafe_reader"
        with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f'DROP ROLE IF EXISTS "{unsafe}"')
                cursor.execute(f'CREATE ROLE "{unsafe}" LOGIN PASSWORD \'disposable\' NOINHERIT')
                cursor.execute(f'GRANT pg_read_all_data TO "{unsafe}"')
        reader = settings.DATABASES["baseball_readonly"]
        original = reader["USER"], reader["PASSWORD"]
        reader["USER"], reader["PASSWORD"] = unsafe, "disposable"
        try:
            with self.assertRaises(CommandError):
                call_command("provision_baseball_reader")
        finally:
            reader["USER"], reader["PASSWORD"] = original
            with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f'REVOKE pg_read_all_data FROM "{unsafe}"')
                    cursor.execute(f'DROP ROLE "{unsafe}"')

    def test_existing_unsafe_default_privileges_are_rejected_without_changes(self):
        reader = settings.DATABASES["baseball_readonly"]["USER"]
        cases = (
            (
                pg_sql.SQL("ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO PUBLIC"),
                pg_sql.SQL("ALTER DEFAULT PRIVILEGES REVOKE SELECT ON TABLES FROM PUBLIC"),
            ),
            (
                pg_sql.SQL("ALTER DEFAULT PRIVILEGES GRANT USAGE ON SEQUENCES TO {}").format(
                    pg_sql.Identifier(reader)
                ),
                pg_sql.SQL("ALTER DEFAULT PRIVILEGES REVOKE USAGE ON SEQUENCES FROM {}").format(
                    pg_sql.Identifier(reader)
                ),
            ),
            (
                pg_sql.SQL("ALTER DEFAULT PRIVILEGES GRANT USAGE ON SCHEMAS TO {}").format(
                    pg_sql.Identifier(reader)
                ),
                pg_sql.SQL("ALTER DEFAULT PRIVILEGES REVOKE USAGE ON SCHEMAS FROM {}").format(
                    pg_sql.Identifier(reader)
                ),
            ),
            (
                pg_sql.SQL("ALTER DEFAULT PRIVILEGES GRANT EXECUTE ON FUNCTIONS TO {}").format(
                    pg_sql.Identifier(reader)
                ),
                pg_sql.SQL("ALTER DEFAULT PRIVILEGES REVOKE EXECUTE ON FUNCTIONS FROM {}").format(
                    pg_sql.Identifier(reader)
                ),
            ),
        )
        for index, (grant, revoke) in enumerate(cases):
            with self.subTest(index=index):
                with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(grant)
                try:
                    with self.assertRaises(CommandError):
                        call_command("provision_baseball_reader")
                    if index == 0:
                        with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
                            with connection.cursor() as cursor:
                                cursor.execute("CREATE TABLE default_acl_future_secret (secret text)")
                        with psycopg.connect(**self.reader_dsn, autocommit=True) as connection:
                            with connection.cursor() as cursor:
                                cursor.execute("SELECT * FROM default_acl_future_secret")
                finally:
                    with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
                        with connection.cursor() as cursor:
                            cursor.execute("DROP TABLE IF EXISTS default_acl_future_secret")
                            cursor.execute(revoke)

    def test_provisioning_failure_rolls_back_new_role(self):
        rollback_role = "baseball_rollback_reader"
        with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    pg_sql.SQL("REVOKE TEMPORARY ON DATABASE {} FROM {}").format(
                        pg_sql.Identifier(settings.DATABASES["default"]["NAME"]),
                        pg_sql.Identifier(settings.DATABASES["default"]["USER"]),
                    )
                )
                cursor.execute(
                    pg_sql.SQL("GRANT TEMPORARY ON DATABASE {} TO PUBLIC").format(
                        pg_sql.Identifier(settings.DATABASES["default"]["NAME"])
                    )
                )
                cursor.execute('ALTER TABLE "FACILITY" RENAME TO "FACILITY_MISSING"')
        reader = settings.DATABASES["baseball_readonly"]
        original = reader["USER"], reader["PASSWORD"]
        reader["USER"], reader["PASSWORD"] = rollback_role, "disposable"
        try:
            with self.assertRaises(CommandError):
                call_command("provision_baseball_reader", prepare_db_permissions=True)
            with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [rollback_role])
                    self.assertIsNone(cursor.fetchone())
                    cursor.execute(
                        """SELECT a.grantee FROM pg_database d
                           CROSS JOIN LATERAL aclexplode(d.datacl) a
                           WHERE d.datname = current_database()
                             AND a.privilege_type = 'TEMPORARY'
                             AND a.grantee IN (0, (SELECT oid FROM pg_roles WHERE rolname = %s))""",
                        [settings.DATABASES["default"]["USER"]],
                    )
                    self.assertEqual([row[0] for row in cursor.fetchall()], [0])
        finally:
            reader["USER"], reader["PASSWORD"] = original
            with psycopg.connect(**self.owner_dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        pg_sql.SQL("GRANT TEMPORARY ON DATABASE {} TO {}").format(
                            pg_sql.Identifier(settings.DATABASES["default"]["NAME"]),
                            pg_sql.Identifier(settings.DATABASES["default"]["USER"]),
                        )
                    )
                    cursor.execute(
                        pg_sql.SQL("REVOKE TEMPORARY ON DATABASE {} FROM PUBLIC").format(
                            pg_sql.Identifier(settings.DATABASES["default"]["NAME"])
                        )
                    )
                    cursor.execute('ALTER TABLE "FACILITY_MISSING" RENAME TO "FACILITY"')


if __name__ == "__main__":
    unittest.main(verbosity=2)
