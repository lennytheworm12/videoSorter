import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import core.database as db
from core.database import init_db, insert_insight, insert_video
from core.game_registry import analysis_spec, canonical_aoe2_civilization, find_aoe2_civilizations
import pipeline.guide_analyze as guide_analyze
import pipeline.guide_transcribe as guide_transcribe
import pipeline.embed as embed
from scrape.aoe2_import import import_rows
from scrape.aoe2_pdf_import import import_pdf
from scrape.aoe2_video_import import _classify_row, _read_url_lines, import_urls
from scrape.aoe2_wiki_scrape import discover_portal_pages, extract_page_text
import retrieval.query as retrieval_query
from retrieval.query import _aoe2_query_profile, detect_aoe2_intent
from retrieval.questions import normalize


class Aoe2PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db.DB_PATH = pathlib.Path(self._tmpdir.name) / "aoe2_test.db"
        embed._ALL_DBS = [str(db.DB_PATH)]
        init_db()
        with db.get_connection() as conn:
            conn.execute("ALTER TABLE insights ADD COLUMN embedding BLOB")
            conn.commit()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _set_embedding(self, insight_id: int, values: list[float]) -> None:
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE insights SET embedding = ? WHERE id = ?",
                (np.array(values, dtype=np.float32).tobytes(), insight_id),
            )
            conn.commit()

    def test_aoe2_subject_filter_uses_insight_subject_not_video_subject(self) -> None:
        insert_video(
            video_id="franks_video",
            video_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
            video_title="Franks guide",
            description="",
            game="aoe2",
            role="general",
            subject="Franks",
            champion=None,
            message_timestamp="2026-04-21T00:00:00+00:00",
            source="aoe2_video",
        )

        civ_id = insert_insight(
            "franks_video",
            "civilization_identity",
            "Use the Franks farm and cavalry bonuses to hit stronger scout into knight timings.",
            subject="Franks",
            subject_type="civ",
        )
        general_id = insert_insight(
            "franks_video",
            "economy_macro",
            "Keep town center uptime clean before adding extra production.",
            subject=None,
            subject_type="general",
        )
        self._set_embedding(civ_id, [1.0, 0.0, 0.0])
        self._set_embedding(general_id, [0.0, 1.0, 0.0])

        _, civ_texts, civ_meta, _ = embed.load_all_vectors(game="aoe2", subject="Franks")
        self.assertEqual(civ_texts, [
            "Use the Franks farm and cavalry bonuses to hit stronger scout into knight timings."
        ])
        self.assertEqual(civ_meta[0]["subject"], "Franks")
        self.assertEqual(civ_meta[0]["subject_type"], "civ")

        _, general_texts, general_meta, _ = embed.load_all_vectors(game="aoe2", subject=None)
        self.assertEqual(general_texts, [
            "Keep town center uptime clean before adding extra production."
        ])
        self.assertIsNone(general_meta[0]["subject"])
        self.assertEqual(general_meta[0]["subject_type"], "general")

    def test_manual_import_marks_text_rows_transcribed(self) -> None:
        inserted, transcribed = import_rows([
            {
                "title": "Hera Franks guide",
                "video_url": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
                "source": "aoe2_video",
                "civ": "Franks",
            },
            {
                "title": "Franks civilization page",
                "source": "aoe2_wiki",
                "civ": "frank",
                "content": "Franks have strong cavalry bonuses and faster berry gathering.",
            },
        ])
        self.assertEqual(inserted, 2)
        self.assertEqual(transcribed, 1)

        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT video_id, source, subject, status, transcription FROM videos ORDER BY video_id"
            ).fetchall()

        by_source = {row["source"]: row for row in rows}
        self.assertEqual(by_source["aoe2_video"]["subject"], "Franks")
        self.assertEqual(by_source["aoe2_video"]["status"], "pending")
        self.assertIsNone(by_source["aoe2_video"]["transcription"])

        self.assertEqual(by_source["aoe2_wiki"]["subject"], "Franks")
        self.assertEqual(by_source["aoe2_wiki"]["status"], "transcribed")
        self.assertIn("strong cavalry bonuses", by_source["aoe2_wiki"]["transcription"])

    def test_analysis_spec_routes_aoe2_coaching_prompt(self) -> None:
        video_spec = analysis_spec("aoe2", "aoe2_video")
        coaching_spec = analysis_spec("aoe2", "aoe2_coaching")
        wiki_spec = analysis_spec("aoe2", "aoe2_wiki")
        pdf_spec = analysis_spec("aoe2", "aoe2_pdf")

        self.assertIn("guide-style content", video_spec.system_prompt)
        self.assertIn("coaching sessions, replay reviews", coaching_spec.system_prompt)
        self.assertIn("written guides and reference pages", wiki_spec.system_prompt)
        self.assertIn("written guides and reference pages", pdf_spec.system_prompt)
        self.assertEqual(video_spec.insight_types, coaching_spec.insight_types)
        self.assertEqual(video_spec.insight_types, wiki_spec.insight_types)
        self.assertEqual(video_spec.insight_types, pdf_spec.insight_types)

    def test_pdf_import_marks_extracted_text_transcribed_and_idempotent(self) -> None:
        pdf_path = pathlib.Path(self._tmpdir.name) / "hera.pdf"
        pdf_path.write_bytes(b"%PDF test")

        with mock.patch("scrape.aoe2_pdf_import.extract_pdf_text", return_value="Malay fast uptime opening guide."):
            first_id = import_pdf(pdf_path, title="Hera Strategy Guide 2025")
            second_id = import_pdf(pdf_path, title="Hera Strategy Guide 2025")

        self.assertEqual(first_id, second_id)
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT video_id, source, game, role, status, transcription, video_title FROM videos"
            ).fetchall()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source"], "aoe2_pdf")
        self.assertEqual(row["game"], "aoe2")
        self.assertEqual(row["role"], "general")
        self.assertEqual(row["status"], "transcribed")
        self.assertEqual(row["video_title"], "Hera Strategy Guide 2025")
        self.assertIn("Malay fast uptime", row["transcription"])

    def test_txt_video_import_supports_overrides_and_mixed_playlist_classification(self) -> None:
        input_path = pathlib.Path(self._tmpdir.name) / "aoe2_urls.txt"
        input_path.write_text(
            "# comment\n"
            "video|https://www.youtube.com/watch?v=ccccccccccc\n"
            "\n"
            "https://www.youtube.com/playlist?list=PL123\n",
            encoding="utf-8",
        )
        urls = _read_url_lines(input_path)
        self.assertEqual(urls, [
            {
                "url": "https://www.youtube.com/watch?v=ccccccccccc",
                "forced_source": "aoe2_video",
            },
            {
                "url": "https://www.youtube.com/playlist?list=PL123",
                "forced_source": None,
            },
        ])

        fake_rows = {
            "https://www.youtube.com/watch?v=ccccccccccc": [
                {
                    "video_id": "ccccccccccc",
                    "video_url": "https://www.youtube.com/watch?v=ccccccccccc",
                    "video_title": "Single video",
                    "description": None,
                    "playlist_title": None,
                    "channel": "Hera",
                    "uploader": "Hera",
                }
            ],
            "https://www.youtube.com/playlist?list=PL123": [
                {
                    "video_id": "ddddddddddd",
                    "video_url": "https://www.youtube.com/watch?v=ddddddddddd",
                    "video_title": "Hera coaching a 900 elo student",
                    "description": "Imported from playlist: Test",
                    "playlist_title": "Coaching set",
                    "channel": "Hera",
                    "uploader": "Hera",
                },
                {
                    "video_id": "eeeeeeeeeee",
                    "video_url": "https://www.youtube.com/watch?v=eeeeeeeeeee",
                    "video_title": "Franks beginner guide and build order",
                    "description": "Imported from playlist: Test",
                    "playlist_title": "AoE2 fundamentals",
                    "channel": "Survivalist",
                    "uploader": "Survivalist",
                },
                {
                    "video_id": "fffffffffff",
                    "video_url": "https://www.youtube.com/watch?v=fffffffffff",
                    "video_title": "Arabia game 1",
                    "description": "Imported from playlist: Mixed set",
                    "playlist_title": "Mixed set",
                    "channel": "Unknown",
                    "uploader": "Unknown",
                },
            ],
        }

        with mock.patch("scrape.aoe2_video_import._expand_url", side_effect=lambda url: fake_rows[url]):
            summary = import_urls(urls)

        self.assertEqual(summary["imported"], 4)
        self.assertEqual(summary["expanded"], 4)
        self.assertEqual(summary["coaching"], 1)
        self.assertEqual(summary["video"], 2)
        self.assertEqual(summary["uncertain"], 1)
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT video_id, source, game, role, status FROM videos ORDER BY video_id"
            ).fetchall()
        self.assertEqual(
            [row["video_id"] for row in rows],
            ["ccccccccccc", "ddddddddddd", "eeeeeeeeeee", "fffffffffff"],
        )
        sources = {row["video_id"]: row["source"] for row in rows}
        self.assertEqual(sources["ccccccccccc"], "aoe2_video")
        self.assertEqual(sources["ddddddddddd"], "aoe2_coaching")
        self.assertEqual(sources["eeeeeeeeeee"], "aoe2_video")
        self.assertEqual(sources["fffffffffff"], "aoe2_video")
        self.assertTrue(all(row["game"] == "aoe2" for row in rows))
        self.assertTrue(all(row["role"] == "general" for row in rows))
        self.assertTrue(all(row["status"] == "pending" for row in rows))

    def test_classify_row_marks_uncertain_when_metadata_lacks_strong_cues(self) -> None:
        classified = _classify_row({
            "video_id": "ggggggggggg",
            "video_url": "https://www.youtube.com/watch?v=ggggggggggg",
            "video_title": "Arabia ladder game 2",
            "description": None,
            "playlist_title": "Ranked games",
            "channel": "Player",
            "uploader": "Player",
        })
        self.assertEqual(classified["source"], "aoe2_video")
        self.assertTrue(classified["uncertain"])
        self.assertEqual(classified["classification"], "uncertain")

    def test_guide_chunk_transcript_uses_sentence_aware_boundaries(self) -> None:
        chunk_a = ("Scout early and keep town center production constant. " * 180)
        chunk_b = ("Add farms before floating wood and rebalance eco cleanly. " * 180)
        transcript = chunk_a + chunk_b

        with mock.patch.object(guide_analyze, "BACKEND", "ollama"):
            chunks = guide_analyze.chunk_transcript(transcript)

        self.assertGreater(len(transcript), guide_analyze.CHUNK_CHARS)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith("."))
        self.assertIn("Scout early", chunks[0])
        self.assertTrue(any("Add farms" in chunk for chunk in chunks[1:]))

    def test_guide_chunk_transcript_can_force_split_for_large_pdf_sources(self) -> None:
        transcript = "Malay fast uptime opening. " * 900

        with mock.patch.object(guide_analyze, "BACKEND", "gemini"):
            normal_chunks = guide_analyze.chunk_transcript(transcript)
            pdf_chunks = guide_analyze.chunk_transcript(transcript, force_split=True)

        self.assertEqual(len(normal_chunks), 1)
        self.assertGreater(len(pdf_chunks), 1)

    def test_guide_retranscribe_reset_only_touches_transcribable_sources(self) -> None:
        insert_video(
            video_id="aoe2_video_1",
            video_url="https://www.youtube.com/watch?v=hhhhhhhhhhh",
            video_title="AoE2 video",
            description="",
            game="aoe2",
            role="general",
            subject=None,
            champion=None,
            message_timestamp="2026-04-21T00:00:00+00:00",
            source="aoe2_video",
        )
        insert_video(
            video_id="aoe2_wiki_1",
            video_url="aoe2://aoe2_wiki/aoe2_wiki_1",
            video_title="AoE2 wiki",
            description="",
            game="aoe2",
            role="general",
            subject="Franks",
            champion=None,
            message_timestamp="2026-04-21T00:00:00+00:00",
            source="aoe2_wiki",
        )

        with db.get_connection() as conn:
            conn.execute(
                "UPDATE videos SET status = 'analyzed', transcription = 'old transcript' WHERE video_id = 'aoe2_video_1'"
            )
            conn.execute(
                "UPDATE videos SET status = 'transcribed', transcription = 'wiki text' WHERE video_id = 'aoe2_wiki_1'"
            )
            conn.commit()

        insert_insight("aoe2_video_1", "principles", "Keep villagers working.")

        reset_count, deleted_insights = guide_transcribe._reset_for_retranscribe("aoe2_video")
        self.assertEqual(reset_count, 1)
        self.assertEqual(deleted_insights, 1)

        with db.get_connection() as conn:
            video_row = conn.execute(
                "SELECT status, transcription FROM videos WHERE video_id = 'aoe2_video_1'"
            ).fetchone()
            wiki_row = conn.execute(
                "SELECT status, transcription FROM videos WHERE video_id = 'aoe2_wiki_1'"
            ).fetchone()
            remaining_insights = conn.execute(
                "SELECT COUNT(*) FROM insights WHERE video_id = 'aoe2_video_1'"
            ).fetchone()[0]

        self.assertEqual(video_row["status"], "pending")
        self.assertIsNone(video_row["transcription"])
        self.assertEqual(wiki_row["status"], "transcribed")
        self.assertEqual(wiki_row["transcription"], "wiki text")
        self.assertEqual(remaining_insights, 0)

    def test_guide_selected_sources_for_aoe2_game_excludes_league(self) -> None:
        self.assertEqual(
            guide_transcribe._selected_sources(game="aoe2"),
            ["aoe2_video", "aoe2_coaching"],
        )
        self.assertEqual(
            guide_transcribe._selected_sources(game="lol"),
            ["youtube_guide"],
        )
        self.assertEqual(
            guide_transcribe._selected_sources(source="aoe2_video", game="aoe2"),
            ["aoe2_video"],
        )

    def test_portal_discovery_collects_core_pages_and_supplemental_ages(self) -> None:
        portal_html = """
        <html><body>
          <div class="mw-parser-output">
            <p><a href="/wiki/Age_of_Empires_II:_The_Age_of_Kings">Age of Empires II: The Age of Kings</a></p>
            <p><a href="/wiki/Franks">Franks</a></p>
            <p><a href="/wiki/Civilizations_(Age_of_Empires_II)">Civilizations</a></p>
            <p><a href="/wiki/Tech_tree_(Age_of_Empires_II)">Tech tree</a></p>
            <p><a href="/wiki/Wiki_Help_desk">Wiki Help desk</a></p>
            <p><a href="/wiki/Sandbox">Sandbox</a></p>
            <p><a href="https://community.fandom.com/wiki/Help">External Help</a></p>
          </div>
        </body></html>
        """
        aok_html = """
        <html><body>
          <div class="mw-parser-output">
            <p><a href="/wiki/Dark_Age_(Age_of_Empires_II)">Dark Age</a></p>
            <p><a href="/wiki/Hotkey">Hotkeys</a></p>
          </div>
        </body></html>
        """

        with mock.patch(
            "scrape.aoe2_wiki_scrape._fetch_html",
            side_effect=lambda url: aok_html if "Age_of_Empires_II:_The_Age_of_Kings" in url else portal_html,
        ):
            pages = discover_portal_pages(portal_html)

        titles = {page["title"] for page in pages}
        self.assertIn("Age of Empires II:Portal", titles)
        self.assertIn("Age of Empires II: The Age of Kings", titles)
        self.assertIn("Franks", titles)
        self.assertIn("Civilizations", titles)
        self.assertIn("Tech tree", titles)
        self.assertIn("Wiki Help desk", titles)
        self.assertIn("Dark Age", titles)
        self.assertIn("Hotkeys", titles)
        self.assertNotIn("Sandbox", titles)
        franks = next(page for page in pages if page["title"] == "Franks")
        self.assertEqual(franks["subject"], "Franks")

    def test_wiki_page_cleaner_keeps_infobox_and_article_text(self) -> None:
        page_html = """
        <html><body>
          <h1 id="firstHeading">Franks</h1>
          <aside class="portable-infobox">
            <section class="pi-data">
              <h3 class="pi-data-label">Team bonus</h3>
              <div class="pi-data-value">Knights +2 line of sight</div>
            </section>
          </aside>
          <div class="mw-parser-output">
            <h2>Overview</h2>
            <p>The Franks are a cavalry civilization.</p>
            <ul><li>Scout Cavalry can open cleanly into Knights.</li></ul>
            <p>Sign in to save</p>
          </div>
        </body></html>
        """
        title, text = extract_page_text(page_html)
        self.assertEqual(title, "Franks")
        self.assertIn("## Reference", text)
        self.assertIn("Team bonus: Knights +2 line of sight", text)
        self.assertIn("## Overview", text)
        self.assertIn("The Franks are a cavalry civilization.", text)
        self.assertIn("- Scout Cavalry can open cleanly into Knights.", text)
        self.assertNotIn("Sign in to save", text)

    def test_aoe2_query_profile_detects_micro_and_defense_intent(self) -> None:
        profile = _aoe2_query_profile("how do I defend with better micro and control groups in feudal?")
        preferred_types = profile["preferred_types"]
        self.assertIn("feudal_age", preferred_types)
        self.assertIn("micro", preferred_types)
        self.assertIn("controls_settings", preferred_types)
        self.assertIn("scouting", preferred_types)
        self.assertIn("economy_macro", preferred_types)
        self.assertIn("defense", profile["situation_tags"])
        self.assertTrue(profile["guidance"])

    def test_aoe2_query_profile_detects_system_rule_questions(self) -> None:
        profile = _aoe2_query_profile("how do armor classes and bonus damage interactions work?")
        preferred_types = profile["preferred_types"]
        self.assertIn("game_mechanics", preferred_types)
        self.assertIn("unit_compositions", preferred_types)
        self.assertNotIn("controls_settings", preferred_types)
        self.assertEqual(profile["situation_tags"], [])

    def test_aoe2_query_profile_expands_civ_overview_and_detail_questions(self) -> None:
        profile = _aoe2_query_profile("how should I play the Malay in detail?")
        preferred_types = profile["preferred_types"]
        self.assertTrue(profile["civ_overview"])
        self.assertTrue(profile["detail"])
        self.assertIn("civilization_identity", preferred_types)
        self.assertIn("build_orders", preferred_types)
        self.assertIn("dark_age", preferred_types)
        self.assertIn("imperial_age", preferred_types)
        self.assertIn("Opening / First Minutes", profile["guidance"])

    def test_aoe2_normalization_expands_how_to_play_civ_questions(self) -> None:
        raw = (
            '{"normalized":"What is the core identity and win condition of Malay?",'
            '"subject":"Malay","role":null,'
            '"insight_types":["civilization_identity","principles"],'
            '"reasoning":"Broad civ question."}'
        )
        with mock.patch("retrieval.questions.llm_chat", return_value=raw):
            result = normalize("how should I play the Malay", game="aoe2")

        self.assertEqual(
            result["normalized"],
            "How should I play Malay from opening through win condition?",
        )
        self.assertEqual(result["subject"], "Malay")
        self.assertIn("build_orders", result["insight_types"])

    def test_aoe2_normalization_uses_mistyped_subject_match(self) -> None:
        raw = (
            '{"normalized":"What is the core identity and win condition of Malay?",'
            '"subject":"Malians","role":null,'
            '"insight_types":["civilization_identity","principles"],'
            '"reasoning":"Broad civ question."}'
        )
        with mock.patch("retrieval.questions.llm_chat", return_value=raw):
            result = normalize("how should I play malya", game="aoe2")

        self.assertEqual(result["subject"], "Malay")
        self.assertIn("local civ-name correction", result["reasoning"])
        self.assertEqual(
            result["normalized"],
            "How should I play Malay from opening through win condition?",
        )

    def test_aoe2_normalization_preserves_detail_mode_for_civ_overviews(self) -> None:
        raw = (
            '{"normalized":"What is the core identity and win condition of Malay?",'
            '"subject":"Malay","role":null,'
            '"insight_types":["civilization_identity","principles"],'
            '"reasoning":"Broad detailed civ question."}'
        )
        with mock.patch("retrieval.questions.llm_chat", return_value=raw):
            result = normalize("how should I play the Malay in detail", game="aoe2")

        self.assertEqual(
            result["normalized"],
            "How should I play Malay from opening through win condition in detail?",
        )

    def test_aoe2_civ_overview_retrieves_five_insights_per_standard_section(self) -> None:
        calls = []

        def fake_retrieve(*args, **kwargs):
            calls.append(kwargs)
            insight_type = kwargs["preferred_types"][0]
            scope = "subject" if kwargs.get("subject") else "general"
            return [
                {
                    "text": f"{scope} {insight_type} section insight {i}",
                    "insight_type": insight_type,
                    "score": 0.9,
                    "confidence": 0.8,
                    "source_weight": 2.0,
                    "source": "aoe2_pdf",
                    "retrieval_layer": "direct",
                }
                for i in range(10)
            ]

        with mock.patch.object(retrieval_query, "retrieve", side_effect=fake_retrieve):
            sections = retrieval_query._retrieve_aoe2_civ_overview_sections(
                "How should I play Malay from opening through win condition in detail?",
                "Malay",
            )

        self.assertEqual(
            [section["name"] for section in sections],
            [
                "Core Identity / Gameplan",
                "Opening / First Minutes",
                "Dark Age",
                "Feudal Age",
                "Castle Age",
                "Imperial / Win Condition",
                "Common Mistakes",
            ],
        )
        self.assertEqual(len(calls), len(retrieval_query.AOE2_CIV_OVERVIEW_SECTIONS) * 2)
        self.assertTrue(all(call["top_k"] == 10 for call in calls))
        self.assertTrue(all(calls[i]["subject"] == "Malay" for i in range(0, len(calls), 2)))
        self.assertTrue(all(calls[i]["subject"] is None for i in range(1, len(calls), 2)))
        self.assertIn("build_orders", calls[2]["preferred_types"])
        self.assertIn("feudal_age", calls[6]["preferred_types"])
        self.assertIn("imperial_age", calls[10]["preferred_types"])
        self.assertTrue(all(len(section["insights"]) == 5 for section in sections))

    def test_aoe2_civ_overview_answer_uses_grouped_section_prompt_and_sources(self) -> None:
        duplicate = {
            "text": "Use a clean opening with constant villager production.",
            "insight_type": "build_orders",
            "score": 0.8,
            "confidence": 0.8,
            "source_weight": 2.0,
            "source": "aoe2_pdf",
            "retrieval_layer": "direct",
        }
        sections = [
            {
                "name": "Core Identity / Gameplan",
                "insights": [
                    {
                        "text": "Malay advance quickly and should use timing windows.",
                        "insight_type": "civilization_identity",
                        "score": 0.9,
                        "confidence": 0.8,
                        "source_weight": 1.0,
                        "source": "aoe2_video",
                        "retrieval_layer": "direct",
                    },
                    duplicate,
                ],
            },
            {
                "name": "Opening / First Minutes",
                "insights": [duplicate],
            },
        ]

        with mock.patch.object(
            retrieval_query,
            "_retrieve_aoe2_civ_overview_sections",
            return_value=sections,
        ), mock.patch.object(
            retrieval_query,
            "llm_chat",
            return_value="Detailed Malay answer",
        ) as mocked_llm:
            answer = retrieval_query.answer(
                "How should I play Malay from opening through win condition in detail?",
                game="aoe2",
                subject="Malay",
                show_sources=True,
            )

        self.assertIn("Detailed Malay answer", answer)
        self.assertEqual(answer.count("Use a clean opening"), 2)
        self.assertIn("Sources by section:", answer)
        self.assertIn("  Core Identity / Gameplan:", answer)
        self.assertIn("  Opening / First Minutes:", answer)
        self.assertIn("### Opening / First Minutes", mocked_llm.call_args.kwargs["system"])
        self.assertIn("2-3 evidence-backed sentences", mocked_llm.call_args.kwargs["system"])
        self.assertIn("## Core Identity / Gameplan", mocked_llm.call_args.kwargs["user"])
        self.assertIn("## Opening / First Minutes", mocked_llm.call_args.kwargs["user"])
        self.assertIn("Detail mode: yes", mocked_llm.call_args.kwargs["user"])
        self.assertIn("2-3 sentences", mocked_llm.call_args.kwargs["user"])

    def test_detect_aoe2_intent_marks_civ_matchups(self) -> None:
        self.assertEqual(
            detect_aoe2_intent("Franks vs Hindustanis"),
            {"type": "matchup", "a": "Franks", "b": "Hindustanis"},
        )

    def test_aoe2_civ_matching_handles_aliases_and_common_typos(self) -> None:
        self.assertEqual(canonical_aoe2_civilization("byz"), "Byzantines")
        self.assertEqual(canonical_aoe2_civilization("hindustan"), "Hindustanis")
        self.assertEqual(find_aoe2_civilizations("how to play byzintines")[0][1], "Byzantines")
        self.assertEqual(
            find_aoe2_civilizations("malya vs hindustans"),
            [(0, "Malay"), (9, "Hindustanis")],
        )

    def test_detect_aoe2_intent_handles_mistyped_civ_names(self) -> None:
        self.assertEqual(
            detect_aoe2_intent("malya vs hindustans"),
            {"type": "matchup", "a": "Malay", "b": "Hindustanis"},
        )

    def test_answer_routes_aoe2_matchup_questions_to_duo_mode(self) -> None:
        with mock.patch.object(retrieval_query, "_retrieve_aoe2_duo", return_value=(
            [{"text": "Open scouts quickly.", "insight_type": "build_orders", "score": 0.9, "confidence": 0.8, "source_weight": 1.0, "source": "aoe2_video", "retrieval_layer": "direct"}],
            [{"text": "Respect early cavalry pressure.", "insight_type": "matchup_advice", "score": 0.9, "confidence": 0.8, "source_weight": 1.0, "source": "aoe2_video", "retrieval_layer": "direct"}],
            [{"text": "Scouting informs your transitions.", "insight_type": "scouting", "score": 0.7, "confidence": 0.7, "source_weight": 1.0, "source": "aoe2_wiki", "retrieval_layer": "direct"}],
        )) as mocked_retrieve, mock.patch.object(
            retrieval_query,
            "llm_chat",
            return_value="AOE2 matchup answer",
        ) as mocked_llm:
            answer = retrieval_query.answer("Franks vs Hindustanis", game="aoe2", show_sources=False)

        self.assertEqual(answer, "AOE2 matchup answer")
        mocked_retrieve.assert_called_once()
        self.assertIn("Age of Empires II coaching assistant", mocked_llm.call_args.kwargs["system"])

    def test_init_db_migrates_only_aoe2_controls_rows_to_controls_settings(self) -> None:
        insert_video(
            video_id="aoe2_controls",
            video_url="https://www.youtube.com/watch?v=iiiiiiiiiii",
            video_title="AoE2 controls",
            description="",
            game="aoe2",
            role="general",
            subject=None,
            champion=None,
            message_timestamp="2026-04-21T00:00:00+00:00",
            source="aoe2_video",
        )
        insert_video(
            video_id="aoe2_rules",
            video_url="https://www.youtube.com/watch?v=jjjjjjjjjjj",
            video_title="AoE2 rules",
            description="",
            game="aoe2",
            role="general",
            subject=None,
            champion=None,
            message_timestamp="2026-04-21T00:00:00+00:00",
            source="aoe2_video",
        )
        insert_video(
            video_id="lol_controls",
            video_url="https://www.youtube.com/watch?v=kkkkkkkkkkk",
            video_title="LoL controls",
            description="",
            game="lol",
            role="mid",
            subject="Ahri",
            champion="Ahri",
            message_timestamp="2026-04-21T00:00:00+00:00",
            source="youtube_guide",
        )

        aoe2_controls_id = insert_insight(
            "aoe2_controls",
            "game_mechanics",
            "Use control groups and hotkeys so your army and town centers stay accessible.",
        )
        aoe2_rules_id = insert_insight(
            "aoe2_rules",
            "game_mechanics",
            "Armor classes and bonus damage determine how efficiently units trade into each other.",
        )
        lol_controls_id = insert_insight(
            "lol_controls",
            "game_mechanics",
            "Use quick cast and comfortable hotkeys so your combos come out faster.",
        )

        init_db()

        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, insight_type FROM insights WHERE id IN (?, ?, ?)",
                (aoe2_controls_id, aoe2_rules_id, lol_controls_id),
            ).fetchall()

        by_id = {row["id"]: row["insight_type"] for row in rows}
        self.assertEqual(by_id[aoe2_controls_id], "controls_settings")
        self.assertEqual(by_id[aoe2_rules_id], "game_mechanics")
        self.assertEqual(by_id[lol_controls_id], "game_mechanics")


if __name__ == "__main__":
    unittest.main()
