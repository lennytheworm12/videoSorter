import unittest

from core.game_registry import analysis_spec
import pipeline.analyze as coaching_analyze
from prompts.aoe2 import (
    AOE2_GUIDE_COACHING_SYSTEM_PROMPT,
    AOE2_GUIDE_VIDEO_SYSTEM_PROMPT,
    AOE2_GUIDE_WRITTEN_SYSTEM_PROMPT,
)
from prompts.lol import (
    LOL_COACHING_EXTRACTION_PROMPT,
    LOL_COACHING_SYSTEM_PROMPT,
    LOL_GUIDE_VIDEO_SYSTEM_PROMPT,
    LOL_GUIDE_WRITTEN_SYSTEM_PROMPT,
)
from retrieval.questions import AOE2_CANONICAL_QUESTIONS


class PromptModuleTests(unittest.TestCase):
    def test_guide_analysis_spec_uses_prompt_modules(self) -> None:
        self.assertEqual(
            analysis_spec("lol", "youtube_guide").system_prompt,
            LOL_GUIDE_VIDEO_SYSTEM_PROMPT,
        )
        self.assertEqual(
            analysis_spec("lol", "mobafire_guide").system_prompt,
            LOL_GUIDE_WRITTEN_SYSTEM_PROMPT,
        )
        self.assertEqual(
            analysis_spec("aoe2", "aoe2_video").system_prompt,
            AOE2_GUIDE_VIDEO_SYSTEM_PROMPT,
        )
        self.assertEqual(
            analysis_spec("aoe2", "aoe2_coaching").system_prompt,
            AOE2_GUIDE_COACHING_SYSTEM_PROMPT,
        )
        self.assertEqual(
            analysis_spec("aoe2", "aoe2_wiki").system_prompt,
            AOE2_GUIDE_WRITTEN_SYSTEM_PROMPT,
        )

    def test_coaching_pipeline_imports_prompt_module_constants(self) -> None:
        self.assertEqual(coaching_analyze.LOL_COACHING_SYSTEM_PROMPT, LOL_COACHING_SYSTEM_PROMPT)
        self.assertEqual(coaching_analyze.LOL_COACHING_EXTRACTION_PROMPT, LOL_COACHING_EXTRACTION_PROMPT)

    def test_guide_prompt_templates_still_expose_runtime_placeholders(self) -> None:
        aoe2_prompt = analysis_spec("aoe2", "aoe2_video").extraction_prompt
        lol_prompt = analysis_spec("lol", "youtube_guide").extraction_prompt

        self.assertIn("{subject}", aoe2_prompt)
        self.assertIn("{role}", aoe2_prompt)
        self.assertIn("{title}", aoe2_prompt)
        self.assertIn("{chunk_label}", aoe2_prompt)
        self.assertIn("{transcript_chunk}", aoe2_prompt)

        self.assertIn("{subject}", lol_prompt)
        self.assertIn("{role}", lol_prompt)
        self.assertIn("{title}", lol_prompt)
        self.assertIn("{chunk_label}", lol_prompt)
        self.assertIn("{transcript_chunk}", lol_prompt)

    def test_aoe2_prompt_disambiguates_game_mechanics_from_general_rules(self) -> None:
        self.assertIn("AoE2 system knowledge and interaction rules", AOE2_GUIDE_VIDEO_SYSTEM_PROMPT)
        self.assertIn("controls_settings", AOE2_GUIDE_VIDEO_SYSTEM_PROMPT)
        self.assertIn("Use game_mechanics for AoE2 system knowledge", AOE2_GUIDE_COACHING_SYSTEM_PROMPT)
        self.assertIn("Use controls_settings for hotkeys", AOE2_GUIDE_WRITTEN_SYSTEM_PROMPT)

    def test_aoe2_beginner_template_prefers_broad_fundamentals_not_controls(self) -> None:
        beginner_template = next(
            entry for entry in AOE2_CANONICAL_QUESTIONS
            if entry[0] == "What fundamental AoE2 habits should a beginner focus on first?"
        )
        self.assertEqual(beginner_template[1], ["principles", "economy_macro", "scouting"])

    def test_aoe2_micro_template_avoids_mechanics_buckets_by_default(self) -> None:
        micro_template = next(
            entry for entry in AOE2_CANONICAL_QUESTIONS
            if entry[0] == "How should I micro my army and control fights more effectively?"
        )
        self.assertEqual(micro_template[1], ["micro", "unit_compositions", "principles"])


if __name__ == "__main__":
    unittest.main()
