import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from pipeline.phase2g_endpoint_recovery import load_benchmark
from pipeline.phase2g_silver import canonical_sha256
from pipeline.phase2h_endpoint_scoring import (
    KEEP,
    build_dataset,
    extract_dense_features,
)
from pipeline.phase2i_syntax import (
    BOUNDARY_AMBIGUOUS,
    BOUNDARY_EXACT,
    BOUNDARY_PARTIAL,
    BOUNDARY_TOKEN_ALIGNED,
    BOUNDARY_UNALIGNED,
    DENSE_C_EXTRA_FEATURES,
    LOCKED_ASSETS_MANIFEST_SHA256,
    PARSE_SCHEMA_VERSION,
    STANZA_PROCESSORS,
    STANZA_VERSION,
    SYNTAX_FEATURE_SCHEMA_VERSION,
    UdParse,
    UdSentence,
    UdToken,
    UdWord,
    CandidateSyntax,
    Phase2IParseError,
    Phase2ISyntaxError,
    SyntaxEncoder,
    _is_finite,
    assets_manifest_sha256,
    compute_candidate_syntax,
    dense_c_matrix,
    feature_schema_c,
    is_sha256_hex,
    load_parse_artifact,
    parse_window_text,
    syntax_groups_from_records,
    verify_parser_asset_path,
    verify_assets_provenance,
)
from scripts.setup_phase2i_parser_assets import (
    _prepare_asset_path,
    main as setup_main,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"
ASSETS = ROOT / "data" / "phase2i_assets"


def _word(
    sentence_id,
    word_id,
    token_id,
    text,
    lemma,
    upos,
    xpos,
    head,
    deprel,
    start,
    end,
    *,
    feats="",
    offset_kind="TOKEN",
):
    return UdWord(
        word_id=word_id,
        sentence_id=sentence_id,
        token_id=token_id,
        text=text,
        lemma=lemma,
        upos=upos,
        xpos=xpos,
        feats=feats,
        head=head,
        deprel=deprel,
        deps="_",
        start_char=start,
        end_char=end,
        offset_kind=offset_kind,
    )


def _parse_from_sentences(window_id, text, sentences):
    parse = UdParse(
        window_id=window_id,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        sentences=tuple(sentences),
        parser="fixture",
        parser_version="test",
        package="ewt",
        processors=STANZA_PROCESSORS,
        language="en",
        model_assets=(),
        assets_manifest_sha256=assets_manifest_sha256(()),
        pipeline_version="test-fixture",
        parse_sha256="",
    )
    sha256 = canonical_sha256(parse.canonical_serialization())
    return UdParse(
        **{**parse.__dict__, "parse_sha256": sha256},
    )


def _fixture_parse(text, token_specs, word_specs):
    """Build a one-sentence fixture parse from explicit token/word specs."""
    tokens = []
    words = []
    for token_id, token_text, start, end, word_ids in token_specs:
        tokens.append(UdToken(
            token_id=token_id,
            sentence_id=1,
            text=token_text,
            start_char=start,
            end_char=end,
            multiword=len(word_ids) > 1,
            word_ids=tuple(word_ids),
        ))
    for spec in word_specs:
        (
            word_id, token_id, wtext, lemma, upos, xpos, head, deprel,
            start, end,
        ) = spec[:10]
        feats = spec[10] if len(spec) > 10 else ""
        multiword = len([
            spec for spec in token_specs
            if spec[0] == token_id and len(spec[4]) > 1
        ]) > 0
        words.append(_word(
            1, word_id, token_id, wtext, lemma, upos, xpos, head, deprel,
            start, end,
            feats=feats,
            offset_kind="MWT_PROJECTED" if multiword else "TOKEN",
        ))
    sentence = UdSentence(
        sentence_id=1,
        start_char=tokens[0].start_char,
        end_char=tokens[-1].end_char,
        text=text[tokens[0].start_char:tokens[-1].end_char],
        tokens=tuple(tokens),
        words=tuple(words),
    )
    return _parse_from_sentences("fixture", text, (sentence,))


def _row(candidate_id, start, end, text):
    return type(
        "Row",
        (),
        {
            "window_id": "fixture",
            "candidate_id": candidate_id,
            "start": start,
            "end": end,
            "text": text,
            "label": KEEP,
        },
    )()


class Phase2ISyntaxFixtureTests(unittest.TestCase):
    def test_exact_token_alignment_and_multi_token_state(self):
        text = "alpha beta gamma"
        parse = _fixture_parse(
            text,
            [
                (1, "alpha", 0, 5, [1]),
                (2, "beta", 6, 10, [2]),
                (3, "gamma", 11, 16, [3]),
            ],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
                (3, 3, "gamma", "gamma", "NOUN", "NN", 1, "obj", 11, 16),
            ],
        )
        exact = compute_candidate_syntax(parse, _row("c1", 0, 5, "alpha"))
        self.assertEqual(exact.boundary_status, BOUNDARY_EXACT)
        self.assertTrue(exact.start_aligned and exact.end_aligned)
        self.assertFalse(exact.multi_token)
        self.assertEqual(exact.word_ids, ("s1:w1",))
        self.assertEqual(exact.candidate_head_id, "s1:w1")
        self.assertTrue(exact.span_connected)
        aligned = compute_candidate_syntax(
            parse, _row("c2", 0, 10, "alpha beta"),
        )
        self.assertEqual(aligned.boundary_status, BOUNDARY_TOKEN_ALIGNED)
        self.assertTrue(aligned.multi_token)
        self.assertEqual(aligned.token_ids, ("s1:t1", "s1:t2"))
        self.assertEqual(aligned.word_ids, ("s1:w1", "s1:w2"))

    def test_partial_and_unaligned_boundary_statuses(self):
        text = "alpha beta gamma "
        parse = _fixture_parse(
            text,
            [
                (1, "alpha", 0, 5, [1]),
                (2, "beta", 6, 10, [2]),
                (3, "gamma", 11, 16, [3]),
            ],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
                (3, 3, "gamma", "gamma", "NOUN", "NN", 1, "obj", 11, 16),
            ],
        )
        partial = compute_candidate_syntax(
            parse, _row("c1", 1, 7, "lpha b"),
        )
        self.assertEqual(partial.boundary_status, BOUNDARY_PARTIAL)
        self.assertFalse(partial.start_aligned)
        self.assertFalse(partial.end_aligned)
        unaligned = compute_candidate_syntax(
            parse, _row("c2", 16, 17, " "),
        )
        self.assertEqual(unaligned.boundary_status, BOUNDARY_UNALIGNED)
        self.assertEqual(unaligned.ambiguity, ("NO_INTERSECTING_TOKENS",))
        self.assertEqual(unaligned.candidate_head_id, None)

    def test_mwt_projection_and_ambiguous_boundary_cut(self):
        # "don't" is an MWT token (2,3) with words 2 (do) and 3 (n't).
        text = "I don't go"
        parse = _fixture_parse(
            text,
            [
                (1, "I", 0, 1, [1]),
                (2, "don't", 2, 7, [2, 3]),
                (4, "go", 8, 10, [4]),
            ],
            [
                (1, 1, "I", "I", "PRON", "PRP", 4, "nsubj", 0, 1),
                (2, 2, "do", "do", "AUX", "VBP", 4, "aux", 2, 7),
                (3, 2, "n't", "not", "PART", "RB", 4, "neg", 2, 7),
                (4, 4, "go", "go", "VERB", "VB", 0, "root", 8, 10),
            ],
        )
        exact = compute_candidate_syntax(
            parse, _row("c1", 2, 7, "don't"),
        )
        self.assertEqual(exact.boundary_status, BOUNDARY_EXACT)
        self.assertEqual(exact.word_ids, ("s1:w2", "s1:w3"))
        word_2 = next(
            w for w in parse.sentences[0].words if w.word_id == 2
        )
        word_3 = next(
            w for w in parse.sentences[0].words if w.word_id == 3
        )
        self.assertTrue(word_2.is_mwt_projected)
        self.assertTrue(word_3.is_mwt_projected)
        self.assertEqual((word_2.start_char, word_2.end_char), (2, 7))
        self.assertEqual((word_3.start_char, word_3.end_char), (2, 7))
        self.assertIn("neg", exact.relations)
        ambiguous = compute_candidate_syntax(
            parse, _row("c2", 3, 7, "on't"),
        )
        self.assertEqual(ambiguous.boundary_status, BOUNDARY_AMBIGUOUS)
        self.assertIn("MULTIWORD_BOUNDARY_CUT", ambiguous.ambiguity)

    def test_head_depth_children_subtree_and_crossing_arcs(self):
        # reset(root) <- you(nsubj), after(mark)->killing(advcl)->someone(obj)
        text = "you reset after killing someone"
        parse = _fixture_parse(
            text,
            [
                (1, "you", 0, 3, [1]),
                (2, "reset", 4, 9, [2]),
                (3, "after", 10, 15, [3]),
                (4, "killing", 16, 23, [4]),
                (5, "someone", 24, 31, [5]),
            ],
            [
                (1, 1, "you", "you", "PRON", "PRP", 2, "nsubj", 0, 3),
                (2, 2, "reset", "reset", "VERB", "VBP", 0, "root", 4, 9),
                (3, 3, "after", "after", "SCONJ", "IN", 4, "mark", 10, 15),
                (4, 4, "killing", "kill", "VERB", "VBG", 2, "advcl", 16, 23),
                (5, 5, "someone", "someone", "PRON", "NN", 4, "obj", 24, 31),
            ],
        )
        full = compute_candidate_syntax(
            parse, _row("c1", 4, 31, "reset after killing someone"),
        )
        self.assertEqual(full.candidate_head_id, "s1:w2")
        self.assertEqual(full.candidate_root_id, "s1:w2")
        self.assertEqual(full.candidate_depth_values, (0, 2, 1, 2))
        self.assertEqual(full.subtree_size, 5)
        self.assertEqual(full.subtree_intersection_count, 4)
        self.assertFalse(full.subtree_exact)
        self.assertTrue(full.span_connected)
        self.assertIn("mark", full.relations)
        self.assertEqual(full.crossing_arc_count, 1)
        # W = {2,3,4,5}: no candidate word is governed from outside; the one
        # crossing arc is the outside dependent "you" headed inside (reset).
        self.assertEqual(full.external_governor_ids, ())
        partial = compute_candidate_syntax(
            parse, _row("c2", 24, 31, "someone"),
        )
        self.assertEqual(partial.candidate_head_id, "s1:w5")
        self.assertEqual(partial.subtree_size, 1)
        self.assertTrue(partial.subtree_exact)
        self.assertEqual(partial.crossing_arc_count, 1)
        killing_span = compute_candidate_syntax(
            parse, _row("c3", 16, 31, "killing someone"),
        )
        self.assertEqual(killing_span.candidate_head_id, "s1:w4")
        self.assertEqual(killing_span.boundary_head_ids, ("s1:w4",))
        self.assertIn("s1:w2", killing_span.external_governor_ids)
        self.assertIn("advcl", killing_span.relations)
        self.assertIn("obj", killing_span.relations)

    def test_finite_aux_modal_neg_subject_object_pronoun_features(self):
        # "you should not have reset the wave"
        text = "you should not have reset the wave"
        parse = _fixture_parse(
            text,
            [
                (1, "you", 0, 3, [1]),
                (2, "should", 4, 10, [2]),
                (3, "not", 11, 14, [3]),
                (4, "have", 15, 19, [4]),
                (5, "reset", 20, 25, [5]),
                (6, "the", 26, 29, [6]),
                (7, "wave", 30, 34, [7]),
            ],
            [
                (1, 1, "you", "you", "PRON", "PRP", 5, "nsubj", 0, 3),
                (2, 2, "should", "should", "AUX", "MD", 5, "aux", 4, 10),
                (3, 3, "not", "not", "PART", "RB", 5, "neg", 11, 14),
                (4, 4, "have", "have", "AUX", "VBP", 5, "aux", 15, 19),
                (5, 5, "reset", "reset", "VERB", "VBP", 0, "root", 20, 25),
                (6, 6, "the", "the", "DET", "DT", 7, "det", 26, 29),
                (7, 7, "wave", "wave", "NOUN", "NN", 5, "obj", 30, 34),
            ],
        )
        record = compute_candidate_syntax(
            parse, _row("c1", 0, 34, text),
        )
        self.assertIn("s1:w5", record.finite_verb_ids)
        self.assertIn("s1:w4", record.finite_verb_ids)  # "have" is finite
        self.assertIn("s1:w2", record.finite_verb_ids)  # modal is finite
        self.assertEqual(len(record.finite_verb_ids), 3)
        # Clause roots are structural, not bare finiteness: only the root
        # "reset" is a clause root; ordinary finite auxiliaries never are.
        self.assertEqual(record.clause_root_ids, ("s1:w5",))
        self.assertFalse(len(record.clause_root_ids) > 1)
        self.assertIn("s1:w2", record.aux_ids)
        self.assertIn("s1:w4", record.aux_ids)
        self.assertIn("s1:w2", record.modal_ids)
        self.assertIn("s1:w3", record.neg_ids)
        self.assertTrue(record.predicate_internal_argument)
        self.assertFalse(record.predicate_external_argument)
        self.assertTrue(record.aux_internal)
        self.assertTrue(record.neg_internal)
        self.assertIn("nsubj", record.relations)
        self.assertIn("obj", record.relations)
        self.assertIn("aux", record.relations)
        self.assertIn("neg", record.relations)
        self.assertIn("s1:w1", record.pronoun_ids)
        self.assertTrue(record.pronoun_inside_governor)
        self.assertIn("s1:w2", record.scope_internal_ids)
        self.assertIn("s1:w3", record.scope_internal_ids)
        self.assertEqual(
            record.group_values("pronoun_governor_upos"),
            ("VERB",),
        )
        self.assertEqual(
            record.group_values("modal_lemma"),
            ("should",),
        )
        self.assertIn("nsubj:VERB:PRON", record.group_values("rel_context"))
        self.assertIn("obj:VERB:NOUN", record.group_values("rel_context"))

    def test_external_governor_and_multiple_candidate_heads_flagged(self):
        # Single boundary head: "beta gamma" where beta.head=gamma (inside)
        # and gamma.head=alpha (outside).
        text = "alpha beta gamma"
        parse = _fixture_parse(
            text,
            [
                (1, "alpha", 0, 5, [1]),
                (2, "beta", 6, 10, [2]),
                (3, "gamma", 11, 16, [3]),
            ],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 3, "obj", 6, 10),
                (3, 3, "gamma", "gamma", "NOUN", "NN", 1, "conj", 11, 16),
            ],
        )
        record = compute_candidate_syntax(
            parse, _row("c1", 6, 16, "beta gamma"),
        )
        self.assertEqual(record.candidate_head_id, "s1:w3")
        self.assertEqual(record.boundary_head_ids, ("s1:w3",))
        self.assertEqual(record.external_governor_ids, ("s1:w1",))
        self.assertEqual(record.crossing_arc_count, 1)
        self.assertIn("conj", record.relations)

        # Two boundary heads: both words point outside -> explicit ambiguity.
        text2 = "alpha beta gamma"
        parse2 = _fixture_parse(
            text2,
            [
                (1, "alpha", 0, 5, [1]),
                (2, "beta", 6, 10, [2]),
                (3, "gamma", 11, 16, [3]),
            ],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
                (3, 3, "gamma", "gamma", "NOUN", "NN", 1, "conj", 11, 16),
            ],
        )
        record2 = compute_candidate_syntax(
            parse2, _row("c2", 6, 16, "beta gamma"),
        )
        self.assertIn("MULTIPLE_CANDIDATE_HEADS", record2.ambiguity)
        self.assertEqual(len(record2.boundary_head_ids), 2)
        self.assertEqual(len(record2.external_governor_ids), 1)
        self.assertEqual(record2.crossing_arc_count, 2)


def _word_by_key(parse):
    return {
        f"s{sentence.sentence_id}:w{word.word_id}": word
        for sentence in parse.sentences
        for word in sentence.words
    }


class Phase2IFinitenessTests(unittest.TestCase):
    def test_verbform_feats_are_decisive_for_finiteness(self):
        cases = [
            ("Tense=Past|VerbForm=Part", "VERB", "VBN", False),
            ("Tense=Pres|VerbForm=Part", "VERB", "VBG", False),
            ("VerbForm=Inf", "VERB", "VB", False),
            ("VerbForm=Ger", "VERB", "VBG", False),
            ("VerbForm=Vnoun", "VERB", "NN", False),
            ("VerbForm=Conv", "VERB", "VB", False),
            ("Mood=Ind|Tense=Pres|VerbForm=Fin", "VERB", "VBP", True),
            ("Mood=Imp|VerbForm=Fin", "VERB", "VB", True),
            ("VerbForm=Fin", "AUX", "VBZ", True),
        ]
        for feats, upos, xpos, expected in cases:
            word = _word(
                1, 1, 1, "x", "x", upos, xpos, 0, "root", 0, 1,
                feats=feats,
            )
            self.assertEqual(
                _is_finite(word), expected, (feats, upos, xpos),
            )

    def test_xpos_fallback_only_without_verbform(self):
        cases = [
            ("", "VERB", "VBD", True),
            ("", "VERB", "VBP", True),
            ("", "VERB", "VBZ", True),
            ("", "AUX", "MD", True),
            ("", "VERB", "VB", False),
            ("", "VERB", "VBG", False),
            ("", "VERB", "VBN", False),
            ("", "NOUN", "VBZ", False),
            ("_", "AUX", "VBZ", True),
        ]
        for feats, upos, xpos, expected in cases:
            word = _word(
                1, 1, 1, "x", "x", upos, xpos, 0, "root", 0, 1,
                feats=feats,
            )
            self.assertEqual(
                _is_finite(word), expected, (feats, upos, xpos),
            )

    def test_aux_cop_are_never_clause_roots_but_root_is_structural(self):
        # "you are winning" -- "are" is a finite aux dependent and must not
        # be a clause root; "winning" (Part) is the root.
        text = "you are winning"
        parse = _fixture_parse(
            text,
            [
                (1, "you", 0, 3, [1]),
                (2, "are", 4, 7, [2]),
                (3, "winning", 8, 15, [3]),
            ],
            [
                (1, 1, "you", "you", "PRON", "PRP", 3, "nsubj", 0, 3),
                (
                    2, 2, "are", "be", "AUX", "VBP", 3, "aux", 4, 7,
                    "Tense=Pres|VerbForm=Fin",
                ),
                (
                    3, 3, "winning", "win", "VERB", "VBG", 0, "root",
                    8, 15, "Tense=Pres|VerbForm=Part",
                ),
            ],
        )
        record = compute_candidate_syntax(
            parse, _row("c1", 0, 15, text),
        )
        self.assertEqual(record.finite_verb_ids, ("s1:w2",))
        self.assertEqual(record.clause_root_ids, ("s1:w3",))
        self.assertFalse(len(record.clause_root_ids) > 1)

        # A root auxiliary remains a structural clause root (head == 0 is
        # checked before the aux/cop exclusion).
        fragment = "will you?"
        parse2 = _fixture_parse(
            fragment,
            [
                (1, "will", 0, 4, [1]),
                (2, "you", 5, 8, [2]),
                (3, "?", 8, 9, [3]),
            ],
            [
                (
                    1, 1, "will", "will", "AUX", "MD", 0, "root", 0, 4,
                    "VerbForm=Fin",
                ),
                (2, 2, "you", "you", "PRON", "PRP", 1, "nsubj", 5, 8),
                (3, 3, "?", "?", "PUNCT", ".", 1, "punct", 8, 9),
            ],
        )
        record2 = compute_candidate_syntax(
            parse2, _row("c2", 0, 4, "will"),
        )
        self.assertEqual(record2.clause_root_ids, ("s1:w1",))

    def test_finite_verb_conj_is_justified_only_by_clause_root_governor(self):
        # "he ran and jumped" -- "jumped" is a finite VERB conj of the root
        # "ran": both are clause roots.
        text = "he ran and jumped"
        parse = _fixture_parse(
            text,
            [
                (1, "he", 0, 2, [1]),
                (2, "ran", 3, 6, [2]),
                (3, "and", 7, 10, [3]),
                (4, "jumped", 11, 17, [4]),
            ],
            [
                (1, 1, "he", "he", "PRON", "PRP", 2, "nsubj", 0, 2),
                (
                    2, 2, "ran", "run", "VERB", "VBD", 0, "root", 3, 6,
                    "VerbForm=Fin",
                ),
                (3, 3, "and", "and", "CCONJ", "CC", 4, "cc", 7, 10),
                (
                    4, 4, "jumped", "jump", "VERB", "VBD", 2, "conj",
                    11, 17, "VerbForm=Fin",
                ),
            ],
        )
        record = compute_candidate_syntax(
            parse, _row("c1", 3, 17, "ran and jumped"),
        )
        self.assertEqual(
            record.clause_root_ids, ("s1:w2", "s1:w4"),
        )
        self.assertTrue(len(record.clause_root_ids) > 1)

        # "you can run and jump" -- the conjunct "jump" heads to the bare
        # infinitive "run" (not a clause root), so it is not justified.
        text2 = "you can run and jump"
        parse2 = _fixture_parse(
            text2,
            [
                (1, "you", 0, 3, [1]),
                (2, "can", 4, 7, [2]),
                (3, "run", 8, 11, [3]),
                (4, "and", 12, 15, [4]),
                (5, "jump", 16, 20, [5]),
            ],
            [
                (1, 1, "you", "you", "PRON", "PRP", 3, "nsubj", 0, 3),
                (
                    2, 2, "can", "can", "AUX", "MD", 3, "aux", 4, 7,
                    "VerbForm=Fin",
                ),
                (
                    3, 3, "run", "run", "VERB", "VB", 0, "root", 8, 11,
                    "VerbForm=Inf",
                ),
                (4, 4, "and", "and", "CCONJ", "CC", 5, "cc", 12, 15),
                (
                    5, 5, "jump", "jump", "VERB", "VB", 3, "conj", 16, 20,
                    "VerbForm=Inf",
                ),
            ],
        )
        record2 = compute_candidate_syntax(
            parse2, _row("c2", 8, 20, "run and jump"),
        )
        self.assertEqual(record2.clause_root_ids, ("s1:w3",))


class Phase2IObliqueRoleTests(unittest.TestCase):
    def test_oblique_is_separate_and_does_not_activate_object(self):
        text = "you reset by hand"
        parse = _fixture_parse(
            text,
            [
                (1, "you", 0, 3, [1]),
                (2, "reset", 4, 9, [2]),
                (3, "by", 10, 12, [3]),
                (4, "hand", 13, 17, [4]),
            ],
            [
                (1, 1, "you", "you", "PRON", "PRP", 2, "nsubj", 0, 3),
                (
                    2, 2, "reset", "reset", "VERB", "VBP", 0, "root",
                    4, 9, "VerbForm=Fin",
                ),
                (3, 3, "by", "by", "ADP", "IN", 4, "case", 10, 12),
                (4, 4, "hand", "hand", "NOUN", "NN", 2, "obl", 13, 17),
            ],
        )
        record = compute_candidate_syntax(
            parse, _row("c1", 4, 17, "reset by hand"),
        )
        # Oblique-only evidence never activates the object role.
        self.assertFalse(record.predicate_internal_object)
        self.assertFalse(record.predicate_external_object)
        self.assertTrue(record.predicate_internal_oblique)
        self.assertFalse(record.predicate_external_oblique)
        self.assertIn("oblique:internal", record.predicate_argument_context)
        self.assertNotIn("object:internal", record.predicate_argument_context)
        self.assertIn(
            "oblique:internal",
            record.group_values("predicate_argument_context"),
        )
        matrix = dense_c_matrix([record])
        columns = {
            name: index for index, name in enumerate(DENSE_C_EXTRA_FEATURES)
        }
        self.assertEqual(
            matrix[0, columns["syntax_predicate_internal_oblique"]], 1.0,
        )
        self.assertEqual(
            matrix[0, columns["syntax_predicate_external_oblique"]], 0.0,
        )
        self.assertEqual(
            matrix[0, columns["syntax_predicate_internal_object"]], 0.0,
        )
        from pipeline.phase2i_endpoint_scoring import _syntax_summary

        summary = _syntax_summary(record)
        self.assertIs(summary["predicate_internal_oblique"], True)
        self.assertIs(summary["predicate_external_oblique"], False)

        # With both an object and an oblique child, the two roles stay
        # separate on the internal and external sides.
        text2 = "you reset the wave by hand"
        parse2 = _fixture_parse(
            text2,
            [
                (1, "you", 0, 3, [1]),
                (2, "reset", 4, 9, [2]),
                (3, "the", 10, 13, [3]),
                (4, "wave", 14, 18, [4]),
                (5, "by", 19, 21, [5]),
                (6, "hand", 22, 26, [6]),
            ],
            [
                (1, 1, "you", "you", "PRON", "PRP", 2, "nsubj", 0, 3),
                (
                    2, 2, "reset", "reset", "VERB", "VBP", 0, "root",
                    4, 9, "VerbForm=Fin",
                ),
                (3, 3, "the", "the", "DET", "DT", 4, "det", 10, 13),
                (4, 4, "wave", "wave", "NOUN", "NN", 2, "obj", 14, 18),
                (5, 5, "by", "by", "ADP", "IN", 6, "case", 19, 21),
                (6, 6, "hand", "hand", "NOUN", "NN", 2, "obl", 22, 26),
            ],
        )
        full = compute_candidate_syntax(
            parse2, _row("c2", 4, 26, "reset the wave by hand"),
        )
        self.assertTrue(full.predicate_internal_object)
        self.assertTrue(full.predicate_internal_oblique)
        # Truncating the oblique leaves the object internal and the oblique
        # external, without cross-activating either role.
        without_oblique = compute_candidate_syntax(
            parse2, _row("c3", 4, 18, "reset the wave"),
        )
        self.assertTrue(without_oblique.predicate_internal_object)
        self.assertFalse(without_oblique.predicate_external_object)
        self.assertFalse(without_oblique.predicate_internal_oblique)
        self.assertTrue(without_oblique.predicate_external_oblique)

    def test_csubj_is_subject_not_complement(self):
        text = "resetting helps you"
        parse = _fixture_parse(
            text,
            [
                (1, "resetting", 0, 9, [1]),
                (2, "helps", 10, 15, [2]),
                (3, "you", 16, 19, [3]),
            ],
            [
                (
                    1, 1, "resetting", "reset", "VERB", "VBG", 2, "csubj",
                    0, 9, "VerbForm=Ger",
                ),
                (
                    2, 2, "helps", "help", "VERB", "VBZ", 0, "root",
                    10, 15, "VerbForm=Fin",
                ),
                (3, 3, "you", "you", "PRON", "PRP", 2, "obj", 16, 19),
            ],
        )
        record = compute_candidate_syntax(
            parse, _row("c1", 0, 19, text),
        )
        self.assertTrue(record.predicate_internal_subject)
        self.assertFalse(record.predicate_internal_complement)
        self.assertIn("subject:internal", record.predicate_argument_context)
        self.assertNotIn(
            "complement:internal", record.predicate_argument_context,
        )
        # The gerund is a non-finite csubj clause root (structural), not a
        # finite verb.
        self.assertNotIn("s1:w1", record.finite_verb_ids)
        self.assertEqual(record.clause_root_ids, ("s1:w1", "s1:w2"))


class Phase2IMwtBoundaryTests(unittest.TestCase):
    def test_aligned_mwt_endpoint_and_partial_ordinary_boundary(self):
        text = "we don't go"
        parse = _fixture_parse(
            text,
            [
                (1, "we", 0, 2, [1]),
                (2, "don't", 3, 8, [2, 3]),
                (4, "go", 9, 11, [4]),
            ],
            [
                (1, 1, "we", "we", "PRON", "PRP", 4, "nsubj", 0, 2),
                (2, 2, "do", "do", "AUX", "VBP", 4, "aux", 3, 8),
                (3, 2, "n't", "not", "PART", "RB", 4, "neg", 3, 8),
                (4, 4, "go", "go", "VERB", "VB", 0, "root", 9, 11),
            ],
        )
        # Fully aligned MWT endpoint: ordinary exact evidence.
        exact = compute_candidate_syntax(
            parse, _row("c1", 3, 8, "don't"),
        )
        self.assertEqual(exact.boundary_status, BOUNDARY_EXACT)
        self.assertNotIn("MULTIWORD_BOUNDARY_CUT", exact.ambiguity)
        # Start is aligned with the MWT token but the end partially cuts the
        # ordinary single-word token "go": PARTIAL, never AMBIGUOUS.
        partial = compute_candidate_syntax(
            parse, _row("c2", 3, 10, "don't g"),
        )
        self.assertEqual(partial.boundary_status, BOUNDARY_PARTIAL)
        self.assertNotIn("MULTIWORD_BOUNDARY_CUT", partial.ambiguity)
        # The end is aligned with the MWT token but the start partially cuts
        # the ordinary token "we": PARTIAL, never AMBIGUOUS.
        partial2 = compute_candidate_syntax(
            parse, _row("c3", 1, 8, "e don't"),
        )
        self.assertEqual(partial2.boundary_status, BOUNDARY_PARTIAL)
        self.assertNotIn("MULTIWORD_BOUNDARY_CUT", partial2.ambiguity)
        # An actual boundary cut through the MWT remains AMBIGUOUS.
        ambiguous = compute_candidate_syntax(
            parse, _row("c4", 4, 8, "on't"),
        )
        self.assertEqual(ambiguous.boundary_status, BOUNDARY_AMBIGUOUS)
        self.assertIn("MULTIWORD_BOUNDARY_CUT", ambiguous.ambiguity)


class Phase2ISerializationTests(unittest.TestCase):
    def test_parse_roundtrip_and_self_verification(self):
        text = "alpha beta gamma"
        parse = _fixture_parse(
            text,
            [
                (1, "alpha", 0, 5, [1]),
                (2, "beta", 6, 10, [2]),
                (3, "gamma", 11, 16, [3]),
            ],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
                (3, 3, "gamma", "gamma", "NOUN", "NN", 1, "obj", 11, 16),
            ],
        )
        restored = UdParse.from_dict(parse.to_dict())
        self.assertEqual(restored, parse)
        self.assertEqual(restored.parse_sha256, parse.parse_sha256)
        self.assertEqual(
            restored.canonical_serialization()["schema_version"],
            PARSE_SCHEMA_VERSION,
        )

    def test_failed_parse_schema_is_rejected(self):
        text = "alpha beta"
        parse = _fixture_parse(
            text,
            [(1, "alpha", 0, 5, [1]), (2, "beta", 6, 10, [2])],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
            ],
        )
        data = parse.to_dict()
        data["text_sha256"] = "0" * 64
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)
        data = parse.to_dict()
        data["parse_sha256"] = "0" * 64
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)
        data = parse.to_dict()
        data["sentences"][0]["words"][0]["head"] = -3
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict({"schema_version": PARSE_SCHEMA_VERSION})

    def test_candidate_offsets_must_stay_inside_bronze(self):
        parse = _fixture_parse(
            "alpha beta",
            [(1, "alpha", 0, 5, [1]), (2, "beta", 6, 10, [2])],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
            ],
        )
        with self.assertRaises(Phase2ISyntaxError):
            compute_candidate_syntax(parse, _row("bad", 9, 20, "beyond"))
        with self.assertRaises(Phase2ISyntaxError):
            compute_candidate_syntax(parse, _row("bad", 0, 5, "not alpha"))

    def test_empty_sentence_list_is_representable(self):
        parse = _parse_from_sentences("empty", "   ", ())
        record = compute_candidate_syntax(parse, _row("c1", 0, 3, "   "))
        self.assertEqual(record.boundary_status, BOUNDARY_UNALIGNED)
        restored = UdParse.from_dict(parse.to_dict())
        self.assertEqual(restored, parse)

    def test_non_whitespace_bronze_requires_complete_token_coverage(self):
        empty = _parse_from_sentences("empty", "alpha", ())
        with self.assertRaisesRegex(
            Phase2ISyntaxError, "do not cover every non-whitespace Bronze",
        ):
            UdParse.from_dict(empty.to_dict())

        missing_tail = _fixture_parse(
            "alpha beta",
            [(1, "alpha", 0, 5, [1])],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
            ],
        )
        with self.assertRaisesRegex(
            Phase2ISyntaxError, "do not cover every non-whitespace Bronze",
        ):
            UdParse.from_dict(missing_tail.to_dict())

    def _simple_parse_dict(self):
        parse = _fixture_parse(
            "alpha beta gamma",
            [
                (1, "alpha", 0, 5, [1]),
                (2, "beta", 6, 10, [2]),
                (3, "gamma", 11, 16, [3]),
            ],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
                (3, 3, "gamma", "gamma", "NOUN", "NN", 1, "obj", 11, 16),
            ],
        )
        return parse.to_dict()

    def test_token_surface_mismatch_is_rejected(self):
        data = self._simple_parse_dict()
        data["sentences"][0]["tokens"][0]["text"] = "alphax"
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)

    def test_raw_parse_schema_keys_and_json_types_are_exact(self):
        data = self._simple_parse_dict()
        data["hidden_payload"] = {"unexpected": True}
        with self.assertRaisesRegex(Phase2ISyntaxError, "key set"):
            UdParse.from_dict(data)

        data = self._simple_parse_dict()
        del data["processors"]
        with self.assertRaisesRegex(Phase2ISyntaxError, "key set"):
            UdParse.from_dict(data)

        data = self._simple_parse_dict()
        data["sentences"][0]["tokens"][0]["multiword"] = 1
        with self.assertRaisesRegex(Phase2ISyntaxError, "malformed"):
            UdParse.from_dict(data)

        data = self._simple_parse_dict()
        data["sentences"][0]["words"][0]["hidden"] = "payload"
        with self.assertRaisesRegex(Phase2ISyntaxError, "key set"):
            UdParse.from_dict(data)

    def test_parse_file_rejects_duplicate_keys_and_noncanonical_bytes(self):
        data = self._simple_parse_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parse.json"
            canonical = json.dumps(data, indent=2) + "\n"
            duplicate = canonical.replace(
                "{\n  \"parse_sha256\"",
                "{\n  \"window_id\": \"fabricated\",\n  \"parse_sha256\"",
                1,
            )
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(Phase2ISyntaxError, "duplicate JSON"):
                load_parse_artifact(path)

            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(Phase2ISyntaxError, "canonical"):
                load_parse_artifact(path)

    def test_parse_file_rejects_nonfinite_json_constants(self):
        data = self._simple_parse_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parse.json"
            canonical = json.dumps(data, indent=2) + "\n"
            for constant in ("NaN", "Infinity", "-Infinity"):
                path.write_text(
                    canonical.replace(
                        '"alpha beta gamma"', constant, 1,
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    Phase2ISyntaxError, "non-finite JSON number",
                ):
                    load_parse_artifact(path)

    def test_duplicate_word_ids_are_rejected(self):
        data = self._simple_parse_dict()
        words = data["sentences"][0]["words"]
        words.append(dict(words[0], text="alpha"))
        data["sentences"][0]["words"] = words
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)

    def test_missing_word_parent_is_rejected(self):
        data = self._simple_parse_dict()
        data["sentences"][0]["words"][1]["head"] = 99
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)

    def test_dependency_cycle_is_rejected(self):
        data = self._simple_parse_dict()
        # 1 -> 2 -> 1 cycle with no root.
        data["sentences"][0]["words"][0]["head"] = 2
        data["sentences"][0]["words"][1]["head"] = 1
        data["sentences"][0]["words"][2]["head"] = 1
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)

    def test_duplicate_token_ids_are_rejected(self):
        data = self._simple_parse_dict()
        tokens = data["sentences"][0]["tokens"]
        tokens.append(dict(tokens[0]))
        data["sentences"][0]["tokens"] = tokens
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)

    def test_word_referencing_missing_token_is_rejected(self):
        data = self._simple_parse_dict()
        data["sentences"][0]["words"][0]["token_id"] = 9
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)

    def test_word_offsets_must_agree_with_parent_token(self):
        data = self._simple_parse_dict()
        data["sentences"][0]["words"][0]["start_char"] = 1
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)

    def test_multiple_roots_are_rejected(self):
        data = self._simple_parse_dict()
        data["sentences"][0]["words"][1]["head"] = 0
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)

    def test_malformed_model_assets_and_manifest_are_rejected(self):
        data = self._simple_parse_dict()
        data["model_assets"] = [{"path": "en/tokenize/ewt.pt", "sha256": "x"}]
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)
        data = self._simple_parse_dict()
        data["model_assets"] = [{
            "path": "en/tokenize/ewt.pt",
            "sha256": "0" * 64,
        }]
        data["assets_manifest_sha256"] = "0" * 64
        with self.assertRaises(Phase2ISyntaxError):
            UdParse.from_dict(data)


class Phase2ISyntaxEvidenceTests(unittest.TestCase):
    def _parse(self):
        return _fixture_parse(
            "you reset after killing someone",
            [
                (1, "you", 0, 3, [1]),
                (2, "reset", 4, 9, [2]),
                (3, "after", 10, 15, [3]),
                (4, "killing", 16, 23, [4]),
                (5, "someone", 24, 31, [5]),
            ],
            [
                (1, 1, "you", "you", "PRON", "PRP", 2, "nsubj", 0, 3),
                (2, 2, "reset", "reset", "VERB", "VBP", 0, "root", 4, 9),
                (3, 3, "after", "after", "SCONJ", "IN", 4, "mark", 10, 15),
                (4, 4, "killing", "kill", "VERB", "VBG", 2, "advcl", 16, 23),
                (5, 5, "someone", "someone", "PRON", "NN", 4, "obj", 24, 31),
            ],
        )

    def test_distinct_relation_contexts(self):
        parse = self._parse()
        full = compute_candidate_syntax(
            parse, _row("c1", 4, 31, "reset after killing someone"),
        )
        # Head relation: reset is the sentence root.
        self.assertIn(
            "root:ROOT:VERB", full.group_values("head_relation_context"),
        )
        # Wholly-inside relations: mark/obj/advcl with inside governors.
        self.assertIn(
            "mark:VERB:SCONJ", full.group_values("internal_rel_context"),
        )
        self.assertIn(
            "obj:VERB:PRON", full.group_values("internal_rel_context"),
        )
        self.assertIn(
            "advcl:VERB:VERB", full.group_values("internal_rel_context"),
        )
        # Crossing outgoing: external dependent "you" headed inside.
        self.assertIn(
            "nsubj:VERB:PRON",
            full.group_values("crossing_outgoing_rel_context"),
        )
        self.assertEqual(
            full.group_values("crossing_incoming_rel_context"), (),
        )
        self.assertEqual(full.external_governor_lemmas, ())
        # Head child labels are exposed.
        self.assertEqual(
            set(full.head_child_deprels),
            {"nsubj", "advcl"},
        )
        self.assertEqual(full.head_dependent_count, 2)

    def test_crossing_incoming_and_external_governor_evidence(self):
        parse = self._parse()
        record = compute_candidate_syntax(
            parse, _row("c2", 16, 31, "killing someone"),
        )
        self.assertIn(
            "advcl:VERB:VERB",
            record.group_values("crossing_incoming_rel_context"),
        )
        self.assertIn(
            "mark:VERB:SCONJ",
            record.group_values("crossing_outgoing_rel_context"),
        )
        self.assertEqual(
            record.external_governor_lemmas, ("reset",),
        )
        self.assertEqual(record.external_governor_uposes, ("VERB",))
        self.assertEqual(record.external_governor_deprels, ("root",))
        self.assertEqual(record.external_governor_ids, ("s1:w2",))

    def test_subtree_fractions_are_bounded(self):
        parse = self._parse()
        full = compute_candidate_syntax(
            parse, _row("c1", 4, 31, "reset after killing someone"),
        )
        self.assertEqual(full.subtree_size, 5)
        self.assertEqual(full.subtree_intersection_count, 4)
        word_fraction = (
            full.subtree_intersection_count / len(full.word_ids)
        )
        subtree_fraction = (
            full.subtree_intersection_count / full.subtree_size
        )
        self.assertAlmostEqual(word_fraction, 1.0)
        self.assertAlmostEqual(subtree_fraction, 0.8)
        matrix = dense_c_matrix([full])
        columns = list(DENSE_C_EXTRA_FEATURES)
        self.assertAlmostEqual(
            matrix[0, columns.index("syntax_subtree_word_fraction")],
            word_fraction,
        )
        self.assertAlmostEqual(
            matrix[0, columns.index("syntax_subtree_fraction")],
            subtree_fraction,
        )
        # Every fraction column is bounded to [0, 1].
        for name in (
            "syntax_subtree_word_fraction",
            "syntax_subtree_fraction",
        ):
            self.assertGreaterEqual(matrix[0, columns.index(name)], 0.0)
            self.assertLessEqual(matrix[0, columns.index(name)], 1.0)

    def test_predicate_evidence_is_syntactic_not_action_lexicon(self):
        parse = self._parse()
        record = compute_candidate_syntax(
            parse, _row("c2", 16, 31, "killing someone"),
        )
        # "killing" is a syntactic predicate (VERB/advcl) even though its
        # lemma is not in the bounded action-token lexicon.
        self.assertEqual(record.syntactic_predicate_ids, ("s1:w4",))
        self.assertTrue(record.predicate_internal_object)
        self.assertFalse(record.predicate_internal_subject)
        self.assertIn(
            "object:internal",
            record.group_values("predicate_argument_context"),
        )
        self.assertIn(
            "mark:external",
            record.group_values("predicate_argument_context"),
        )
        matrix = dense_c_matrix([record])
        columns = list(DENSE_C_EXTRA_FEATURES)
        self.assertEqual(
            matrix[0, columns.index("syntax_predicate_count")], 1.0,
        )
        self.assertEqual(
            matrix[0, columns.index("syntax_has_predicate")], 1.0,
        )
        self.assertEqual(
            matrix[0, columns.index("syntax_predicate_internal_object")],
            1.0,
        )
        self.assertEqual(
            matrix[0, columns.index("syntax_predicate_external_mark")],
            1.0,
        )
        self.assertEqual(
            matrix[0, columns.index("syntax_external_governor_exists")],
            1.0,
        )
        self.assertEqual(
            matrix[0, columns.index("syntax_head_dependent_count")],
            2.0,
        )

    def test_new_categorical_groups_are_schema_exposed(self):
        schema = feature_schema_c()
        groups = schema["feature_set_C"]["categorical_groups"]
        for group in (
            "head_relation_context",
            "head_child_deprels",
            "internal_rel_context",
            "crossing_incoming_rel_context",
            "crossing_outgoing_rel_context",
            "external_governor_lemma",
            "external_governor_upos",
            "external_governor_deprel",
            "predicate_argument_context",
        ):
            self.assertIn(group, groups)


class Phase2IFeatureExtractionTests(unittest.TestCase):
    def test_feature_schema_is_versioned_and_complete(self):
        schema = feature_schema_c()
        self.assertEqual(schema["version"], SYNTAX_FEATURE_SCHEMA_VERSION)
        self.assertEqual(
            schema["feature_set_C"]["dense_extras"],
            list(DENSE_C_EXTRA_FEATURES),
        )
        self.assertEqual(schema["fit_scope"], "training windows only")
        self.assertIn(
            "generative endpoint predictions",
            schema["prohibited_features"],
        )

    def test_dense_matrix_is_deterministic_and_missing_free(self):
        text = "alpha beta gamma "
        parse = _fixture_parse(
            text,
            [
                (1, "alpha", 0, 5, [1]),
                (2, "beta", 6, 10, [2]),
                (3, "gamma", 11, 16, [3]),
            ],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
                (3, 3, "gamma", "gamma", "NOUN", "NN", 1, "obj", 11, 16),
            ],
        )
        records = [
            compute_candidate_syntax(parse, _row("c1", 0, 5, "alpha")),
            compute_candidate_syntax(parse, _row("c2", 6, 16, "beta gamma")),
            compute_candidate_syntax(parse, _row("c3", 16, 17, " ")),
        ]
        first = dense_c_matrix(records)
        second = dense_c_matrix(records)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, len(DENSE_C_EXTRA_FEATURES)))
        self.assertFalse(np.isnan(first).any())
        self.assertFalse(np.isinf(first).any())
        self.assertEqual(first[0, 0], 1)  # token count
        self.assertEqual(first[2, 7], 1)  # unaligned flag

    def test_syntax_encoder_fit_is_isolated_to_training_records(self):
        train = [
            {
                "head_lemma": ("alpha",),
                "rel_context": ("nsubj:VERB:PRON",),
                "modal_lemma": ("should",),
            },
            {
                "head_lemma": ("beta",),
                "rel_context": ("obj:VERB:NOUN",),
                "modal_lemma": (),
            },
        ]
        held_out = [
            {
                "head_lemma": ("gamma",),
                "rel_context": ("advcl:VERB:VERB",),
                "modal_lemma": ("would",),
            },
            {
                "head_lemma": ("alpha",),
                "rel_context": ("nsubj:VERB:PRON",),
                "modal_lemma": (),
            },
        ]
        encoder = SyntaxEncoder().fit(train)
        matrix = encoder.transform(held_out)
        self.assertEqual(matrix.shape[0], 2)
        names = encoder.feature_names()
        self.assertIn("syntax:head_lemma=alpha", names)
        self.assertNotIn("syntax:head_lemma=gamma", names)
        self.assertNotIn("syntax:modal_lemma=would", names)
        audit = encoder.oov_audit(held_out)
        self.assertIn("gamma", audit["per_group"]["head_lemma"])
        self.assertIn("would", audit["per_group"]["modal_lemma"])
        self.assertNotIn("alpha", audit["per_group"]["head_lemma"])
        self.assertEqual(audit["oov_value_count"], 3)
        # OOV rows produce zero columns for unknown values.
        row = matrix.toarray()[0]
        self.assertTrue(np.isclose(row.sum(), 0.0))

    def test_syntax_encoder_requires_fit_before_transform(self):
        encoder = SyntaxEncoder()
        with self.assertRaises(Phase2ISyntaxError):
            encoder.transform([{"head_lemma": ("a",)}])
        with self.assertRaises(Phase2ISyntaxError):
            encoder.feature_names()

    def test_evidence_sha256_is_deterministic(self):
        text = "alpha beta gamma"
        parse = _fixture_parse(
            text,
            [
                (1, "alpha", 0, 5, [1]),
                (2, "beta", 6, 10, [2]),
                (3, "gamma", 11, 16, [3]),
            ],
            [
                (1, 1, "alpha", "alpha", "NOUN", "NN", 0, "root", 0, 5),
                (2, 2, "beta", "beta", "NOUN", "NN", 1, "obj", 6, 10),
                (3, 3, "gamma", "gamma", "NOUN", "NN", 1, "obj", 11, 16),
            ],
        )
        first = compute_candidate_syntax(parse, _row("c1", 0, 5, "alpha"))
        second = compute_candidate_syntax(parse, _row("c1", 0, 5, "alpha"))
        self.assertEqual(first.evidence_sha256(), second.evidence_sha256())
        self.assertEqual(first, second)

    def test_train_only_preprocessor_isolation(self):
        from pipeline.phase2i_endpoint_scoring import CellPreprocessorC
        from pipeline.phase2h_endpoint_scoring import (
            DENSE_A_FEATURES,
            DENSE_B_EXTRA_FEATURES,
        )

        dense_count = (
            len(DENSE_A_FEATURES)
            + len(DENSE_B_EXTRA_FEATURES)
            + len(DENSE_C_EXTRA_FEATURES)
        )
        train_dense = np.zeros((2, dense_count), dtype=np.float64)
        train_dense[:, 0] = (1.0, 3.0)
        train_dense[:, 1] = (2.0, 4.0)
        test_dense = np.zeros((1, dense_count), dtype=np.float64)
        test_dense[:, 0] = 100.0
        test_dense[:, 1] = 200.0
        train_texts = ["alpha alpha", "beta beta"]
        test_texts = ["gamma only-holdout"]
        train_boundaries = [("alpha", "alpha", "alpha"), ("beta", "beta", "beta")]
        test_boundaries = [("gamma", "holdout", "holdout")]
        train_syntax = [
            {"head_lemma": ("alpha",), "rel_context": ("nsubj:VERB:PRON",)},
            {"head_lemma": ("beta",), "rel_context": ("obj:VERB:NOUN",)},
        ]
        test_syntax = [
            {
                "head_lemma": ("gamma",),
                "rel_context": ("advcl:VERB:VERB",),
            },
        ]
        preprocessor = CellPreprocessorC()
        preprocessor.fit(
            train_dense, train_texts, train_boundaries, train_syntax,
        )
        transformed = preprocessor.transform(
            test_dense, test_texts, test_boundaries, test_syntax,
        )
        names = preprocessor.feature_names([
            *DENSE_A_FEATURES,
            *DENSE_B_EXTRA_FEATURES,
            *DENSE_C_EXTRA_FEATURES,
        ])
        self.assertIn("syntax:head_lemma=alpha", names)
        self.assertNotIn("syntax:head_lemma=gamma", names)
        self.assertNotIn("ngram=gamma", names)
        # Scaler was fit on training rows only: mean 2/3 for column 0.
        self.assertAlmostEqual(
            float(preprocessor.scaler.mean_[0]), 2.0, places=10,
        )
        self.assertEqual(transformed.shape[0], 1)


class Phase2IRealParseTests(unittest.TestCase):
    def test_real_parse_is_deterministic_and_mwt_aware(self):
        if not ASSETS.is_dir():
            self.skipTest("parser assets not present")
        text = "I don't wanna go; we're gonna win."
        first = parse_window_text(text, "det", assets_dir=ASSETS)
        second = parse_window_text(text, "det", assets_dir=ASSETS)
        self.assertEqual(first, second)
        self.assertEqual(first.parse_sha256, second.parse_sha256)
        self.assertEqual(first.window_id, "det")
        self.assertEqual(first.text, text)
        self.assertEqual(first.parser, "stanza")
        self.assertEqual(first.parser_version, STANZA_VERSION)
        self.assertEqual(first.package, "ewt")
        self.assertEqual(set(first.processors), set(STANZA_PROCESSORS))
        self.assertEqual(
            first.assets_manifest_sha256, LOCKED_ASSETS_MANIFEST_SHA256,
        )
        provenance = verify_assets_provenance(ASSETS)
        locked_files = {
            entry["path"]: entry["sha256"]
            for entry in provenance["files"]
        }
        self.assertEqual(dict(first.model_assets), locked_files)
        words = [
            word
            for sentence in first.sentences
            for word in sentence.words
        ]
        projected = [w for w in words if w.is_mwt_projected]
        self.assertGreaterEqual(len(projected), 4)
        for word in projected:
            self.assertIsNotNone(word.start_char)
            self.assertIsNotNone(word.end_char)


class Phase2IRealFinitenessTests(unittest.TestCase):
    def _record(self, text):
        parse = parse_window_text(text, "fin", assets_dir=ASSETS)
        record = compute_candidate_syntax(
            parse, _row("c1", 0, len(text), text),
        )
        words = _word_by_key(parse)
        finite = {words[key].text for key in record.finite_verb_ids}
        roots = {words[key].text for key in record.clause_root_ids}
        return parse, record, finite, roots

    def _run_construction(self, text, finite, roots):
        if not ASSETS.is_dir():
            self.skipTest("parser assets not present")
        _, _, actual_finite, actual_roots = self._record(text)
        self.assertEqual(actual_finite, finite)
        self.assertEqual(actual_roots, roots)

    def test_real_perfect_progressive_passive_and_modal(self):
        self._run_construction(
            "he has reset the wave", {"has"}, {"reset"},
        )
        self._run_construction(
            "you are pushing the wave", {"are"}, {"pushing"},
        )
        self._run_construction(
            "the wave is reset by you", {"is"}, {"reset"},
        )
        self._run_construction(
            "you should reset the wave", {"should"}, {"reset"},
        )

    def test_real_participial_infinitive_and_imperative(self):
        self._run_construction(
            "killing someone, you reset the wave",
            {"reset"}, {"killing", "reset"},
        )
        self._run_construction(
            "you want to reset the wave",
            {"want"}, {"want", "reset"},
        )
        self._run_construction(
            "reset the wave", {"reset"}, {"reset"},
        )


class _ProvenanceFixtureMixin:
    def _write_provenance(self, assets_dir, files, **overrides):
        provenance = {
            "schema_version": "phase2i-parser-assets-v1",
            "stanza_version": "1.14.0",
            "package": "ewt",
            "processors": list(STANZA_PROCESSORS),
            "created_at": "2026-01-01T00:00:00Z",
            "files": files,
            "manifest_sha256": canonical_sha256([
                {"path": entry["path"], "sha256": entry["sha256"]}
                for entry in files
            ]),
            **overrides,
        }
        path = assets_dir / "ASSET_PROVENANCE.json"
        path.write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _asset(self, assets_dir, relative, content=b"data"):
        path = assets_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
        }


class Phase2IAssetProvenanceTests(_ProvenanceFixtureMixin, unittest.TestCase):
    def test_valid_provenance_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp)
            files = [
                self._asset(assets_dir, "en/tokenize/ewt.pt"),
                self._asset(assets_dir, "resources.json"),
            ]
            self._write_provenance(assets_dir, files)
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertTrue(result["verified"], result.get("problems"))

    def test_noncanonical_file_entry_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp)
            files = [
                self._asset(assets_dir, "en/tokenize/ewt.pt"),
                self._asset(assets_dir, "resources.json"),
            ]
            # Reverse the two canonical entries; the manifest hash is
            # recomputed over the reversed list so the only defect is the
            # ordering itself.
            self._write_provenance(assets_dir, list(reversed(files)))
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any(
                    "canonical path order" in item
                    for item in result["problems"]
                ),
                result,
            )

    def test_malformed_schema_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp)
            files = [
                self._asset(assets_dir, "en/tokenize/ewt.pt"),
            ]
            self._write_provenance(
                assets_dir, files, schema_version="wrong",
            )
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("schema_version" in item for item in result["problems"]),
            )
            files = [
                self._asset(assets_dir, "en/tokenize/ewt.pt"),
                {"path": "en/tokenize/ewt.pt", "sha256": "0" * 64},
            ]
            self._write_provenance(assets_dir, files)
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("duplicated" in item for item in result["problems"]),
            )

            files = [self._asset(assets_dir, "en/tokenize/ewt.pt")]
            path = self._write_provenance(
                assets_dir, files, hidden_payload={"unexpected": True},
            )
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("top-level key set" in item for item in result["problems"]),
                result,
            )

            body = json.loads(path.read_text(encoding="utf-8"))
            del body["hidden_payload"]
            body["files"][0]["hidden_payload"] = True
            path.write_text(
                json.dumps(body, indent=2) + "\n", encoding="utf-8",
            )
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("file entry is invalid" in item for item in result["problems"]),
                result,
            )

    def test_extra_on_disk_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp)
            files = [
                self._asset(assets_dir, "en/tokenize/ewt.pt"),
            ]
            self._write_provenance(assets_dir, files)
            self._asset(assets_dir, "en/pos/ewt.pt", b"extra")
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("unlisted asset" in item for item in result["problems"]),
            )

    def test_symlinked_parent_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets_dir = root / "assets"
            assets_dir.mkdir()
            external = root / "external"
            external.mkdir()
            content = b"model weights"
            (external / "tokenize").mkdir()
            (external / "tokenize" / "ewt.pt").write_bytes(content)
            (assets_dir / "en").symlink_to(
                external, target_is_directory=True,
            )
            files = [{
                "path": "en/tokenize/ewt.pt",
                "sha256": hashlib.sha256(content).hexdigest(),
            }]
            self._write_provenance(assets_dir, files)
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("symlink" in item for item in result["problems"]),
                result.get("problems"),
            )

    def test_symlinked_asset_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets_dir = root / "assets"
            assets_dir.mkdir()
            external = root / "external.pt"
            content = b"model weights"
            external.write_bytes(content)
            (assets_dir / "en").mkdir()
            (assets_dir / "en" / "tokenize").mkdir()
            (assets_dir / "en" / "tokenize" / "ewt.pt").symlink_to(external)
            files = [{
                "path": "en/tokenize/ewt.pt",
                "sha256": hashlib.sha256(content).hexdigest(),
            }]
            self._write_provenance(assets_dir, files)
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("symlink" in item for item in result["problems"]),
                result.get("problems"),
            )

    def test_symlinked_assets_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.mkdir()
            content = b"model weights"
            (target / "en").mkdir()
            (target / "en" / "tokenize").mkdir()
            (target / "en" / "tokenize" / "ewt.pt").write_bytes(content)
            files = [{
                "path": "en/tokenize/ewt.pt",
                "sha256": hashlib.sha256(content).hexdigest(),
            }]
            self._write_provenance(target, files)
            link = root / "assets-link"
            link.symlink_to(target, target_is_directory=True)
            result = verify_assets_provenance(
                link, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("symlink" in item for item in result["problems"]),
                result.get("problems"),
            )

    def test_symlinked_ancestor_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            assets_dir = real / "assets"
            assets_dir.mkdir()
            files = [
                self._asset(assets_dir, "en/tokenize/ewt.pt"),
            ]
            self._write_provenance(assets_dir, files)
            link = root / "linked-parent"
            link.symlink_to(real, target_is_directory=True)
            result = verify_assets_provenance(
                link / "assets", require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("symlink" in item for item in result["problems"]),
                result.get("problems"),
            )

    def test_proc_self_cwd_style_ancestor_symlink_is_rejected(self):
        if not Path("/proc/self/cwd").is_symlink():
            self.skipTest("/proc/self/cwd is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "phase2i_assets"
            real.mkdir()
            supplied = Path("/proc/self/cwd") / str(real).lstrip("/")
            check = verify_parser_asset_path(supplied)
            self.assertFalse(check["verified"])
            self.assertTrue(
                any("symlink" in item for item in check["problems"]),
                check.get("problems"),
            )
            # The fail-closed preflight runs before any walk/hash of the
            # supplied path, so provenance verification also rejects it.
            provenance = verify_assets_provenance(
                supplied, require_locked=False,
            )
            self.assertFalse(provenance["verified"])
            self.assertTrue(
                any(
                    "symlink" in item
                    for item in provenance["problems"]
                ),
                provenance.get("problems"),
            )

    def test_normal_absolute_asset_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp) / "assets"
            files = [
                self._asset(assets_dir, "en/tokenize/ewt.pt"),
            ]
            self._write_provenance(assets_dir, files)
            check = verify_parser_asset_path(assets_dir)
            self.assertTrue(check["verified"], check.get("problems"))
            self.assertEqual(check["path"], str(assets_dir))
            result = verify_assets_provenance(
                str(assets_dir), require_locked=False,
            )
            self.assertTrue(result["verified"], result.get("problems"))

    def test_is_sha256_hex_helper(self):
        self.assertTrue(is_sha256_hex("0" * 64))
        self.assertFalse(is_sha256_hex("0" * 63))
        self.assertFalse(is_sha256_hex("G" + "0" * 63))
        self.assertFalse(is_sha256_hex(""))


class Phase2ISetupAssetPathTests(unittest.TestCase):
    def test_setup_preflight_rejects_temp_ancestor_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            link = root / "assets-link"
            link.symlink_to(real, target_is_directory=True)
            with mock.patch("builtins.print") as fake_print:
                prepared = _prepare_asset_path(link / "phase2i_assets")
            self.assertIsNone(prepared)
            messages = " ".join(
                " ".join(str(arg) for arg in call.args)
                for call in fake_print.call_args_list
            )
            self.assertIn("symlink", messages)

    def test_setup_preflight_accepts_normal_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase2i_assets"
            prepared = _prepare_asset_path(path)
            self.assertEqual(prepared, path)

    def test_setup_main_rejects_ancestor_symlink_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            link = root / "assets-link"
            link.symlink_to(real, target_is_directory=True)
            import stanza
            with mock.patch.object(
                stanza, "download",
                side_effect=AssertionError("network download attempted"),
            ), mock.patch("builtins.print"):
                code = setup_main([
                    "--assets-dir", str(link / "phase2i_assets"),
                ])
            self.assertEqual(code, 1)
            self.assertFalse((link / "phase2i_assets").exists())


class Phase2IParserFailClosedTests(
    _ProvenanceFixtureMixin, unittest.TestCase,
):
    def test_locked_asset_provenance_verifies(self):
        if not ASSETS.is_dir():
            self.skipTest("parser assets not present")
        result = verify_assets_provenance(ASSETS)
        self.assertTrue(result["verified"], result.get("problems"))
        self.assertEqual(
            result["manifest_sha256"], LOCKED_ASSETS_MANIFEST_SHA256,
        )
        self.assertEqual(result["stanza_version"], STANZA_VERSION)
        self.assertEqual(result["package"], "ewt")
        self.assertEqual(
            list(result["processors"]), list(STANZA_PROCESSORS),
        )

    def test_nonlocked_manifest_and_wrong_version_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp)
            files = [
                self._asset(assets_dir, "en/tokenize/ewt.pt"),
            ]
            self._write_provenance(assets_dir, files)
            result = verify_assets_provenance(assets_dir)
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("locked" in item for item in result["problems"]),
            )
            self._write_provenance(
                assets_dir, files, stanza_version="1.15.0",
            )
            result = verify_assets_provenance(
                assets_dir, require_locked=False,
            )
            self.assertFalse(result["verified"])
            self.assertTrue(
                any(
                    "stanza_version" in item
                    for item in result["problems"]
                ),
            )

    def test_parse_requires_exact_runtime_version(self):
        import stanza
        with mock.patch.object(
            stanza, "__version__", "1.15.0",
        ):
            with self.assertRaises(Phase2IParseError) as caught:
                parse_window_text(
                    "reset the wave", "det", assets_dir=ASSETS,
                )
            self.assertIn("1.15.0", str(caught.exception))

    def test_parse_fails_closed_on_provenance_failure(self):
        failure = {
            "verified": False,
            "problems": ["asset manifest sha256 does not self-verify"],
        }
        with mock.patch(
            "pipeline.phase2i_syntax.verify_assets_provenance",
            return_value=failure,
        ):
            with self.assertRaises(Phase2IParseError) as caught:
                parse_window_text(
                    "reset the wave", "det", assets_dir=ASSETS,
                )
            self.assertIn("self-verify", str(caught.exception))

    def test_parse_fails_closed_on_locked_manifest_mismatch(self):
        wrong_manifest = {
            "verified": True,
            "problems": [],
            "manifest_sha256": "0" * 64,
            "files": [],
        }
        with mock.patch(
            "pipeline.phase2i_syntax.verify_assets_provenance",
            return_value=wrong_manifest,
        ):
            with self.assertRaises(Phase2IParseError) as caught:
                parse_window_text(
                    "reset the wave", "det", assets_dir=ASSETS,
                )
            self.assertIn("locked", str(caught.exception))


class Phase2IBenchmarkInvarianceTests(unittest.TestCase):
    def test_candidate_identity_offsets_and_folds_are_unchanged(self):
        benchmark = load_benchmark(BENCHMARK)
        dataset = build_dataset(benchmark)
        self.assertEqual(len(dataset["windows"]), 5)
        total = sum(
            len(window["rows"]) for window in dataset["windows"].values()
        )
        self.assertEqual(total, 16624)
        self.assertEqual(
            sum(
                1
                for window in dataset["windows"].values()
                for row in window["rows"]
                if row.is_gold_positive
            ),
            33,
        )
        for window_id, window in dataset["windows"].items():
            for row in window["rows"]:
                self.assertTrue(
                    row.candidate_id.startswith(row.window_id + ":m"),
                )
                self.assertEqual(
                    window["bronze_text"][row.start:row.end],
                    row.text,
                )

    def test_frozen_baseline_hashes_and_metrics(self):
        from pipeline.phase2i_endpoint_scoring import (
            PHASE2H_RUN1_AGGREGATE_SHA256,
            PHASE2H_RUN1_ARCHIVE_SHA256,
            load_phase2h_baseline,
            close_phase2h_baseline,
        )

        archive = (
            ROOT / "data/phase2h_artifacts/"
            "phase2h-endpoint-scoring-run1.tar.gz"
        )
        import hashlib as _hashlib
        self.assertEqual(
            _hashlib.sha256(archive.read_bytes()).hexdigest(),
            PHASE2H_RUN1_ARCHIVE_SHA256,
        )
        baseline = load_phase2h_baseline(archive)
        try:
            aggregate = baseline["aggregate"]
            self.assertEqual(
                aggregate["content_sha256"],
                PHASE2H_RUN1_AGGREGATE_SHA256,
            )
            self.assertEqual(
                aggregate["input_hashes"]["benchmark_content_sha256"],
                "a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd",
            )
            b_logistic = aggregate["metrics"]["logistic_B"]
            self.assertAlmostEqual(
                b_logistic["precision"]["rate"], 0.08771929824561403,
            )
            self.assertAlmostEqual(
                b_logistic["recall"]["rate"], 0.30303030303030304,
            )
            self.assertAlmostEqual(
                b_logistic["f1"]["value"], 0.1360544217687075,
            )
            self.assertAlmostEqual(
                b_logistic["average_precision"]["value"], 0.08127472712518234,
            )
            self.assertAlmostEqual(
                b_logistic["roc_auc"]["value"], 0.939700787027651,
            )
            self.assertEqual(b_logistic["recall_at_k"]["10"]["hit_count"], 6)
            self.assertEqual(b_logistic["gold_rank"]["median"], 92.0)
            self.assertEqual(b_logistic["selected"], 114)
            b_lightgbm = aggregate["metrics"]["lightgbm_B"]
            self.assertAlmostEqual(
                b_lightgbm["precision"]["rate"], 0.02756892230576441,
            )
            self.assertAlmostEqual(
                b_lightgbm["recall"]["rate"], 0.3333333333333333,
            )
            self.assertEqual(b_lightgbm["recall_at_k"]["10"]["hit_count"], 4)
            self.assertEqual(b_lightgbm["gold_rank"]["median"], 157.0)
            self.assertEqual(b_lightgbm["selected"], 399)
        finally:
            close_phase2h_baseline(baseline)

    def test_universally_missed_derivation_matches_frozen_audit(self):
        from pipeline.phase2i_endpoint_scoring import (
            UNIVERSALLY_MISSED_LOCK,
            close_phase2h_baseline,
            derive_universally_missed,
            load_phase2h_baseline,
            validate_universally_missed,
        )

        archive = (
            ROOT / "data/phase2h_artifacts/"
            "phase2h-endpoint-scoring-run1.tar.gz"
        )
        baseline = load_phase2h_baseline(archive)
        try:
            missed = derive_universally_missed(baseline["window_tables"])
            self.assertEqual(len(missed), 7)
            problems = validate_universally_missed(missed)
            self.assertEqual(problems, [])
            expected_texts = {
                candidate_id: text
                for candidate_id, text in UNIVERSALLY_MISSED_LOCK
            }
            for entry in missed:
                self.assertEqual(
                    entry["text"], expected_texts[entry["candidate_id"]],
                )
        finally:
            close_phase2h_baseline(baseline)


if __name__ == "__main__":
    unittest.main()
