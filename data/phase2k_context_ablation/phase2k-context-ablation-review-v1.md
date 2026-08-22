# Phase 2K Full-Transcript Ablation — Human Review Packet

Human-reviewable A/B comparison per target. Score semantic recovery only; do not credit prose fluency.

Scoring scales:

- correctness: CORRECT, PARTIAL, INCORRECT, ABSENT_CORRECTLY
- unsupported_inference: NONE, MINOR, MAJOR
- source_grounding: GROUNDED, PARTIAL, UNGROUNDED, NOT_APPLICABLE

Strict success = correctness in {CORRECT, ABSENT_CORRECTLY} AND unsupported_inference = NONE AND source_grounding in {GROUNDED, NOT_APPLICABLE}

Do NOT score prose quality.  A more fluent answer earns nothing unless its semantic recovery is actually better.

---

## TARGET p2k:case:0001 (selection rank 1)

### Exact Bronze

```text
play with flash because like you're going exhaust SM but you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot because you probably have ultimate Hunter eyeball right yeah I do right that doesn't mean this is bad like I would
```

Location in full transcript: [4942:5222]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (2 items)
- [1] Coached player (student): repeatedly addressed as 'you' and self-referring as 'I' when confirming his loadout ('yeah I do'); the spell, rune-loss, and fighting-power claims concern him. (resolution: context_resolved)
  - cite[174:215] byte-exact: 'you probably have ultimate Hunter eyeball'
  - cite[222:231] byte-exact: 'yeah I do'
- [2] Coach (speaker/advisor): implicit second participant who recommends playing with Flash and evaluates the player's setup; inferred from dialogue structure (prompt followed by the player's confirmation). (resolution: context_resolved)
  - cite[0:15] byte-exact: 'play with flash'
  - cite[216:237] byte-exact: 'right yeah I do right'

#### reference_bindings (10 items)
- [1] 'you' in 'you're going exhaust SM' binds to the coached player whose spell setup is under discussion. (resolution: context_resolved)
  - cite[29:52] byte-exact: "you're going exhaust SM"
- [2] 'you' in 'you have less fighting Power' binds to the coached player. (resolution: context_resolved)
  - cite[57:85] byte-exact: 'you have less fighting Power'
- [3] 'you' in 'you're losing Last Stand' binds to the coached player. (resolution: context_resolved)
  - cite[94:118] byte-exact: "you're losing Last Stand"
- [4] 'you' in 'you're losing Harvest' binds to the coached player. (resolution: context_resolved)
  - cite[119:140] byte-exact: "you're losing Harvest"
- [5] 'you' in 'you're losing cheap shot' binds to the coached player. (resolution: context_resolved)
  - cite[141:165] byte-exact: "you're losing cheap shot"
- [6] 'you' in 'you probably have ultimate Hunter eyeball' binds to the coached player. (resolution: context_resolved)
  - cite[174:215] byte-exact: 'you probably have ultimate Hunter eyeball'
- [7] 'I' in 'yeah I do' binds to the coached player, confirming possession of the loadout elements just asked about. (resolution: context_resolved)
  - cite[222:231] byte-exact: 'yeah I do'
  - cite[174:215] byte-exact: 'you probably have ultimate Hunter eyeball'
- [8] 'that' in 'that doesn't mean this is bad' refers back to the immediately preceding point (the player running ultimate Hunter and 'eyeball' in place of the lost loadout elements), scoped to the setup under discussion. (resolution: context_resolved)
  - cite[238:267] byte-exact: "that doesn't mean this is bad"
  - cite[166:215] byte-exact: 'because you probably have ultimate Hunter eyeball'
- [9] 'this' in 'that doesn't mean this is bad' refers to the player's current setup, but its precise scope (spell choice vs rune setup) cannot be pinned down from the supplied material. (resolution: unresolved)
  - cite[238:267] byte-exact: "that doesn't mean this is bad"
- [10] 'I' in the truncated closing fragment 'like I would': the sentence breaks off mid-thought; the speaker (plausibly the coach) and intended continuation cannot be recovered. (resolution: unresolved)
  - cite[268:280] byte-exact: 'like I would'

#### abilities_resources (8 items)
- [1] Flash: summoner spell the speaker urges the player to play with; 'flash' matches the supplied summoner-spell vocabulary. (resolution: vocabulary_supported)
  - cite[0:15] byte-exact: 'play with flash'
- [2] Exhaust: summoner spell the coached player is 'going' (taking); 'exhaust' matches the supplied summoner-spell vocabulary. (resolution: vocabulary_supported)
  - cite[29:52] byte-exact: "you're going exhaust SM"
- [3] 'SM': unrecovered ASR-corrupted token adjacent to 'exhaust'; plausibly 'Smite' given the summoner-spell vocabulary and jungle context, but not confirmable from the supplied material. (resolution: unresolved)
  - cite[42:52] byte-exact: 'exhaust SM'
- [4] 'Last Stand': loadout element (rune-name surface form) the coached player is said to be losing. (resolution: literal_explicit)
  - cite[94:118] byte-exact: "you're losing Last Stand"
- [5] 'Harvest': partial loadout-element name the coached player is said to be losing; the full canonical name (e.g., a longer name ending in 'Harvest') is not recoverable from the supplied material. (resolution: unresolved)
  - cite[119:140] byte-exact: "you're losing Harvest"
- [6] 'cheap shot': loadout element (rune-name surface form) the coached player is said to be losing. (resolution: literal_explicit)
  - cite[141:165] byte-exact: "you're losing cheap shot"
- [7] 'ultimate Hunter': loadout element (rune-name surface form) the coached player is said to probably have; hedged by 'probably'. (resolution: literal_explicit)
  - cite[174:215] byte-exact: 'you probably have ultimate Hunter eyeball'
- [8] 'eyeball': partial loadout-element name the coached player is said to probably have alongside 'ultimate Hunter'; full canonical name not recoverable from the supplied material. (resolution: unresolved)
  - cite[192:215] byte-exact: 'ultimate Hunter eyeball'

#### events_actions (2 items)
- [1] The coached player is committing to a spell setup built around Exhaust (together with the unrecovered 'SM'), the alternative to the Flash setup the coach raises. (resolution: context_resolved)
  - cite[29:52] byte-exact: "you're going exhaust SM"
  - cite[0:15] byte-exact: 'play with flash'
- [2] The coached player verbally confirms, in response to the coach's prompt, that he has the 'ultimate Hunter'/'eyeball' loadout elements. (resolution: context_resolved)
  - cite[216:237] byte-exact: 'right yeah I do right'
  - cite[174:215] byte-exact: 'you probably have ultimate Hunter eyeball'

#### states (3 items)
- [1] The coached player has less fighting power under the discussed setup. (resolution: literal_explicit)
  - cite[53:85] byte-exact: 'but you have less fighting Power'
- [2] The coached player runs 'ultimate Hunter' together with 'eyeball'; raised as a hedge ('probably') by the coach and then confirmed by the player. (resolution: context_resolved)
  - cite[174:215] byte-exact: 'you probably have ultimate Hunter eyeball'
  - cite[222:231] byte-exact: 'yeah I do'
- [3] The coached player is giving up Last Stand, Harvest, and cheap shot under the discussed configuration ('losing' interpreted as not having/forgoing them). (resolution: context_resolved)
  - cite[94:165] byte-exact: "you're losing Last Stand you're losing Harvest you're losing cheap shot"

#### conditions (0 items)

(none)

#### recommended_advice (1 items)
- [1] The coach advises the player to play with Flash (against the Exhaust-based setup being discussed). (resolution: literal_explicit)
  - cite[0:15] byte-exact: 'play with flash'

#### consequences_outcomes (3 items)
- [1] Outcome: the player ends up with less fighting power, attributed by the coach to losing Last Stand, Harvest, and cheap shot. (resolution: literal_explicit)
  - cite[57:165] byte-exact: "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [2] Stated cause of the losses: running 'ultimate Hunter' and 'eyeball' is given as the reason the player loses cheap shot (and the other listed loadout elements). (resolution: literal_explicit)
  - cite[141:215] byte-exact: "you're losing cheap shot because you probably have ultimate Hunter eyeball"
- [3] Evaluation after the trade-off: despite the losses, the coach states this does not mean the player's setup is bad. (resolution: literal_explicit)
  - cite[238:267] byte-exact: "that doesn't mean this is bad"

#### explicit_relationships (6 items)
- [1] coached player --USES--> Exhaust (with unrecovered 'SM'): the player's chosen spells per 'you're going exhaust SM'. (resolution: context_resolved; relation: USES)
  - cite[29:52] byte-exact: "you're going exhaust SM"
- [2] coached player --USES--> 'ultimate Hunter' and 'eyeball' loadout elements: hedged by the coach, then confirmed with 'yeah I do'. (resolution: context_resolved; relation: USES)
  - cite[174:215] byte-exact: 'you probably have ultimate Hunter eyeball'
  - cite[222:231] byte-exact: 'yeah I do'
- [3] losing Last Stand, Harvest, and cheap shot --CAUSES--> the player having less fighting power (explicit 'because' linkage). (resolution: literal_explicit; relation: CAUSES)
  - cite[57:165] byte-exact: "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [4] having 'ultimate Hunter' and 'eyeball' --CAUSES--> losing the listed loadout elements (explicit 'because' linkage). (resolution: literal_explicit; relation: CAUSES)
  - cite[94:215] byte-exact: "you're losing Last Stand you're losing Harvest you're losing cheap shot because you probably have ultimate Hunter eye..."
- [5] coach's reassurance 'that doesn't mean this is bad' --NEGATES--> the conclusion that the player's setup is bad. (resolution: literal_explicit; relation: NEGATES)
  - cite[238:267] byte-exact: "that doesn't mean this is bad"
- [6] confirmation 'yeah I do' --REFERS_TO--> the coach's hedged question about having 'ultimate Hunter' and 'eyeball'. (resolution: context_resolved; relation: REFERS_TO)
  - cite[216:237] byte-exact: 'right yeah I do right'
  - cite[174:215] byte-exact: 'you probably have ultimate Hunter eyeball'

#### uncertainty_unresolved (6 items)
- [1] Token 'SM' after 'exhaust' is ASR-corrupted and unrecovered; 'Smite' (present in the supplied summoner-spell list) is a plausible reading but cannot be confirmed. (resolution: unresolved)
  - cite[42:52] byte-exact: 'exhaust SM'
- [2] 'Harvest' appears to be a truncated loadout/rune name (e.g., a longer name ending in 'Harvest'); the canonical full name is unrecoverable from the supplied material. (resolution: unresolved)
  - cite[119:140] byte-exact: "you're losing Harvest"
- [3] 'eyeball' likewise appears truncated (e.g., a longer rune name containing 'Eyeball'); the canonical form is unrecoverable from the supplied material. (resolution: unresolved)
  - cite[192:215] byte-exact: 'ultimate Hunter eyeball'
- [4] Closing fragment 'like I would' cuts off mid-sentence; the intended continuation and its content cannot be recovered. (resolution: unresolved)
  - cite[268:280] byte-exact: 'like I would'
- [5] Residual referential ambiguity in 'that doesn't mean this is bad': whether 'this' scopes over the spell choice, the rune setup, or both is not fully determined. (resolution: unresolved)
  - cite[238:267] byte-exact: "that doesn't mean this is bad"
- [6] The coach's hedge 'probably' leaves initial doubt about the player's 'ultimate Hunter'/'eyeball' loadout; only the subsequent 'yeah I do' resolves it. (resolution: context_resolved)
  - cite[166:215] byte-exact: 'because you probably have ultimate Hunter eyeball'
  - cite[222:231] byte-exact: 'yeah I do'

#### supporting_source_spans (10 items)
- [1] Span covering the Flash recommendation. (resolution: literal_explicit)
  - cite[0:15] byte-exact: 'play with flash'
  - span [0:15]
- [2] Span covering the player's Exhaust-based spell setup and the unrecovered 'SM'. (resolution: literal_explicit)
  - cite[16:52] byte-exact: "because like you're going exhaust SM"
  - span [16:52]
- [3] Span covering the reduced-fighting-power claim. (resolution: literal_explicit)
  - cite[53:85] byte-exact: 'but you have less fighting Power'
  - span [53:85]
- [4] Span covering the loss of Last Stand. (resolution: literal_explicit)
  - cite[86:118] byte-exact: "because you're losing Last Stand"
  - span [86:118]
- [5] Span covering the loss of Harvest. (resolution: literal_explicit)
  - cite[119:140] byte-exact: "you're losing Harvest"
  - span [119:140]
- [6] Span covering the loss of cheap shot. (resolution: literal_explicit)
  - cite[141:165] byte-exact: "you're losing cheap shot"
  - span [141:165]
- [7] Span covering the probable 'ultimate Hunter'/'eyeball' holdings and their causal role. (resolution: literal_explicit)
  - cite[166:215] byte-exact: 'because you probably have ultimate Hunter eyeball'
  - span [166:215]
- [8] Span covering the dialogue exchange (prompt and player confirmation). (resolution: literal_explicit)
  - cite[216:237] byte-exact: 'right yeah I do right'
  - span [216:237]
- [9] Span covering the reassurance that the setup is not bad. (resolution: literal_explicit)
  - cite[238:267] byte-exact: "that doesn't mean this is bad"
  - span [238:267]
- [10] Span covering the truncated closing fragment. (resolution: unresolved)
  - cite[268:280] byte-exact: 'like I would'
  - span [268:280]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (3 items)
- [1] Karthus (surface ASR form 'carus'), the champion the coached player is playing; bound from adjacent transcript context. (resolution: context_resolved)
  - cite[4492:4553] byte-exact: 'for carus you are never allowed to go Smite and exhaust again'
- [2] The coached player, addressed as 'you', whose summoner spells and runes are under discussion. (resolution: context_resolved)
  - cite[4971:5027] byte-exact: "you're going exhaust SM but you have less fighting Power"
- [3] The coach, first-person 'I', stating which setup he himself would play. (resolution: context_resolved)
  - cite[4893:4957] byte-exact: 'I would always play First Strike but but I would play with flash'

#### reference_bindings (4 items)
- [1] Implied subject of 'play with flash' binds to the coach's first-person 'I' carried over from the immediately preceding clause 'but but I would'. (resolution: context_resolved)
  - cite[4926:4957] byte-exact: 'but but I would play with flash'
- [2] 'you' across the passage ('you're going', 'you have', 'you're losing', 'you probably have') binds to the coached player. (resolution: context_resolved)
  - cite[4971:5060] byte-exact: "you're going exhaust SM but you have less fighting Power because you're losing Last Stand"
- [3] 'yeah I do' binds to the coached player, confirming that he has Ultimate Hunter and eyeball. (resolution: context_resolved)
  - cite[5116:5179] byte-exact: 'you probably have ultimate Hunter eyeball right yeah I do right'
- [4] 'this' in 'that doesn't mean this is bad' binds to the player's current rune/spell setup under discussion, inferred from the continuation 'like I would go the same thing except for this one'. (resolution: context_resolved)
  - cite[5180:5260] byte-exact: "that doesn't mean this is bad like I would go the same thing except for this one"

#### abilities_resources (8 items)
- [1] Flash (summoner spell): the coach says he would play with Flash. (resolution: literal_explicit)
  - cite[4926:4994] byte-exact: "but but I would play with flash because like you're going exhaust SM"
- [2] Exhaust (summoner spell): currently equipped by the player's Karthus. (resolution: literal_explicit)
  - cite[4971:5027] byte-exact: "you're going exhaust SM but you have less fighting Power"
- [3] Smite (summoner spell): surface token 'SM' repaired to Smite using the supplied summoner-spell vocabulary and adjacent transcript support. (resolution: vocabulary_supported)
  - cite[4971:4994] byte-exact: "you're going exhaust SM"
  - cite[4492:4553] byte-exact: 'for carus you are never allowed to go Smite and exhaust again'
- [4] Last Stand (rune): not present on the player's page under the current setup ('losing' it). (resolution: literal_explicit)
  - cite[5036:5060] byte-exact: "you're losing Last Stand"
- [5] Dark Harvest (rune): referenced as 'Harvest'; bound via adjacent statement 'you should at least go dark Harvest'. (resolution: context_resolved)
  - cite[5061:5082] byte-exact: "you're losing Harvest"
  - cite[4676:4711] byte-exact: 'you should at least go dark Harvest'
- [6] Cheap Shot (rune): not present on the player's page under the current setup ('losing' it). (resolution: literal_explicit)
  - cite[5083:5107] byte-exact: "you're losing cheap shot"
- [7] Ultimate Hunter (rune): on the player's page, first hedged with 'probably' then confirmed by the player. (resolution: literal_explicit)
  - cite[5108:5179] byte-exact: 'because you probably have ultimate Hunter eyeball right yeah I do right'
- [8] 'eyeball' (rune): literal surface token only; canonical full name not recoverable from supplied material. (resolution: literal_explicit)
  - cite[5134:5163] byte-exact: 'ultimate Hunter eyeball right'

#### events_actions (0 items)

(none)

#### states (4 items)
- [1] The player's Karthus is running Exhaust plus Smite as summoner spells (i.e., without Flash). (resolution: literal_explicit)
  - cite[4971:5027] byte-exact: "you're going exhaust SM but you have less fighting Power"
  - cite[4492:4553] byte-exact: 'for carus you are never allowed to go Smite and exhaust again'
- [2] The player's rune page includes Ultimate Hunter and eyeball; the player verbally confirms this. (resolution: literal_explicit)
  - cite[5108:5179] byte-exact: 'because you probably have ultimate Hunter eyeball right yeah I do right'
- [3] The player's page is missing Last Stand, Harvest (Dark Harvest), and Cheap Shot, described as 'losing' them. (resolution: literal_explicit)
  - cite[5036:5107] byte-exact: "you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [4] Resulting state: the player has less fighting power under the current setup. (resolution: literal_explicit)
  - cite[4995:5027] byte-exact: 'but you have less fighting Power'

#### conditions (1 items)
- [1] Conditional frame: while the player goes Exhaust (plus Smite) with this rune page, his fighting power is lower than when playing with Flash. (resolution: context_resolved)
  - cite[4926:5027] byte-exact: "but but I would play with flash because like you're going exhaust SM but you have less fighting Power"

#### recommended_advice (3 items)
- [1] Play Karthus with Flash. (resolution: literal_explicit)
  - cite[4926:4994] byte-exact: "but but I would play with flash because like you're going exhaust SM"
- [2] Coach's own stated plan: always play First Strike, but paired with Flash rather than the current Exhaust-based spell setup. (resolution: literal_explicit)
  - cite[4893:4957] byte-exact: 'I would always play First Strike but but I would play with flash'
- [3] Implied by contrast: prefer Flash over the current Exhaust (plus Smite) pairing, which leaves less fighting power. (resolution: context_resolved)
  - cite[4942:5027] byte-exact: "play with flash because like you're going exhaust SM but you have less fighting Power"

#### consequences_outcomes (4 items)
- [1] Going Exhaust (plus Smite) results in the player having less fighting power. (resolution: literal_explicit)
  - cite[4971:5027] byte-exact: "you're going exhaust SM but you have less fighting Power"
- [2] Losing Last Stand, Harvest, and Cheap Shot results in less fighting power. (resolution: literal_explicit)
  - cite[4999:5107] byte-exact: "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [3] Having Ultimate Hunter and eyeball causes the loss of the Last Stand / Harvest / Cheap Shot slots. (resolution: literal_explicit)
  - cite[5083:5157] byte-exact: "you're losing cheap shot because you probably have ultimate Hunter eyeball"
- [4] Despite the reduced fighting power, the coach states the current setup is not bad. (resolution: literal_explicit)
  - cite[5180:5222] byte-exact: "that doesn't mean this is bad like I would"

#### explicit_relationships (11 items)
- [1] Player's Karthus uses the Exhaust summoner spell. (resolution: literal_explicit; relation: USES)
  - cite[4971:5027] byte-exact: "you're going exhaust SM but you have less fighting Power"
- [2] Player's Karthus uses Smite, recovered from surface token 'SM'. (resolution: vocabulary_supported; relation: USES)
  - cite[4971:4994] byte-exact: "you're going exhaust SM"
  - cite[4492:4553] byte-exact: 'for carus you are never allowed to go Smite and exhaust again'
- [3] Player's rune page includes Ultimate Hunter and eyeball. (resolution: literal_explicit; relation: USES)
  - cite[5108:5179] byte-exact: 'because you probably have ultimate Hunter eyeball right yeah I do right'
- [4] Going Exhaust (plus Smite) causes the player to have less fighting power. (resolution: literal_explicit; relation: CAUSES)
  - cite[4971:5027] byte-exact: "you're going exhaust SM but you have less fighting Power"
- [5] Losing Last Stand, Harvest, and Cheap Shot causes the reduced fighting power. (resolution: literal_explicit; relation: CAUSES)
  - cite[4999:5107] byte-exact: "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [6] Having Ultimate Hunter and eyeball causes the loss of Last Stand, Harvest, and Cheap Shot. (resolution: literal_explicit; relation: CAUSES)
  - cite[5036:5157] byte-exact: "you're losing Last Stand you're losing Harvest you're losing cheap shot because you probably have ultimate Hunter eye..."
- [7] Condition: with the Exhaust (plus Smite) setup in effect, fighting power is lower than with Flash. (resolution: context_resolved; relation: CONDITION)
  - cite[4926:5027] byte-exact: "but but I would play with flash because like you're going exhaust SM but you have less fighting Power"
- [8] The implied subject 'I' of 'play with flash' refers to the coach. (resolution: context_resolved; relation: REFERS_TO)
  - cite[4926:4957] byte-exact: 'but but I would play with flash'
- [9] 'yeah I do' refers to the coached player possessing Ultimate Hunter and eyeball. (resolution: context_resolved; relation: REFERS_TO)
  - cite[5116:5179] byte-exact: 'you probably have ultimate Hunter eyeball right yeah I do right'
- [10] 'this' in 'that doesn't mean this is bad' refers to the current setup under discussion. (resolution: context_resolved; relation: REFERS_TO)
  - cite[5180:5260] byte-exact: "that doesn't mean this is bad like I would go the same thing except for this one"
- [11] The coach is the actor of the recommendation 'play with flash'. (resolution: context_resolved; relation: ACTOR)
  - cite[4893:4957] byte-exact: 'I would always play First Strike but but I would play with flash'

#### uncertainty_unresolved (4 items)
- [1] Surface token 'SM' is ASR-corrupted; repaired to 'Smite' using the supplied summoner-spell vocabulary and the adjacent statement about Smite plus Exhaust. The byte-level span remains corrupted in the source. (resolution: vocabulary_supported)
  - cite[4971:4994] byte-exact: "you're going exhaust SM"
  - cite[4492:4553] byte-exact: 'for carus you are never allowed to go Smite and exhaust again'
- [2] 'Harvest' is shorthand; bound to Dark Harvest only via the adjacent utterance 'you should at least go dark Harvest'; no other recovery evidence is supplied. (resolution: context_resolved)
  - cite[5061:5082] byte-exact: "you're losing Harvest"
  - cite[4676:4711] byte-exact: 'you should at least go dark Harvest'
- [3] Referent of 'this' in 'that doesn't mean this is bad' is inferred as the player's current rune/spell setup; narrower readings (e.g., only the Ultimate Hunter + eyeball picks) cannot be excluded from the supplied material. (resolution: context_resolved)
  - cite[5180:5260] byte-exact: "that doesn't mean this is bad like I would go the same thing except for this one"
- [4] Canonical identity of the rune surfaced as 'eyeball' cannot be recovered from the supplied vocabulary or transcript; only the literal token is preserved and no expansion is asserted. (resolution: unresolved)
  - cite[5134:5163] byte-exact: 'ultimate Hunter eyeball right'

#### supporting_source_spans (5 items)
- [1] Primary span covering the full target passage. (resolution: literal_explicit)
  - cite[4942:5222] byte-exact: "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing Last Stan..."
  - span [4942:5222]
- [2] Immediately preceding clause supplying the subject 'I' (coach) and the First Strike context for the target passage. (resolution: literal_explicit)
  - cite[4893:4957] byte-exact: 'I would always play First Strike but but I would play with flash'
  - span [4893:4957]
- [3] Adjacent span fixing the champion (Karthus/'carus') and the Smite plus Exhaust pairing behind the target passage's 'exhaust SM'. (resolution: literal_explicit)
  - cite[4492:4553] byte-exact: 'for carus you are never allowed to go Smite and exhaust again'
  - span [4492:4553]
- [4] Adjacent span supplying the 'dark Harvest' naming that grounds the target passage's 'Harvest'. (resolution: literal_explicit)
  - cite[4676:4711] byte-exact: 'you should at least go dark Harvest'
  - span [4676:4711]
- [5] Following continuation clarifying the 'same thing except for this one' comparison that scopes the passage's closing 'this is bad' remark. (resolution: literal_explicit)
  - cite[5223:5276] byte-exact: 'go the same thing except for this one I would go this'
  - span [5223:5276]

Raw response binding A: `557fce70554f73ea9ee59920fc6f98d56b3124d10d01c154f90c60974e319657`
Raw response binding B: `030a3f9feff18ad3a57161d677d25e146cee52de06fc2182e5f2ca3afd1c7b7d`

---

## TARGET p2k:case:0002 (selection rank 2)

### Exact Bronze

```text
look at atrox atrox has 11 CS after this if at he's giving a freeb right because this if yeah if atrox T is back here do you really think atrox with clo armor full HP will beat this darus probably not right because I don't think so
```

Location in full transcript: [30531:30762]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (3 items)
- [1] Aatrox (surface form 'atrox', resolved via supplied alias rule) (resolution: vocabulary_supported)
  - cite[0:13] byte-exact: 'look at atrox'
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
- [2] Darius (surface form 'darus', metadata-licensed alias; champion metadata contains Darius) (resolution: vocabulary_supported)
  - cite[167:200] byte-exact: 'will beat this darus probably not'
- [3] 'you', the listener the coach addresses with the rhetorical fight question (resolution: context_resolved)
  - cite[118:137] byte-exact: 'do you really think'

#### reference_bindings (6 items)
- [1] 'he' in "he's giving a freeb" binds to Aatrox ('atrox'), the champion under discussion (resolution: context_resolved)
  - cite[41:66] byte-exact: "if at he's giving a freeb"
- [2] 'this' in "11 CS after this" refers to an unspecified prior event or moment; not resolvable from the supplied passage (resolution: unresolved)
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
- [3] 'T' in "atrox T is back here" cannot be bound from supplied material; possibly part of Aatrox's name phrase or an ASR-corrupted token (resolution: unresolved)
  - cite[94:117] byte-exact: 'if atrox T is back here'
- [4] 'here' in "is back here" is a deictic location whose exact map position is unresolved (resolution: unresolved)
  - cite[105:117] byte-exact: 'is back here'
- [5] 'this' in "will beat this darus" binds to Darius ('darus') present with the speaker (resolution: context_resolved)
  - cite[167:200] byte-exact: 'will beat this darus probably not'
- [6] 'at' in "if at he's" is a garbled token, plausibly a false start of 'atrox'; unresolved (resolution: unresolved)
  - cite[41:66] byte-exact: "if at he's giving a freeb"

#### abilities_resources (4 items)
- [1] 11 CS held by Aatrox (resolution: literal_explicit)
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
- [2] 'clo armor', an item worn by Aatrox; likely ASR corruption of Cloth Armor but not repairable via the supplied vocabulary (resolution: unresolved)
  - cite[138:166] byte-exact: 'atrox with clo armor full HP'
- [3] 'T' in "atrox T", possibly the summoner spell Teleport or ASR corruption; unconfirmed (resolution: unresolved)
  - cite[94:117] byte-exact: 'if atrox T is back here'
- [4] 'freeb', something Aatrox is giving away for free; term not recoverable from supplied vocabulary (resolution: unresolved)
  - cite[47:66] byte-exact: "he's giving a freeb"

#### events_actions (2 items)
- [1] Aatrox has reached 11 CS at this point in the game (resolution: literal_explicit)
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
- [2] Aatrox is giving away something described as a 'freeb' (likely conceding value/kill), per the coach's remark (resolution: context_resolved)
  - cite[47:66] byte-exact: "he's giving a freeb"

#### states (3 items)
- [1] Aatrox holds 11 CS (state holding after the referenced point) (resolution: literal_explicit)
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
- [2] In the hypothetical, Aatrox has 'clo armor' and is at full HP (resolution: unresolved)
  - cite[138:166] byte-exact: 'atrox with clo armor full HP'
- [3] Hypothetical state: Aatrox (possibly with 'T') is positioned back where the speaker indicates ('back here') (resolution: unresolved)
  - cite[94:117] byte-exact: 'if atrox T is back here'

#### conditions (3 items)
- [1] Condition on the 11-CS count: it holds 'after this', an unspecified referenced event (resolution: unresolved)
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
- [2] Condition: if Aatrox (with 'T') is back here, then the posed fight question applies (resolution: literal_explicit)
  - cite[94:117] byte-exact: 'if atrox T is back here'
- [3] Condition framing the outcome question: Aatrox fighting with 'clo armor' and full HP against this Darius (resolution: literal_explicit)
  - cite[138:166] byte-exact: 'atrox with clo armor full HP'

#### recommended_advice (1 items)
- [1] Coach directs the student to look at Aatrox (resolution: literal_explicit)
  - cite[0:13] byte-exact: 'look at atrox'

#### consequences_outcomes (2 items)
- [1] Implied outcome: even with 'clo armor' and full HP, Aatrox would probably NOT beat this Darus/Darius in a fight (resolution: literal_explicit)
  - cite[167:200] byte-exact: 'will beat this darus probably not'
  - cite[207:231] byte-exact: "because I don't think so"
- [2] Implied consequence: Aatrox giving a 'freeb' means conceding free value to the opponent (exact nature unresolved) (resolution: context_resolved)
  - cite[47:80] byte-exact: "he's giving a freeb right because"

#### explicit_relationships (10 items)
- [1] Aatrox is the actor of the event 'has 11 CS after this' (resolution: literal_explicit; relation: ACTOR)
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
- [2] Aatrox is the actor of 'giving a freeb' (resolution: context_resolved; relation: ACTOR)
  - cite[41:66] byte-exact: "if at he's giving a freeb"
- [3] Darius is the target of Aatrox's hypothetical attempt to 'beat' him (resolution: vocabulary_supported; relation: TARGET)
  - cite[167:200] byte-exact: 'will beat this darus probably not'
- [4] The condition "if atrox T is back here" gates the posed fight-outcome question (resolution: literal_explicit; relation: CONDITION)
  - cite[94:117] byte-exact: 'if atrox T is back here'
- [5] Aatrox's hypothetical state (clo armor, full HP) is the condition under which the 'beat Darius' question is evaluated (resolution: literal_explicit; relation: CONDITION)
  - cite[138:166] byte-exact: 'atrox with clo armor full HP'
- [6] The expected result of the hypothetical fight is that Aatrox does not win ('probably not') (resolution: literal_explicit; relation: RESULT)
  - cite[167:200] byte-exact: 'will beat this darus probably not'
- [7] 'he' REFERS_TO Aatrox (resolution: context_resolved; relation: REFERS_TO)
  - cite[41:66] byte-exact: "if at he's giving a freeb"
- [8] The 11-CS state occurs AFTER an unspecified referent event ('this') (resolution: unresolved; relation: AFTER)
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
- [9] 'darus' REFERS_TO Darius (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[167:200] byte-exact: 'will beat this darus probably not'
- [10] Aatrox's presumed low CS and 'freeb' behavior are presented by the coach as cause for doubt about Aatrox beating Darius ('because') (resolution: unresolved; relation: CAUSES)
  - cite[67:85] byte-exact: 'right because this'
  - cite[207:231] byte-exact: "because I don't think so"

#### uncertainty_unresolved (5 items)
- [1] 'at' in "if at he's" is garbled speech, plausibly a false start of 'atrox'; unrecoverable from supplied material (resolution: unresolved)
  - cite[41:51] byte-exact: "if at he's"
- [2] 'T' in "atrox T" cannot be recovered; could be Teleport or corruption, unsupported by vocabulary (resolution: unresolved)
  - cite[97:104] byte-exact: 'atrox T'
- [3] 'clo armor' is probable ASR corruption of Cloth Armor; no supplied vocabulary support, so left unresolved (resolution: unresolved)
  - cite[149:158] byte-exact: 'clo armor'
- [4] 'freeb' is not recoverable; possibly 'freebie'/free kill, unsupported (resolution: unresolved)
  - cite[59:66] byte-exact: 'a freeb'
- [5] 'this' (referent of 'after this') and the deictic location 'here' are unresolved without surrounding transcript (resolution: unresolved)
  - cite[30:40] byte-exact: 'after this'
  - cite[105:117] byte-exact: 'is back here'

#### supporting_source_spans (4 items)
- [1] Advice to look at Aatrox plus his current CS state (resolution: literal_explicit)
  - cite[0:13] byte-exact: 'look at atrox'
  - cite[14:40] byte-exact: 'atrox has 11 CS after this'
  - span [0:40]
- [2] Conditional positioning premise about Aatrox returning (resolution: literal_explicit)
  - cite[94:117] byte-exact: 'if atrox T is back here'
  - span [94:117]
- [3] Fight question, Aatrox's hypothetical build/state, expected non-win answer, and coach's reasoning close (resolution: literal_explicit)
  - cite[118:200] byte-exact: 'do you really think atrox with clo armor full HP will beat this darus probably not'
  - cite[207:231] byte-exact: "because I don't think so"
  - span [118:231]
- [4] Freebie concession remark supporting the 'giving away value' event (resolution: context_resolved)
  - cite[47:80] byte-exact: "he's giving a freeb right because"
  - span [47:80]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (4 items)
- [1] Aatrox, the enemy top laner, referred to by surface forms 'atrox' throughout the target passage. (resolution: vocabulary_supported)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
  - cite[30625:30731] byte-exact: 'if atrox T is back here do you really think atrox with clo armor full HP will beat this darus probably not'
- [2] Darius, the champion of the reviewed rank-one Darius player, referred to by surface form 'this darus'. (resolution: vocabulary_supported)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'
- [3] The coaching speaker, self-reference 'I'. (resolution: context_resolved)
  - cite[30719:30762] byte-exact: "probably not right because I don't think so"
- [4] The student being coached, addressed as 'you'. (resolution: context_resolved)
  - cite[30649:30731] byte-exact: 'do you really think atrox with clo armor full HP will beat this darus probably not'

#### reference_bindings (7 items)
- [1] 'atrox' (all mentions in the target passage) binds to Aatrox via the supplied alias rule atrox->Aatrox. (resolution: vocabulary_supported)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
  - cite[30625:30648] byte-exact: 'if atrox T is back here'
- [2] 'this darus' binds to Darius, the champion played by the reviewed player, via the metadata-licensed alias darus->Darius. (resolution: vocabulary_supported)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'
- [3] 'he' in 'he's giving a freeb' cannot be securely bound from the supplied material; candidate binders include the reviewed Darius player conceding a free base or Aatrox gaining one. (resolution: unresolved)
  - cite[30572:30603] byte-exact: "if at he's giving a freeb right"
- [4] 'T' in 'atrox T is back here' plausibly denotes Aatrox's Teleport summoner spell, supported by nearby talk ('atro doesn't have teleport'; 'even if he T is back'), but the surface form alone does not confirm it. (resolution: unresolved)
  - cite[30625:30648] byte-exact: 'if atrox T is back here'
  - cite[31329:31401] byte-exact: "even if he T is back with CLW armor fullish spe he can't beat this darus"
  - cite[31205:31284] byte-exact: "so it's only correct to set up the slow push because atro doesn't have teleport"
- [5] 'you' in 'do you really think' binds to the student being coached. (resolution: context_resolved)
  - cite[30649:30731] byte-exact: 'do you really think atrox with clo armor full HP will beat this darus probably not'
- [6] 'I' in 'because I don't think so' binds to the coaching speaker. (resolution: context_resolved)
  - cite[30719:30762] byte-exact: "probably not right because I don't think so"
- [7] 'here' in 'is back here' plausibly binds to the top-lane wave/fight position currently shown in the replay. (resolution: context_resolved)
  - cite[30625:30648] byte-exact: 'if atrox T is back here'

#### abilities_resources (4 items)
- [1] Aatrox's creep-score resource: 11 CS. (resolution: literal_explicit)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
- [2] Cloth armor item on Aatrox in the hypothetical; surface form 'clo armor' is ASR-corrupted spelling recovered from context. (resolution: context_resolved)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'
  - cite[34786:34817] byte-exact: 'whether he has a cloth or sword'
- [3] Full health (full HP) on Aatrox in the hypothetical. (resolution: literal_explicit)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'
- [4] Candidate Teleport summoner spell for Aatrox surfaced only as 'T'; ownership attribution to Aatrox is tentative. (resolution: unresolved)
  - cite[30625:30648] byte-exact: 'if atrox T is back here'
  - cite[31329:31401] byte-exact: "even if he T is back with CLW armor fullish spe he can't beat this darus"
  - cite[31205:31284] byte-exact: "so it's only correct to set up the slow push because atro doesn't have teleport"

#### events_actions (0 items)

(none)

#### states (2 items)
- [1] Aatrox has 11 CS after this point in the game. (resolution: literal_explicit)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
- [2] Hypothetical frame state: Aatrox holds cloth armor and full HP while facing this Darius. (resolution: literal_explicit)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'

#### conditions (3 items)
- [1] Temporal condition 'after this': the 11-CS count is evaluated after the just-described action. (resolution: literal_explicit)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
- [2] Conditional 'if atrox T is back here': the outcome question is posed under Aatrox('s T) returning to the position. (resolution: literal_explicit)
  - cite[30625:30648] byte-exact: 'if atrox T is back here'
- [3] Matchup condition: Aatrox with cloth armor and full HP versus this Darius. (resolution: literal_explicit)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'

#### recommended_advice (0 items)

(none)

#### consequences_outcomes (1 items)
- [1] Assessed outcome: under the stated conditions (T back, cloth armor, full HP), Aatrox would probably not beat this Darius. (resolution: literal_explicit)
  - cite[30625:30731] byte-exact: 'if atrox T is back here do you really think atrox with clo armor full HP will beat this darus probably not'
  - cite[30719:30762] byte-exact: "probably not right because I don't think so"

#### explicit_relationships (9 items)
- [1] In the hypothetical fight, Aatrox ('atrox') is the acting party attempting to beat Darius. (resolution: literal_explicit; relation: ACTOR)
  - cite[30625:30731] byte-exact: 'if atrox T is back here do you really think atrox with clo armor full HP will beat this darus probably not'
- [2] 'this darus' is the target of Aatrox's hypothetical attempt to beat. (resolution: literal_explicit; relation: TARGET)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'
- [3] Aatrox would fight using cloth armor (item) with full HP. (resolution: literal_explicit; relation: USES)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'
- [4] 'if atrox T is back here' is the condition governing the outcome assessment. (resolution: literal_explicit; relation: CONDITION)
  - cite[30625:30648] byte-exact: 'if atrox T is back here'
- [5] Assessed result of the hypothetical fight: Aatrox does not beat this Darius ('probably not'). (resolution: literal_explicit; relation: RESULT)
  - cite[30719:30762] byte-exact: "probably not right because I don't think so"
- [6] The speaker negates the proposition that cloth-armor, full-HP Aatrox beats this Darius. (resolution: literal_explicit; relation: NEGATES)
  - cite[30719:30762] byte-exact: "probably not right because I don't think so"
- [7] 'atrox' refers to Aatrox. (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
- [8] 'darus' refers to Darius. (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'
- [9] The 11-CS state is anchored 'after this', following the reviewed Darius player's push-for-double-long-sword-base action described immediately before the passage. (resolution: context_resolved; relation: AFTER)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
  - cite[30345:30381] byte-exact: 'he pushes for double long sword base'

#### uncertainty_unresolved (6 items)
- [1] 'freeb' is garbled ASR; plausible readings include 'free B' (free recall/base). The giver ('he') and beneficiary are unresolved, so no event or consequence is asserted from it. (resolution: unresolved)
  - cite[30572:30603] byte-exact: "if at he's giving a freeb right"
  - cite[30578:30597] byte-exact: "he's giving a freeb"
- [2] 'T' in 'atrox T is back here' is likely Teleport but is not recoverable with certainty from the surface form; recorded unresolved despite contextual support. (resolution: unresolved)
  - cite[30625:30648] byte-exact: 'if atrox T is back here'
- [3] 'at' in 'if at' appears truncated (plausibly 'atrox'); binding remains unresolved. (resolution: unresolved)
  - cite[30572:30603] byte-exact: "if at he's giving a freeb right"
- [4] Disfluent segment 'because this if yeah' resists clean semantic segmentation and is treated as filler. (resolution: unresolved)
  - cite[30604:30624] byte-exact: 'because this if yeah'
- [5] 'clo armor' is ASR-corrupted spelling of 'cloth armor', recovered via later context; residual orthographic uncertainty is noted. (resolution: context_resolved)
  - cite[30675:30718] byte-exact: 'with clo armor full HP will beat this darus'
  - cite[34786:34817] byte-exact: 'whether he has a cloth or sword'
- [6] Antecedent of 'after this' lies just outside the target passage (the push-for-double-long-sword-base description); bound contextually. (resolution: context_resolved)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
  - cite[30345:30381] byte-exact: 'he pushes for double long sword base'

#### supporting_source_spans (3 items)
- [1] Opening span of the target passage: address to look at Aatrox and his 11 CS after this point. (resolution: literal_explicit)
  - cite[30531:30571] byte-exact: 'look at atrox atrox has 11 CS after this'
  - span [30531:30571]
- [2] Middle span of the target passage: garbled 'freeb' clause. (resolution: unresolved)
  - cite[30572:30603] byte-exact: "if at he's giving a freeb right"
  - span [30572:30603]
- [3] Closing span of the target passage: conditional question of Aatrox (T back, cloth armor, full HP) beating Darius and the speaker's negative assessment. (resolution: literal_explicit)
  - cite[30604:30762] byte-exact: 'because this if yeah if atrox T is back here do you really think atrox with clo armor full HP will beat this darus pr...'
  - span [30604:30762]

Raw response binding A: `0c27f545ab73c6cf03329144db3e77cc4662fcbfba497e7fb408e155534c8b1b`
Raw response binding B: `545e27ac1f60afe5dbd72fb204133c4c0c77d1e719b5d601ebef7fae0038f235`

---

## TARGET p2k:case:0003 (selection rank 3)

### Exact Bronze

```text
I mean I would always go ignite versus Fiora because she heals more than [ __ ] window but I think flash is broken on Camille So I would go flash TP in most cases yeah but I think ignite when you need heal cut is fine because
```

Location in full transcript: [13315:13540]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (3 items)
- [1] Fiora (champion, the opposing matchup referenced) (resolution: literal_explicit)
  - cite[32:44] byte-exact: 'versus Fiora'
- [2] Camille (champion being played; metadata role top) (resolution: literal_explicit)
  - cite[115:125] byte-exact: 'on Camille'
- [3] The speaker ('I'), identity (coach or student) not determinable from the supplied passage alone (resolution: context_resolved)
  - cite[0:6] byte-exact: 'I mean'

#### reference_bindings (7 items)
- [1] 'she' (in 'because she heals more than') binds to Fiora (resolution: context_resolved)
  - cite[45:72] byte-exact: 'because she heals more than'
- [2] 'you' (in 'when you need heal cut') is generic second-person address referring to the player deciding whether to take Ignite into this matchup (resolution: context_resolved)
  - cite[187:209] byte-exact: 'when you need heal cut'
- [3] 'I' (token 1, 'I mean') binds to the speaker (resolution: context_resolved)
  - cite[0:6] byte-exact: 'I mean'
- [4] 'I' (token 2, 'I would always go ignite') binds to the speaker (resolution: context_resolved)
  - cite[7:31] byte-exact: 'I would always go ignite'
- [5] 'I' (token 3, 'I think flash is broken on Camille') binds to the speaker (resolution: context_resolved)
  - cite[87:125] byte-exact: 'but I think flash is broken on Camille'
- [6] 'I' (token 4, 'I would go flash TP') binds to the speaker (resolution: context_resolved)
  - cite[129:148] byte-exact: 'I would go flash TP'
- [7] 'I' (token 5, 'I think ignite ... is fine') binds to the speaker (resolution: context_resolved)
  - cite[168:217] byte-exact: 'but I think ignite when you need heal cut is fine'

#### abilities_resources (5 items)
- [1] Ignite (summoner spell) (resolution: literal_explicit)
  - cite[22:44] byte-exact: 'go ignite versus Fiora'
- [2] Flash (summoner spell), mentioned twice ('flash is broken on Camille' and 'go flash TP') (resolution: literal_explicit)
  - cite[99:125] byte-exact: 'flash is broken on Camille'
  - cite[137:148] byte-exact: 'go flash TP'
- [3] 'TP' refers to the Teleport summoner spell (abbreviation expansion supported by supplied summoner-spell vocabulary) (resolution: vocabulary_supported)
  - cite[140:148] byte-exact: 'flash TP'
- [4] 'heal cut' (healing reduction) cited by the speaker as the condition under which taking Ignite is acceptable (resolution: literal_explicit)
  - cite[180:217] byte-exact: 'ignite when you need heal cut is fine'
- [5] Fiora's healing (referred to via 'she heals'), owned by Fiora through the resolved pronoun binding (resolution: context_resolved)
  - cite[45:72] byte-exact: 'because she heals more than'

#### events_actions (0 items)

(none)

#### states (1 items)
- [1] Presupposed game situation for the advice: the player is playing Camille into Fiora (matchup state governing summoner spell choice) (resolution: context_resolved)
  - cite[15:44] byte-exact: 'always go ignite versus Fiora'

#### conditions (3 items)
- [1] Condition for the default Ignite choice: facing/playing versus Fiora (resolution: literal_explicit)
  - cite[15:44] byte-exact: 'always go ignite versus Fiora'
- [2] Condition under which Ignite is stated to be fine: when you need heal cut (resolution: literal_explicit)
  - cite[180:217] byte-exact: 'ignite when you need heal cut is fine'
- [3] Qualifier on the Flash + Teleport recommendation: it applies only 'in most cases', not universally (resolution: literal_explicit)
  - cite[129:162] byte-exact: 'I would go flash TP in most cases'

#### recommended_advice (3 items)
- [1] Take Ignite versus Fiora (stated as what the speaker would 'always' do in that matchup) (resolution: literal_explicit)
  - cite[0:44] byte-exact: 'I mean I would always go ignite versus Fiora'
- [2] Default to Flash plus Teleport on Camille in most cases, justified by the claim that flash is broken on Camille (resolution: literal_explicit)
  - cite[87:162] byte-exact: 'but I think flash is broken on Camille So I would go flash TP in most cases'
- [3] Taking Ignite is acceptable/fine when you need heal cut (resolution: literal_explicit)
  - cite[180:217] byte-exact: 'ignite when you need heal cut is fine'

#### consequences_outcomes (1 items)
- [1] Fiora heals more than the comparison referent (referent corrupted as '[ __ ] window'); this high healing is given as the reason for going Ignite versus her (resolution: unresolved)
  - cite[45:86] byte-exact: 'because she heals more than [\xa0__\xa0] window'

#### explicit_relationships (7 items)
- [1] The recommendation to take Ignite holds under the condition of the matchup versus Fiora (advice conditioned on opponent) (resolution: literal_explicit; relation: CONDITION)
  - cite[22:44] byte-exact: 'go ignite versus Fiora'
- [2] Fiora's high healing causes/motivates the preference for Ignite versus her ('because she heals more than ...') (resolution: context_resolved; relation: CAUSES)
  - cite[45:72] byte-exact: 'because she heals more than'
- [3] Pronoun 'she' refers to Fiora (resolution: context_resolved; relation: REFERS_TO)
  - cite[45:72] byte-exact: 'because she heals more than'
- [4] The Camille player uses Flash and Teleport as summoner spells (spell set bound to Camille via 'broken on Camille') (resolution: context_resolved; relation: USES)
  - cite[137:148] byte-exact: 'go flash TP'
  - cite[115:125] byte-exact: 'on Camille'
- [5] The statement that Ignite is fine holds under the condition 'when you need heal cut' (resolution: literal_explicit; relation: CONDITION)
  - cite[180:217] byte-exact: 'ignite when you need heal cut is fine'
- [6] Heal cut (the effect sought from Ignite) counters/negates Fiora's healing (resolution: context_resolved; relation: NEGATES)
  - cite[45:86] byte-exact: 'because she heals more than [\xa0__\xa0] window'
  - cite[187:209] byte-exact: 'when you need heal cut'
- [7] Abbreviation 'TP' refers to the Teleport summoner spell (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[140:148] byte-exact: 'flash TP'

#### uncertainty_unresolved (3 items)
- [1] The censored/corrupted span '[ __ ]' inside 'she heals more than [ __ ] window' cannot be recovered from the supplied material; the comparator in Fiora's healing comparison is unknown (possibly an ASR artifact or bleeped token before 'window') (resolution: unresolved)
  - cite[53:86] byte-exact: 'she heals more than [\xa0__\xa0] window'
- [2] The passage ends mid-sentence after a trailing 'because'; the reason the speaker was about to give for Ignite being fine is missing/unrecoverable (resolution: unresolved)
  - cite[210:225] byte-exact: 'is fine because'
- [3] The first-person speaker ('I') cannot be determined to be the coach or the student from the supplied material; bindings are recorded only at the level of 'the speaker' (resolution: unresolved)
  - cite[0:44] byte-exact: 'I mean I would always go ignite versus Fiora'

#### supporting_source_spans (3 items)
- [1] Primary span covering the Ignite-versus-Fiora recommendation and its healing rationale (resolution: literal_explicit)
  - cite[0:86] byte-exact: 'I mean I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window'
  - span [0:86]
- [2] Span covering the Flash-is-broken-on-Camille justification and the Flash+TP default (resolution: literal_explicit)
  - cite[87:162] byte-exact: 'but I think flash is broken on Camille So I would go flash TP in most cases'
  - span [87:162]
- [3] Span covering the conditional acceptance of Ignite when heal cut is needed and the truncated closing clause (resolution: literal_explicit)
  - cite[168:225] byte-exact: 'but I think ignite when you need heal cut is fine because'
  - span [168:225]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (3 items)
- [1] Fiora, the enemy champion matchup the summoner-spell recommendation is made against. (resolution: literal_explicit)
  - cite[13340:13359] byte-exact: 'ignite versus Fiora'
- [2] Camille, the coach's own champion that the Flash assessment and default spell plan are about. (resolution: literal_explicit)
  - cite[13423:13440] byte-exact: 'broken on Camille'
- [3] The coach (speaker 'I') giving the summoner-spell recommendations in the target passage. (resolution: context_resolved)
  - cite[13315:13359] byte-exact: 'I mean I would always go ignite versus Fiora'

#### reference_bindings (4 items)
- [1] 'she' in 'because she heals more than...' binds to Fiora, named immediately before in 'ignite versus Fiora'. (resolution: context_resolved)
  - cite[13360:13401] byte-exact: 'because she heals more than [\xa0__\xa0] window'
- [2] 'I' throughout the passage binds to the coach/speaker making the spell recommendations. (resolution: context_resolved)
  - cite[13315:13359] byte-exact: 'I mean I would always go ignite versus Fiora'
- [3] 'you' in 'when you need heal cut' is the generic addressed player (the coached student) deciding summoner spells. (resolution: context_resolved)
  - cite[13478:13540] byte-exact: 'yeah but I think ignite when you need heal cut is fine because'
- [4] '[ __ ] window': censored token followed by suspect ASR output; the thing Fiora supposedly heals more than cannot be recovered or bound from the supplied material. (resolution: unresolved)
  - cite[13368:13401] byte-exact: 'she heals more than [\xa0__\xa0] window'

#### abilities_resources (7 items)
- [1] Ignite, summoner spell recommended always versus Fiora. (resolution: literal_explicit)
  - cite[13337:13359] byte-exact: 'go ignite versus Fiora'
- [2] Flash, summoner spell assessed as broken (overpowered) on Camille. (resolution: literal_explicit)
  - cite[13402:13440] byte-exact: 'but I think flash is broken on Camille'
- [3] Teleport (TP), summoner spell paired with Flash as the default setup. (resolution: literal_explicit)
  - cite[13441:13477] byte-exact: 'So I would go flash TP in most cases'
- [4] Heal cut, i.e. a healing-reduction effect, cited as the situation in which Ignite is fine; provided by Ignite rather than items on Camille. (resolution: context_resolved)
  - cite[13478:13540] byte-exact: 'yeah but I think ignite when you need heal cut is fine because'
- [5] Executioner's (heal-cut item), named in the continuation of the truncated 'because' clause as unbuyable on Camille. (resolution: context_resolved)
  - cite[13541:13583] byte-exact: "you can't buy executioners on this channel"
- [6] Trinity and Hydra, core items Camille needs, which crowd out buying a heal-cut item. (resolution: literal_explicit)
  - cite[13589:13712] byte-exact: "you need other items too much you need Trinity and you need Hydra too much so you can't buy heal cut like a normal Ch..."
- [7] 'e flash', the E-ability-plus-Flash combo described as too OP; the E belongs to Camille per the surrounding summoner-spell discussion (E slot = Hookshot per supplied vocabulary). (resolution: context_resolved)
  - cite[13749:13793] byte-exact: 'otherwise always flash cuz e flash is too op'

#### events_actions (0 items)

(none)

#### states (0 items)

(none)

#### conditions (3 items)
- [1] Playing versus Fiora is the condition under which the coach would always take Ignite. (resolution: literal_explicit)
  - cite[13315:13359] byte-exact: 'I mean I would always go ignite versus Fiora'
- [2] 'in most cases' qualifies the Flash+TP recommendation as a default rather than universal rule. (resolution: literal_explicit)
  - cite[13441:13477] byte-exact: 'So I would go flash TP in most cases'
- [3] 'when you need heal cut' is the condition under which taking Ignite is fine. (resolution: literal_explicit)
  - cite[13478:13540] byte-exact: 'yeah but I think ignite when you need heal cut is fine because'

#### recommended_advice (3 items)
- [1] Always go Ignite versus Fiora. (resolution: literal_explicit)
  - cite[13315:13359] byte-exact: 'I mean I would always go ignite versus Fiora'
- [2] Go Flash + TP in most cases, because Flash is considered broken on Camille. (resolution: literal_explicit)
  - cite[13402:13477] byte-exact: 'but I think flash is broken on Camille So I would go flash TP in most cases'
- [3] Ignite is fine when you need heal cut; the truncated clause is completed immediately after the passage: Executioner's cannot be bought on Camille because Trinity and Hydra are needed, so take Ignite if heal cut is needed, otherwise always Flash. (resolution: literal_explicit)
  - cite[13478:13540] byte-exact: 'yeah but I think ignite when you need heal cut is fine because'
  - cite[13541:13793] byte-exact: "you can't buy executioners on this channel yeah you need other items too much you need Trinity and you need Hydra too..."

#### consequences_outcomes (3 items)
- [1] Implied outcome: taking Ignite into the Fiora matchup counters her comparatively high healing. (resolution: context_resolved)
  - cite[13340:13359] byte-exact: 'ignite versus Fiora'
  - cite[13360:13401] byte-exact: 'because she heals more than [\xa0__\xa0] window'
- [2] Implied outcome: buying Executioner's for heal cut would cost Camille her needed Trinity/Hydra power spike, so it is not done 'like a normal Champion'. (resolution: context_resolved)
  - cite[13541:13583] byte-exact: "you can't buy executioners on this channel"
  - cite[13589:13712] byte-exact: "you need other items too much you need Trinity and you need Hydra too much so you can't buy heal cut like a normal Ch..."
- [3] Implied outcome: keeping Flash in the default setup preserves the very strong E+Flash play on Camille. (resolution: context_resolved)
  - cite[13749:13793] byte-exact: 'otherwise always flash cuz e flash is too op'

#### explicit_relationships (8 items)
- [1] Coach USES (recommends) Ignite versus Fiora. (resolution: literal_explicit; relation: USES)
  - cite[13315:13359] byte-exact: 'I mean I would always go ignite versus Fiora'
- [2] 'she' REFERS_TO Fiora. (resolution: context_resolved; relation: REFERS_TO)
  - cite[13360:13401] byte-exact: 'because she heals more than [\xa0__\xa0] window'
- [3] Facing Fiora is the CONDITION for always taking Ignite. (resolution: literal_explicit; relation: CONDITION)
  - cite[13315:13359] byte-exact: 'I mean I would always go ignite versus Fiora'
- [4] Needing heal cut is the CONDITION under which Ignite is fine. (resolution: literal_explicit; relation: CONDITION)
  - cite[13478:13540] byte-exact: 'yeah but I think ignite when you need heal cut is fine because'
- [5] Ignite AFFECTS Fiora's healing (provides the heal cut against it). (resolution: context_resolved; relation: AFFECTS)
  - cite[13340:13359] byte-exact: 'ignite versus Fiora'
  - cite[13478:13540] byte-exact: 'yeah but I think ignite when you need heal cut is fine because'
- [6] Flash being broken on Camille CAUSES the default Flash+TP choice. (resolution: literal_explicit; relation: CAUSES)
  - cite[13402:13477] byte-exact: 'but I think flash is broken on Camille So I would go flash TP in most cases'
- [7] Needing Trinity and Hydra PREVENTS buying Executioner's/heal-cut via items on Camille. (resolution: literal_explicit; relation: PREVENTS)
  - cite[13589:13712] byte-exact: "you need other items too much you need Trinity and you need Hydra too much so you can't buy heal cut like a normal Ch..."
- [8] Ignite ENABLES heal cut where item-based heal cut is unavailable on Camille. (resolution: context_resolved; relation: ENABLES)
  - cite[13478:13540] byte-exact: 'yeah but I think ignite when you need heal cut is fine because'
  - cite[13541:13583] byte-exact: "you can't buy executioners on this channel"

#### uncertainty_unresolved (2 items)
- [1] Corrupted span 'she heals more than [ __ ] window': '[ __ ]' is a censorship placeholder and 'window' does not fit supplied vocabulary or context; the comparator for Fiora's healing is unrecoverable and was left unrestored. (resolution: unresolved)
  - cite[13368:13401] byte-exact: 'she heals more than [\xa0__\xa0] window'
- [2] Suspected ASR corruption in the clause completing the passage: 'you can't buy executioners on this channel' reads like 'on this champion'; 'channel' could not be repaired from supplied material and is left verbatim/unresolved. (resolution: unresolved)
  - cite[13541:13583] byte-exact: "you can't buy executioners on this channel"

#### supporting_source_spans (2 items)
- [1] Full target passage span as embedded verbatim in the condition B transcript. (resolution: literal_explicit)
  - cite[13315:13540] byte-exact: 'I mean I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window but I think flash is broken on...'
  - span [13315:13540]
- [2] Immediately following span that completes the passage's truncated final 'because' clause and closes the spell-choice reasoning. (resolution: context_resolved)
  - cite[13541:13793] byte-exact: "you can't buy executioners on this channel yeah you need other items too much you need Trinity and you need Hydra too..."
  - span [13541:13793]

Raw response binding A: `324b66ffa0b39430118e1e486d65cbeea56da43861f6f7ca497662b7ba9b01b9`
Raw response binding B: `0f8925410c0d30290caf854b7bc8e4dbe334825f14c3ee79eaed55d5e31cc1b3`

---

## TARGET p2k:case:0004 (selection rank 4)

### Exact Bronze

```text
move then you should go if Brier doesn't win one one then you should go because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it winning right I hope Talia does not 2v one you right but
```

Location in full transcript: [54321:54566]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (4 items)
- [1] Briar (surface form "Brier", repaired via supplied champion alias rule Brier->Briar); ally whose win/loss is discussed (resolution: vocabulary_supported)
  - cite[24:52] byte-exact: "if Brier doesn't win one one"
  - cite[105:132] byte-exact: 'just because Brier loses to'
- [2] Taliyah (surface form "Talia", repaired via supplied champion alias rule Talia->Taliyah) (resolution: vocabulary_supported)
  - cite[203:235] byte-exact: 'I hope Talia does not 2v one you'
- [3] The coached player (addressee "you"), the top-lane player per session metadata (resolution: context_resolved)
  - cite[10:23] byte-exact: 'you should go'
  - cite[146:166] byte-exact: "you shouldn't invade"
- [4] The coach (speaker, self-reference "I") (resolution: context_resolved)
  - cite[203:235] byte-exact: 'I hope Talia does not 2v one you'

#### reference_bindings (3 items)
- [1] Second-person "you" throughout the passage binds to the coached player (student) (resolution: context_resolved)
  - cite[10:23] byte-exact: 'you should go'
  - cite[80:104] byte-exact: 'you still want to invade'
- [2] First-person "I" binds to the coach (speaker) (resolution: context_resolved)
  - cite[203:235] byte-exact: 'I hope Talia does not 2v one you'
- [3] "it" in "you could make it winning" has no antecedent resolvable from the supplied material; left unresolved (resolution: unresolved)
  - cite[181:196] byte-exact: 'make it winning'

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (0 items)

(none)

#### conditions (2 items)
- [1] Conditional: if Brier (Briar) does not win "one one" (likely ASR-corrupted phrase, see uncertainty), then you should go (resolution: literal_explicit)
  - cite[24:71] byte-exact: "if Brier doesn't win one one then you should go"
- [2] Concessive condition: even if Brier loses, that fact alone does not mean you shouldn't invade (resolution: literal_explicit)
  - cite[105:166] byte-exact: "just because Brier loses to doesn't mean you shouldn't invade"

#### recommended_advice (2 items)
- [1] You should go if Brier doesn't win "one one" (resolution: literal_explicit)
  - cite[24:71] byte-exact: "if Brier doesn't win one one then you should go"
- [2] You should still go and invade even if Brier loses, because you still want to invade; Brier's loss does not cancel the invade plan (resolution: literal_explicit)
  - cite[53:104] byte-exact: 'then you should go because you still want to invade'
  - cite[133:166] byte-exact: "doesn't mean you shouldn't invade"

#### consequences_outcomes (2 items)
- [1] Invading could make "it" (referent unresolved) winning; i.e., invading may produce a winning outcome (resolution: literal_explicit)
  - cite[167:196] byte-exact: 'cuz you could make it winning'
- [2] Implied unwanted outcome the coach hopes to avoid: Taliyah 2v1-ing you ("2v one" likely ASR corruption of "2v1") (resolution: context_resolved)
  - cite[203:235] byte-exact: 'I hope Talia does not 2v one you'

#### explicit_relationships (3 items)
- [1] 'Brier doesn't win one one' is the condition under which 'you should go' holds (resolution: literal_explicit; relation: CONDITION)
  - cite[24:71] byte-exact: "if Brier doesn't win one one then you should go"
- [2] Your invading could cause 'it' (unresolved referent) to become winning (resolution: literal_explicit; relation: CAUSES)
  - cite[80:104] byte-exact: 'you still want to invade'
  - cite[167:196] byte-exact: 'cuz you could make it winning'
- [3] Brier losing does not negate the recommendation to invade ('doesn't mean you shouldn't invade') (resolution: literal_explicit; relation: NEGATES)
  - cite[105:166] byte-exact: "just because Brier loses to doesn't mean you shouldn't invade"

#### uncertainty_unresolved (5 items)
- [1] "one one" after "win" appears to be ASR corruption, plausibly "1v1", but this cannot be confirmed from the supplied material (resolution: unresolved)
  - cite[41:57] byte-exact: 'win one one then'
- [2] Trailing "to" in "Brier loses to" appears to be ASR corruption (possibly "too" or truncation); unrecoverable (resolution: unresolved)
  - cite[118:132] byte-exact: 'Brier loses to'
- [3] "2v one" appears to be ASR corruption of "2v1"; repair not licensed by the supplied vocabulary, left unresolved (resolution: unresolved)
  - cite[221:235] byte-exact: 'not 2v one you'
- [4] Referent of "it" in "make it winning" cannot be resolved from the supplied material (resolution: unresolved)
  - cite[181:196] byte-exact: 'make it winning'
- [5] Passage begins mid-sentence at "move then"; the destination of the initial 'go' and prior context are unavailable (resolution: unresolved)
  - cite[0:23] byte-exact: 'move then you should go'

#### supporting_source_spans (5 items)
- [1] Supports the conditional advice that you should go if Brier doesn't win (resolution: literal_explicit)
  - cite[24:71] byte-exact: "if Brier doesn't win one one then you should go"
  - span [24:71]
- [2] Supports the rationale that you still want to invade (resolution: literal_explicit)
  - cite[53:104] byte-exact: 'then you should go because you still want to invade'
  - span [53:104]
- [3] Supports that Brier losing does not preclude invading (resolution: literal_explicit)
  - cite[105:166] byte-exact: "just because Brier loses to doesn't mean you shouldn't invade"
  - span [105:166]
- [4] Supports the potential positive outcome of invading (resolution: literal_explicit)
  - cite[167:196] byte-exact: 'cuz you could make it winning'
  - span [167:196]
- [5] Supports the coach's stated concern about Taliyah fighting you 2v1 (resolution: literal_explicit)
  - cite[203:235] byte-exact: 'I hope Talia does not 2v one you'
  - span [203:235]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (4 items)
- [1] 'you' — the coached student/player (the Aatrox top laner in this game), the person the advice about moving is addressed to (resolution: context_resolved)
  - cite[54321:54392] byte-exact: "move then you should go if Brier doesn't win one one then you should go"
  - cite[54243:54297] byte-exact: "it's your job to back up Brier if she needs the backup"
- [2] Brier — literal surface name preserved; binds to champion Briar via the supplied alias rules; the player's allied jungler conducting the invade under discussion (resolution: context_resolved)
  - cite[54345:54392] byte-exact: "if Brier doesn't win one one then you should go"
  - cite[54700:54727] byte-exact: "here she's invading Raptors"
- [3] Talia — literal surface name preserved; binds to champion Taliyah via the supplied alias rules; the enemy jungler who might '2v one' the player (resolution: context_resolved)
  - cite[54518:54562] byte-exact: 'right I hope Talia does not 2v one you right'
  - cite[54158:54242] byte-exact: "when is like if if if Talia is invading Brier it's my job to F gar or when is it job"
- [4] IIA — corrupted entity mention in the clause immediately leading into the target passage ('let's say IIA can move'); its referent cannot be recovered from the supplied material (resolution: unresolved)
  - cite[54298:54344] byte-exact: "okay let's say IIA can move then you should go"

#### reference_bindings (4 items)
- [1] 'you' (in 'then you should go', 'you shouldn't invade', 'you could make it winning', '2v one you') binds to the coached student/player (Aatrox top) (resolution: context_resolved)
  - cite[54321:54392] byte-exact: "move then you should go if Brier doesn't win one one then you should go"
  - cite[54393:54517] byte-exact: "because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it ..."
  - cite[54243:54297] byte-exact: "it's your job to back up Brier if she needs the backup"
- [2] 'one one' binds to the 1v1 fight between Brier (Briar) and Talia (Taliyah) inside her invade, established by the surrounding dialogue where the student asks whether his job is to fight Garen when Talia is invading Brier (resolution: context_resolved)
  - cite[54345:54392] byte-exact: "if Brier doesn't win one one then you should go"
  - cite[54158:54242] byte-exact: "when is like if if if Talia is invading Brier it's my job to F gar or when is it job"
- [3] 'it' in 'you could make it winning' binds to the invade fight (the Brier-versus-Talia engagement the player would join by going) (resolution: context_resolved)
  - cite[54393:54517] byte-exact: "because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it ..."
- [4] Opening truncated clause: the grammatical subject of '[X] can move' sits before the target passage start and surfaces as the corrupted token 'IIA'; the binding is unresolved (resolution: unresolved)
  - cite[54298:54344] byte-exact: "okay let's say IIA can move then you should go"

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (2 items)
- [1] Ongoing invade state: Brier (Briar) is inside the enemy jungle (Raptors) at the moment this advice is given, which is why the player's movement decision matters (resolution: context_resolved)
  - cite[54393:54517] byte-exact: "because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it ..."
  - cite[54700:54727] byte-exact: "here she's invading Raptors"
- [2] Player-intent state: the player still wants to invade even in the scenario where Brier loses (resolution: literal_explicit)
  - cite[54393:54517] byte-exact: "because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it ..."

#### conditions (3 items)
- [1] Condition: if Brier does not win the 1v1, then the player should go (resolution: literal_explicit)
  - cite[54345:54392] byte-exact: "if Brier doesn't win one one then you should go"
- [2] Condition with truncated antecedent: '[corrupted subject IIA] can move' is the stated trigger for 'then you should go'; the triggering entity is unresolved (resolution: unresolved)
  - cite[54298:54344] byte-exact: "okay let's say IIA can move then you should go"
- [3] Concessive condition: even in the case Brier loses (surface form 'loses to'), that loss is explicitly not grounds for the player to refrain from invading (resolution: literal_explicit)
  - cite[54426:54487] byte-exact: "just because Brier loses to doesn't mean you shouldn't invade"

#### recommended_advice (2 items)
- [1] Go to back up Brier: the player should go (join the invade) if Brier doesn't win the 1v1 (resolution: literal_explicit)
  - cite[54345:54392] byte-exact: "if Brier doesn't win one one then you should go"
  - cite[54321:54392] byte-exact: "move then you should go if Brier doesn't win one one then you should go"
- [2] Do not skip the invade because Brier lost: the player should still invade, since his presence could make the fight winning (resolution: literal_explicit)
  - cite[54393:54517] byte-exact: "because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it ..."

#### consequences_outcomes (3 items)
- [1] Potential positive outcome: the player going/joining could make the invade fight winning ('you could make it winning') (resolution: literal_explicit)
  - cite[54488:54517] byte-exact: 'cuz you could make it winning'
- [2] Feared negative outcome the coach hopes to avoid: Talia turning the situation into a 2v1 against the player ('does not 2v one you') (resolution: literal_explicit)
  - cite[54518:54562] byte-exact: 'right I hope Talia does not 2v one you right'
- [3] Stated non-outcome: Brier losing the 1v1 does not produce the conclusion that the player shouldn't invade (resolution: literal_explicit)
  - cite[54426:54487] byte-exact: "just because Brier loses to doesn't mean you shouldn't invade"

#### explicit_relationships (8 items)
- [1] CONDITION: 'Brier doesn't win one one' is the condition under which 'then you should go' applies (resolution: literal_explicit; relation: CONDITION)
  - cite[54345:54392] byte-exact: "if Brier doesn't win one one then you should go"
- [2] CONDITION: '[IIA] can move' (corrupted subject) is the condition under which 'then you should go' applies (resolution: unresolved; relation: CONDITION)
  - cite[54298:54344] byte-exact: "okay let's say IIA can move then you should go"
- [3] ENABLES: the player going/joining enables making the invade fight winning (resolution: context_resolved; relation: ENABLES)
  - cite[54393:54517] byte-exact: "because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it ..."
- [4] NEGATES: Brier losing the 1v1 rebuts the conclusion that the player shouldn't invade ('doesn't mean you shouldn't invade') (resolution: literal_explicit; relation: NEGATES)
  - cite[54426:54487] byte-exact: "just because Brier loses to doesn't mean you shouldn't invade"
- [5] ACTOR: 'Brier' is the actor whose 1v1 result ('one one') triggers the advice to go (resolution: literal_explicit; relation: ACTOR)
  - cite[54345:54392] byte-exact: "if Brier doesn't win one one then you should go"
- [6] TARGET: 'you' (the player) is the target of the potential '2v one' by Talia (resolution: literal_explicit; relation: TARGET)
  - cite[54518:54562] byte-exact: 'right I hope Talia does not 2v one you right'
- [7] REFERS_TO: 'one one' refers to the Brier-versus-Talia 1v1 engagement implied by the surrounding question about Talia invading Brier (resolution: context_resolved; relation: REFERS_TO)
  - cite[54345:54392] byte-exact: "if Brier doesn't win one one then you should go"
  - cite[54158:54242] byte-exact: "when is like if if if Talia is invading Brier it's my job to F gar or when is it job"
- [8] ACTOR: 'you' (the player) is the actor of the advised action 'go' (resolution: literal_explicit; relation: ACTOR)
  - cite[54321:54392] byte-exact: "move then you should go if Brier doesn't win one one then you should go"

#### uncertainty_unresolved (3 items)
- [1] ASR-corrupted token 'IIA' in 'okay let's say IIA can move': referent unrecoverable from supplied material and alias vocabulary (could denote Talia, Brier, the player, or another entity); the first condition's trigger therefore stays unresolved (resolution: unresolved)
  - cite[54298:54344] byte-exact: "okay let's say IIA can move then you should go"
- [2] Incomplete ASR phrase 'Brier loses to' in 'just because Brier loses to doesn't mean you shouldn't invade': the clause appears truncated/corrupted (e.g. 'loses too'); exact intended wording unrecoverable (resolution: unresolved)
  - cite[54426:54487] byte-exact: "just because Brier loses to doesn't mean you shouldn't invade"
- [3] Ambiguity in '2v one you': the second participant of the potential 2v1 against the player is never identified in the supplied material (resolution: unresolved)
  - cite[54518:54562] byte-exact: 'right I hope Talia does not 2v one you right'

#### supporting_source_spans (3 items)
- [1] Core target-passage span carrying the advice, its conditions, the reasoning, and the hoped-against outcome (resolution: literal_explicit)
  - cite[54321:54562] byte-exact: "move then you should go if Brier doesn't win one one then you should go because you still want to invade just because..."
  - span [54321:54562]
- [2] Preceding-context span supplying the truncated first clause and the corrupted subject 'IIA' (resolution: unresolved)
  - cite[54298:54344] byte-exact: "okay let's say IIA can move then you should go"
  - span [54298:54344]
- [3] Surrounding-dialogue spans resolving the participants and the backing-up framing of the advice (resolution: context_resolved)
  - cite[54243:54297] byte-exact: "it's your job to back up Brier if she needs the backup"
  - cite[54158:54242] byte-exact: "when is like if if if Talia is invading Brier it's my job to F gar or when is it job"
  - cite[54700:54727] byte-exact: "here she's invading Raptors"
  - span [54158:54727]

Raw response binding A: `ffe689b5a3d383800d25af436f9cbe2261fd0d37e54fe5819c525768ebe12fd6`
Raw response binding B: `effd2af1dab440322eb312c3890f185eebb367bfd160599c604d9674fc0c5e79`

---

## TARGET p2k:case:0005 (selection rank 5)

### Exact Bronze

```text
then does she get to farm no no she loses the whole wave then yes yes so that's why you should run at her now that she uses Q because if she uses e right we then you lose like 100 HP 150 but then she has no
```

Location in full transcript: [55339:55545]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (2 items)
- [1] Unnamed opposing champion/player referred to as 'she'/'her' throughout the passage; the enemy laner under discussion in the matchup. Her specific champion identity is not recoverable from the supplied material. (resolution: context_resolved)
  - cite[5:56] byte-exact: 'does she get to farm no no she loses the whole wave'
  - cite[95:125] byte-exact: 'run at her now that she uses Q'
- [2] The addressed player ('you'): the coached student receiving the advice, who would lose roughly 100-150 HP in the hypothetical branch. (resolution: context_resolved)
  - cite[73:105] byte-exact: "that's why you should run at her"
  - cite[157:186] byte-exact: 'then you lose like 100 HP 150'

#### reference_bindings (9 items)
- [1] 'she' in 'does she get to farm' binds to the unnamed opposing champion under discussion. (resolution: context_resolved)
  - cite[5:25] byte-exact: 'does she get to farm'
- [2] 'she' in 'she loses the whole wave' binds to the same unnamed opposing champion. (resolution: context_resolved)
  - cite[26:56] byte-exact: 'no no she loses the whole wave'
- [3] 'her' in 'run at her' binds to the same unnamed opposing champion. (resolution: context_resolved)
  - cite[73:105] byte-exact: "that's why you should run at her"
- [4] 'she' in 'now that she uses Q' binds to the same unnamed opposing champion. (resolution: context_resolved)
  - cite[106:125] byte-exact: 'now that she uses Q'
- [5] 'she' in 'if she uses e' binds to the same unnamed opposing champion. (resolution: context_resolved)
  - cite[134:186] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
- [6] 'she' in 'but then she has no' binds to the same unnamed opposing champion. (resolution: context_resolved)
  - cite[187:206] byte-exact: 'but then she has no'
- [7] 'you' in 'you should run at her' and 'then you lose like 100 HP 150' binds to the addressed coached player. (resolution: context_resolved)
  - cite[73:105] byte-exact: "that's why you should run at her"
  - cite[157:186] byte-exact: 'then you lose like 100 HP 150'
- [8] 'we' in 'right we then' cannot be confidently bound: possible ASR artifact, self-correction, or speaker-inclusive reference; recorded unresolved rather than guessed. (resolution: unresolved)
  - cite[148:186] byte-exact: 'right we then you lose like 100 HP 150'
- [9] 'that' in 'so that's why' refers back to the immediately preceding exchange establishing that she does not get to farm and loses the whole wave. (resolution: context_resolved)
  - cite[62:105] byte-exact: "yes yes so that's why you should run at her"
  - cite[5:56] byte-exact: 'does she get to farm no no she loses the whole wave'

#### abilities_resources (4 items)
- [1] Ability 'Q' owned by the unnamed opposing champion ('she'), cited as the timing basis for the advice; surface form matches the supplied ability key Q. (resolution: vocabulary_supported)
  - cite[106:125] byte-exact: 'now that she uses Q'
- [2] Ability 'E' (surface form lowercase 'e', read as ability key E) owned by the same unnamed opposing champion, invoked hypothetically under 'if'. (resolution: vocabulary_supported)
  - cite[134:186] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
- [3] Health resource (HP) of the addressed player: a stated loss of 'like 100 HP 150' (approximately 100-150 HP) in the hypothetical branch. (resolution: literal_explicit)
  - cite[162:186] byte-exact: 'you lose like 100 HP 150'
- [4] Farm/wave resources at stake for the unnamed opposing champion: she gets no farm and loses the whole wave. (resolution: literal_explicit)
  - cite[5:25] byte-exact: 'does she get to farm'
  - cite[32:56] byte-exact: 'she loses the whole wave'

#### events_actions (4 items)
- [1] A question is raised whether she gets to farm, answered negatively ('no no'): she does not get to farm. (resolution: literal_explicit)
  - cite[5:56] byte-exact: 'does she get to farm no no she loses the whole wave'
- [2] She loses the whole wave. (resolution: literal_explicit)
  - cite[32:56] byte-exact: 'she loses the whole wave'
- [3] She uses Q, stated as the present basis ('now that she uses Q') for the advised aggression. (resolution: literal_explicit)
  - cite[106:125] byte-exact: 'now that she uses Q'
- [4] Hypothetical event under the stated condition: she uses E ('if she uses e'). (resolution: literal_explicit)
  - cite[134:147] byte-exact: 'if she uses e'

#### states (2 items)
- [1] Wave state: the entire wave is lost to her (she secures no farm from it). (resolution: literal_explicit)
  - cite[26:56] byte-exact: 'no no she loses the whole wave'
- [2] Resource state asserted but truncated: 'she has no' followed by an unstated resource or ability availability; the clause is cut off. (resolution: unresolved)
  - cite[187:206] byte-exact: 'but then she has no'

#### conditions (2 items)
- [1] Condition/timing for the advice: run at her 'now that she uses Q'. (resolution: literal_explicit)
  - cite[106:125] byte-exact: 'now that she uses Q'
- [2] Condition for the HP-loss consequence: 'if she uses e'. (resolution: literal_explicit)
  - cite[134:186] byte-exact: 'if she uses e right we then you lose like 100 HP 150'

#### recommended_advice (1 items)
- [1] Coach's advice: the addressed player should run at her, timed to when she uses Q ('now that she uses Q'). (resolution: literal_explicit)
  - cite[70:125] byte-exact: "so that's why you should run at her now that she uses Q"

#### consequences_outcomes (3 items)
- [1] Outcome for her: she loses the whole wave (securing no farm). (resolution: literal_explicit)
  - cite[26:56] byte-exact: 'no no she loses the whole wave'
- [2] Conditional outcome: if she uses E, the addressed player loses approximately 100-150 HP. (resolution: literal_explicit)
  - cite[134:186] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
- [3] Additional outcome introduced but truncated mid-clause: 'but then she has no ...' — what she lacks is unrecoverable from the supplied text. (resolution: unresolved)
  - cite[187:206] byte-exact: 'but then she has no'

#### explicit_relationships (9 items)
- [1] USES: the unnamed opposing champion ('she') uses her Q ability. (resolution: literal_explicit; relation: USES)
  - cite[106:125] byte-exact: 'now that she uses Q'
- [2] USES: the unnamed opposing champion ('she') would use her E ability in the hypothetical branch. (resolution: literal_explicit; relation: USES)
  - cite[134:147] byte-exact: 'if she uses e'
- [3] CAUSES: her using E causes the addressed player to lose roughly 100-150 HP. (resolution: literal_explicit; relation: CAUSES)
  - cite[134:186] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
- [4] CONDITION: the advice to run at her is conditioned on her using Q. (resolution: literal_explicit; relation: CONDITION)
  - cite[70:125] byte-exact: "so that's why you should run at her now that she uses Q"
- [5] NEGATES: the reply 'no no' negates the proposition that she gets to farm. (resolution: literal_explicit; relation: NEGATES)
  - cite[5:56] byte-exact: 'does she get to farm no no she loses the whole wave'
- [6] ACTOR: the addressed player ('you') is the actor of the advised action 'run at her'. (resolution: literal_explicit; relation: ACTOR)
  - cite[73:105] byte-exact: "that's why you should run at her"
- [7] TARGET: the unnamed opposing champion ('her') is the target of the advised action 'run at her'. (resolution: literal_explicit; relation: TARGET)
  - cite[73:105] byte-exact: "that's why you should run at her"
- [8] RESULT: losing the whole wave is the stated outcome she suffers. (resolution: literal_explicit; relation: RESULT)
  - cite[32:56] byte-exact: 'she loses the whole wave'
- [9] REFERS_TO: 'that' in 'so that's why' refers to the preceding exchange (she does not get to farm; she loses the whole wave). (resolution: context_resolved; relation: REFERS_TO)
  - cite[62:105] byte-exact: "yes yes so that's why you should run at her"

#### uncertainty_unresolved (4 items)
- [1] The specific champion identity behind 'she'/'her' cannot be resolved from the supplied Condition A source; no champion name occurs in the passage, so no binding is attempted. (resolution: unresolved)
  - cite[95:125] byte-exact: 'run at her now that she uses Q'
- [2] 'right we then' appears to be an ASR artifact or self-correction; the referent of 'we' (false start versus speaker-inclusive reference) is unrecoverable. (resolution: unresolved)
  - cite[148:186] byte-exact: 'right we then you lose like 100 HP 150'
- [3] The passage ends mid-sentence ('but then she has no'): the resource or state she lacks is truncated and cannot be recovered. (resolution: unresolved)
  - cite[187:206] byte-exact: 'but then she has no'
- [4] Speaker attribution is uncertain: the exchange alternates between question, answers ('no no', 'yes yes') and advice without speaker labels, so which voice utters each turn cannot be confirmed. (resolution: unresolved)
  - cite[26:56] byte-exact: 'no no she loses the whole wave'
  - cite[62:105] byte-exact: "yes yes so that's why you should run at her"

#### supporting_source_spans (3 items)
- [1] Span covering the opening exchange: question about her farming, negative answer, and wave-loss outcome. (resolution: literal_explicit)
  - cite[5:56] byte-exact: 'does she get to farm no no she loses the whole wave'
  - span [5:56]
- [2] Span covering the advice and its Q-based timing rationale. (resolution: literal_explicit)
  - cite[62:125] byte-exact: "yes yes so that's why you should run at her now that she uses Q"
  - span [62:125]
- [3] Span covering the E-conditional trade consequence and the truncated closing clause. (resolution: literal_explicit)
  - cite[126:206] byte-exact: 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
  - span [126:206]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (2 items)
- [1] Enemy mid laner Syndra, referred to in the target passage only as 'she' and 'her' (resolution: context_resolved)
  - cite[47585:47606] byte-exact: "I'm way versus syndra"
  - cite[55258:55400] byte-exact: "so if you end up on this side and you push her away because let's say she uses e then does she get to farm no no she ..."
- [2] The coached player ('you'), playing mid against Syndra in this game (resolution: context_resolved)
  - cite[55409:55525] byte-exact: "so that's why you should run at her now that she uses Q because if she uses e right we then you lose like 100 HP 150"
  - cite[47585:47606] byte-exact: "I'm way versus syndra"

#### reference_bindings (3 items)
- [1] 'she' in 'does she get to farm', 'now that she uses Q', and 'if she uses e' binds to the enemy mid laner Syndra (resolution: context_resolved)
  - cite[55339:55400] byte-exact: 'then does she get to farm no no she loses the whole wave then'
  - cite[47585:47606] byte-exact: "I'm way versus syndra"
- [2] 'her' in 'you should run at her' binds to the enemy mid laner Syndra (resolution: context_resolved)
  - cite[55409:55472] byte-exact: "so that's why you should run at her now that she uses Q because"
  - cite[47585:47606] byte-exact: "I'm way versus syndra"
- [3] 'you' in 'you should run at her' and 'you lose like 100 HP 150' binds to the coached player (resolution: context_resolved)
  - cite[55409:55472] byte-exact: "so that's why you should run at her now that she uses Q because"
  - cite[55496:55525] byte-exact: 'then you lose like 100 HP 150'

#### abilities_resources (4 items)
- [1] Q - the enemy Syndra's ability; she has just used it when the advice applies ('now that she uses Q') (resolution: context_resolved)
  - cite[55445:55495] byte-exact: 'now that she uses Q because if she uses e right we'
  - cite[49470:49535] byte-exact: 'try to bait out her Q or her W or her e depends like what she has'
  - cite[47585:47606] byte-exact: "I'm way versus syndra"
- [2] E - the enemy Syndra's ability; her hypothesized response while being run at ('if she uses e') (resolution: context_resolved)
  - cite[55473:55525] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
  - cite[49470:49535] byte-exact: 'try to bait out her Q or her W or her e depends like what she has'
- [3] Player health resource: about 100-150 HP lost if she uses E (resolution: literal_explicit)
  - cite[55496:55525] byte-exact: 'then you lose like 100 HP 150'
- [4] Lane farm / the whole minion wave as the lane resource she forfeits when denied (resolution: literal_explicit)
  - cite[55339:55400] byte-exact: 'then does she get to farm no no she loses the whole wave then'

#### events_actions (4 items)
- [1] She uses Q at the moment the advice applies ('now that she uses Q') (resolution: literal_explicit)
  - cite[55445:55495] byte-exact: 'now that she uses Q because if she uses e right we'
- [2] Hypothesized action while being run at: she uses E (resolution: literal_explicit)
  - cite[55473:55525] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
- [3] She does not get to farm and loses the whole wave (resolution: literal_explicit)
  - cite[55339:55400] byte-exact: 'then does she get to farm no no she loses the whole wave then'
- [4] You lose like 100 HP 150 under the 'if she uses e' condition (resolution: literal_explicit)
  - cite[55496:55525] byte-exact: 'then you lose like 100 HP 150'

#### states (2 items)
- [1] After using E she would have no spell available; the target passage truncates at 'but then she has no' and the source continues 'but then she has no spell' (resolution: context_resolved)
  - cite[55526:55545] byte-exact: 'but then she has no'
  - cite[55526:55556] byte-exact: 'but then she has no spell yeah'
- [2] Her Q is already spent when the run-at advice applies (resolution: context_resolved)
  - cite[55445:55495] byte-exact: 'now that she uses Q because if she uses e right we'

#### conditions (2 items)
- [1] Condition 'if she uses e': under this condition you lose like 100 HP 150 (resolution: literal_explicit)
  - cite[55473:55525] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
- [2] Trigger condition 'now that she uses Q': this is when you should run at her (resolution: literal_explicit)
  - cite[55409:55472] byte-exact: "so that's why you should run at her now that she uses Q because"

#### recommended_advice (1 items)
- [1] Run at her now that she has used her Q (resolution: literal_explicit)
  - cite[55409:55472] byte-exact: "so that's why you should run at her now that she uses Q because"

#### consequences_outcomes (2 items)
- [1] Denied the farm, she loses the whole wave (resolution: literal_explicit)
  - cite[55339:55400] byte-exact: 'then does she get to farm no no she loses the whole wave then'
  - cite[55258:55400] byte-exact: "so if you end up on this side and you push her away because let's say she uses e then does she get to farm no no she ..."
- [2] If she uses E, you lose like 100 HP 150 (resolution: literal_explicit)
  - cite[55473:55525] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
  - cite[55496:55525] byte-exact: 'then you lose like 100 HP 150'

#### explicit_relationships (7 items)
- [1] she (enemy Syndra) uses Q (resolution: context_resolved; relation: USES)
  - cite[55445:55495] byte-exact: 'now that she uses Q because if she uses e right we'
  - cite[47585:47606] byte-exact: "I'm way versus syndra"
- [2] she (enemy Syndra) uses E (hypothesized response to being run at) (resolution: context_resolved; relation: USES)
  - cite[55473:55525] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
  - cite[47585:47606] byte-exact: "I'm way versus syndra"
- [3] If she uses E, then you lose like 100 HP 150 (resolution: vocabulary_supported; relation: CAUSES)
  - cite[55473:55525] byte-exact: 'if she uses e right we then you lose like 100 HP 150'
  - cite[55496:55525] byte-exact: 'then you lose like 100 HP 150'
- [4] Pushing her away / denying the farm causes her to lose the whole wave (resolution: vocabulary_supported; relation: CAUSES)
  - cite[55258:55400] byte-exact: "so if you end up on this side and you push her away because let's say she uses e then does she get to farm no no she ..."
  - cite[55339:55400] byte-exact: 'then does she get to farm no no she loses the whole wave then'
- [5] Running at her now that she uses Q stops her from getting to farm, which is why she loses the whole wave (resolution: context_resolved; relation: PREVENTS)
  - cite[55409:55525] byte-exact: "so that's why you should run at her now that she uses Q because if she uses e right we then you lose like 100 HP 150"
  - cite[55339:55400] byte-exact: 'then does she get to farm no no she loses the whole wave then'
- [6] 'she'/'her' in the target passage refer to the enemy mid laner Syndra (resolution: context_resolved; relation: REFERS_TO)
  - cite[55339:55400] byte-exact: 'then does she get to farm no no she loses the whole wave then'
  - cite[55409:55472] byte-exact: "so that's why you should run at her now that she uses Q because"
  - cite[47585:47606] byte-exact: "I'm way versus syndra"
- [7] The advised action of running at 'her' targets the enemy Syndra (resolution: context_resolved; relation: TARGET)
  - cite[55409:55472] byte-exact: "so that's why you should run at her now that she uses Q because"
  - cite[47585:47606] byte-exact: "I'm way versus syndra"

#### uncertainty_unresolved (3 items)
- [1] ASR fragment 'we' in 'if she uses e right we then you lose like 100 HP 150' is unintelligible filler and is left unrestored (resolution: unresolved)
  - cite[55445:55495] byte-exact: 'now that she uses Q because if she uses e right we'
- [2] The target passage breaks off mid-clause at 'but then she has no'; the missing complement is only recoverable from the immediately following transcript words ('spell') and is unsupported within the target passage itself (resolution: unresolved)
  - cite[55526:55545] byte-exact: 'but then she has no'
  - cite[55526:55556] byte-exact: 'but then she has no spell yeah'
- [3] The coached player's own champion cannot be identified from the supplied material; the ASR form 'way' cannot be repaired to any champion in the supplied lexical vocabulary (resolution: unresolved)
  - cite[47585:47606] byte-exact: "I'm way versus syndra"

#### supporting_source_spans (3 items)
- [1] Verbatim target passage span (contiguous slice of the Condition B transcript) carrying the core extraction (resolution: literal_explicit)
  - cite[55339:55545] byte-exact: "then does she get to farm no no she loses the whole wave then yes yes so that's why you should run at her now that sh..."
  - span [55339:55545]
- [2] Immediately preceding context establishing that 'she' is the enemy Syndra being pushed off the wave after her E (resolution: context_resolved)
  - cite[55258:55400] byte-exact: "so if you end up on this side and you push her away because let's say she uses e then does she get to farm no no she ..."
  - cite[47585:47606] byte-exact: "I'm way versus syndra"
  - span [47585:55400]
- [3] Continuation immediately after the passage boundary completing the truncated final state ('no spell') (resolution: context_resolved)
  - cite[55526:55556] byte-exact: 'but then she has no spell yeah'
  - span [55526:55556]

Raw response binding A: `97a8412e97348a8b8e1f94b1d1a74f672fff025f70e541ce87a35ba543e3c212`
Raw response binding B: `81971d4a3517e5e1c1754d3f5f467fc56b5fdb442f6840826e66cdae6c37751b`

---

## TARGET p2k:case:0006 (selection rank 6)

### Exact Bronze

```text
is to proc scorch and comet right but it's a habit you don't want to have because you could just be here and land maybe two more ticks right and on when you pick up other champions this is a extremely bad freeze frame because if this was
```

Location in full transcript: [4234:4471]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (2 items)
- [1] you - the player being coached (Mel mid per supplied metadata) (resolution: context_resolved)
  - cite[43:73] byte-exact: "a habit you don't want to have"
- [2] other champions - referenced in 'when you pick up other champions'; surface wording preserved, intended sense unclear (resolution: literal_explicit)
  - cite[148:180] byte-exact: 'when you pick up other champions'

#### reference_bindings (5 items)
- [1] 'it' in 'it's a habit' binds to the criticized behavior whose stated purpose 'is to proc scorch and comet'; antecedent partially truncated at passage start (resolution: context_resolved)
  - cite[6:50] byte-exact: "proc scorch and comet right but it's a habit"
- [2] 'you' binds to the addressed player (Mel mid per supplied metadata) (resolution: context_resolved)
  - cite[82:134] byte-exact: 'you could just be here and land maybe two more ticks'
- [3] 'this' in 'this is a extremely bad freeze frame' binds to the reviewed game moment/positioning shown at the freeze frame (resolution: context_resolved)
  - cite[181:217] byte-exact: 'this is a extremely bad freeze frame'
- [4] 'here' - deictic location indicated by the coach; no concrete map position recoverable from supplied material (resolution: unresolved)
  - cite[92:113] byte-exact: 'just be here and land'
- [5] 'if this was' opens a truncated conditional; antecedent and comparison target unrecoverable (resolution: unresolved)
  - cite[218:237] byte-exact: 'because if this was'

#### abilities_resources (2 items)
- [1] scorch - one of the two effects the unstated action aims to proc (rune reference; owner not stated in the passage) (resolution: literal_explicit)
  - cite[6:27] byte-exact: 'proc scorch and comet'
- [2] comet - the second effect the unstated action aims to proc (rune reference; owner not stated in the passage) (resolution: literal_explicit)
  - cite[6:27] byte-exact: 'proc scorch and comet'

#### events_actions (1 items)
- [1] An unstated action (passage begins mid-clause) is characterized as being done 'to proc scorch and comet' (resolution: literal_explicit)
  - cite[0:33] byte-exact: 'is to proc scorch and comet right'

#### states (2 items)
- [1] Positional state: compared to the coach's indicated spot 'here', the player's current positioning yields roughly two fewer damage ticks (resolution: context_resolved)
  - cite[82:140] byte-exact: 'you could just be here and land maybe two more ticks right'
- [2] Reviewed-moment state: 'this' moment at the freeze frame is assessed as extremely bad (resolution: literal_explicit)
  - cite[181:217] byte-exact: 'this is a extremely bad freeze frame'

#### conditions (2 items)
- [1] 'when' condition: 'when you pick up other champions' directly frames the assessment 'this is a extremely bad freeze frame'; what 'pick up' denotes is unclear (see uncertainty_unresolved) (resolution: literal_explicit)
  - cite[148:217] byte-exact: 'when you pick up other champions this is a extremely bad freeze frame'
- [2] 'if' condition introduced then truncated: 'because if this was' - condition content and consequent unrecoverable (resolution: unresolved)
  - cite[218:237] byte-exact: 'because if this was'

#### recommended_advice (1 items)
- [1] Coach advises against keeping the habit: the criticized behavior (aimed to proc scorch and comet) is 'a habit you don't want to have' (resolution: literal_explicit)
  - cite[34:73] byte-exact: "but it's a habit you don't want to have"

#### consequences_outcomes (2 items)
- [1] Outcome of alternative positioning: from 'here' the player could land maybe two more ticks (resolution: literal_explicit)
  - cite[82:140] byte-exact: 'you could just be here and land maybe two more ticks right'
- [2] Implied cost of keeping the habit: the criticized positioning forfeits the roughly two additional ticks available from the better spot (resolution: context_resolved)
  - cite[74:140] byte-exact: 'because you could just be here and land maybe two more ticks right'

#### explicit_relationships (6 items)
- [1] you --ACTOR--> hypothetically being 'here' and landing maybe two more ticks (resolution: literal_explicit; relation: ACTOR)
  - cite[82:140] byte-exact: 'you could just be here and land maybe two more ticks right'
- [2] being positioned at the coach-indicated spot 'here' --ENABLES--> landing maybe two more ticks (resolution: literal_explicit; relation: ENABLES)
  - cite[82:140] byte-exact: 'you could just be here and land maybe two more ticks right'
- [3] 'when you pick up other champions' --CONDITION--> the assessment 'this is a extremely bad freeze frame' (resolution: context_resolved; relation: CONDITION)
  - cite[148:217] byte-exact: 'when you pick up other champions this is a extremely bad freeze frame'
- [4] coach's advice 'a habit you don't want to have' --NEGATES--> the habit of the criticized scorch/comet-proc positioning (resolution: context_resolved; relation: NEGATES)
  - cite[22:73] byte-exact: "comet right but it's a habit you don't want to have"
- [5] 'it' --REFERS_TO--> the criticized behavior aimed 'to proc scorch and comet' (resolution: context_resolved; relation: REFERS_TO)
  - cite[6:50] byte-exact: "proc scorch and comet right but it's a habit"
- [6] 'this' --REFERS_TO--> the reviewed game moment shown at the freeze frame (resolution: context_resolved; relation: REFERS_TO)
  - cite[165:217] byte-exact: 'other champions this is a extremely bad freeze frame'

#### uncertainty_unresolved (6 items)
- [1] Truncated opening: the subject/agent of 'is to proc scorch and comet' is cut off; the action being characterized cannot be recovered from supplied material (resolution: unresolved)
  - cite[0:33] byte-exact: 'is to proc scorch and comet right'
- [2] 'and on when you pick up other champions': possible ASR corruption; what 'pick up' applies to cannot be determined (resolution: unresolved)
  - cite[141:180] byte-exact: 'and on when you pick up other champions'
- [3] 'freeze frame' phrasing may be literal (paused video frame) or ASR corruption; not recoverable beyond surface words (resolution: unresolved)
  - cite[181:217] byte-exact: 'this is a extremely bad freeze frame'
- [4] Tick source unspecified: the passage does not identify which ability or rune delivers the 'two more ticks' (resolution: unresolved)
  - cite[109:140] byte-exact: 'land maybe two more ticks right'
- [5] Deictic 'here' cannot be mapped to a concrete map/screen position from supplied material (resolution: unresolved)
  - cite[92:113] byte-exact: 'just be here and land'
- [6] Truncated ending: 'because if this was' leaves the conditional unrecoverable (resolution: unresolved)
  - cite[218:237] byte-exact: 'because if this was'

#### supporting_source_spans (5 items)
- [1] Opening span covering the purpose clause of the unstated action (resolution: literal_explicit)
  - cite[0:33] byte-exact: 'is to proc scorch and comet right'
  - span [0:33]
- [2] Span covering the coach's advice against the habit (resolution: literal_explicit)
  - cite[34:73] byte-exact: "but it's a habit you don't want to have"
  - span [34:73]
- [3] Span covering the alternative-positioning rationale and tick outcome (resolution: literal_explicit)
  - cite[74:140] byte-exact: 'because you could just be here and land maybe two more ticks right'
  - span [74:140]
- [4] Span covering the 'when' clause and the bad freeze frame assessment (resolution: literal_explicit)
  - cite[141:217] byte-exact: 'and on when you pick up other champions this is a extremely bad freeze frame'
  - span [141:217]
- [5] Span covering the truncated closing conditional (resolution: literal_explicit)
  - cite[218:237] byte-exact: 'because if this was'
  - span [218:237]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (3 items)
- [1] 'you': the coached player controlling Mel. (resolution: context_resolved)
  - cite[4272:4374] byte-exact: "it's a habit you don't want to have because you could just be here and land maybe two more ticks right"
- [2] Mel (champion played by the coached player). (resolution: literal_explicit)
  - cite[4189:4267] byte-exact: 'you are Mel and the main point of your queue is to proc scorch and comet right'
- [3] Other champions (hypothetical future picks referenced by the coach). (resolution: literal_explicit)
  - cite[4382:4451] byte-exact: 'when you pick up other champions this is a extremely bad freeze frame'

#### reference_bindings (5 items)
- [1] Elliptical opening subject of 'is to proc scorch and comet' binds to the immediately preceding phrase 'the main point of your queue', i.e. Mel's queue (Q). (resolution: context_resolved)
  - cite[4205:4267] byte-exact: 'the main point of your queue is to proc scorch and comet right'
- [2] 'it' in 'it's a habit you don't want to have' binds to the player's habit of stepping out of range early so the queue lands only partially (elsewhere described as a random step back after farming). (resolution: context_resolved)
  - cite[4272:4374] byte-exact: "it's a habit you don't want to have because you could just be here and land maybe two more ticks right"
  - cite[4593:4692] byte-exact: 'you do this random step back after farming instead of clicking to here and maintaining your Q range'
- [3] 'you' (in 'you don't want to have' and 'you could just be here') binds to the coached player playing Mel. (resolution: context_resolved)
  - cite[4316:4374] byte-exact: 'you could just be here and land maybe two more ticks right'
- [4] 'this' in 'this is a extremely bad freeze frame' binds to the current reviewed game frame/moment of the player's positioning. (resolution: context_resolved)
  - cite[4382:4451] byte-exact: 'when you pick up other champions this is a extremely bad freeze frame'
- [5] 'this' in 'if this was' binds forward to the transcript continuation 'Synindra Q or something else'; the surface form 'Synindra' cannot be canonically bound from the supplied vocabulary. (resolution: unresolved)
  - cite[4460:4526] byte-exact: 'if this was Synindra Q or something else maybe you miss everything'

#### abilities_resources (3 items)
- [1] Mel's queue (Q): ASR surface form 'queue' rendered for Q by surrounding context; its main point is to proc scorch and comet. (resolution: context_resolved)
  - cite[4205:4267] byte-exact: 'the main point of your queue is to proc scorch and comet right'
- [2] Scorch and Comet: effects procced by Mel's queue (runes referenced by name). (resolution: literal_explicit)
  - cite[4205:4267] byte-exact: 'the main point of your queue is to proc scorch and comet right'
- [3] 'Synindra Q': a substitute champion's Q ability invoked in the counterfactual; owner/champion unresolved due to ASR-corrupted surface form. (resolution: unresolved)
  - cite[4460:4526] byte-exact: 'if this was Synindra Q or something else maybe you miss everything'

#### events_actions (1 items)
- [1] Habitual action attributed to the player: stepping out of position early after farming, so the queue lands only partially instead of landing more ticks. (resolution: context_resolved)
  - cite[4272:4374] byte-exact: "it's a habit you don't want to have because you could just be here and land maybe two more ticks right"
  - cite[4593:4692] byte-exact: 'you do this random step back after farming instead of clicking to here and maintaining your Q range'

#### states (1 items)
- [1] Position state during the passage: the player is positioned away from the spot ('here') from which the queue would land maybe two more ticks; that spot was available. (resolution: context_resolved)
  - cite[4308:4374] byte-exact: 'because you could just be here and land maybe two more ticks right'

#### conditions (3 items)
- [1] Condition: 'when you pick up other champions', the current frame is an extremely bad freeze frame (the habit would be severely punished). (resolution: literal_explicit)
  - cite[4382:4451] byte-exact: 'when you pick up other champions this is a extremely bad freeze frame'
- [2] Counterfactual condition: 'if this was Synindra Q or something else', the queue attempt may miss everything; the substituted ability identity is unresolved. (resolution: unresolved)
  - cite[4460:4526] byte-exact: 'if this was Synindra Q or something else maybe you miss everything'
- [3] Implied condition: being/staying at the spot 'here' is the condition under which the player lands maybe two more ticks of the queue. (resolution: context_resolved)
  - cite[4316:4374] byte-exact: 'you could just be here and land maybe two more ticks right'

#### recommended_advice (2 items)
- [1] Do not keep this habit: it's a habit you don't want to have (leaving position so only a little bit of the queue lands). (resolution: literal_explicit)
  - cite[4272:4374] byte-exact: "it's a habit you don't want to have because you could just be here and land maybe two more ticks right"
- [2] Instead of stepping away, just stay at the spot ('just be here') to land maybe two more ticks. (resolution: context_resolved)
  - cite[4316:4374] byte-exact: 'you could just be here and land maybe two more ticks right'

#### consequences_outcomes (3 items)
- [1] Staying at the spot would result in landing maybe two more ticks of the queue. (resolution: context_resolved)
  - cite[4316:4374] byte-exact: 'you could just be here and land maybe two more ticks right'
- [2] When picking up other champions, this frame is an extremely bad freeze frame. (resolution: literal_explicit)
  - cite[4382:4451] byte-exact: 'when you pick up other champions this is a extremely bad freeze frame'
- [3] With the substitute champion's Q (surface form 'Synindra Q', unresolved), the player may miss everything. (resolution: unresolved)
  - cite[4460:4526] byte-exact: 'if this was Synindra Q or something else maybe you miss everything'
  - cite[4501:4526] byte-exact: 'maybe you miss everything'

#### explicit_relationships (6 items)
- [1] Mel's queue CAUSES (procs) scorch and comet. (resolution: literal_explicit; relation: CAUSES)
  - cite[4205:4267] byte-exact: 'the main point of your queue is to proc scorch and comet right'
- [2] 'you' (the coached Mel player) is the ACTOR who could be here and land maybe two more ticks. (resolution: context_resolved; relation: ACTOR)
  - cite[4316:4374] byte-exact: 'you could just be here and land maybe two more ticks right'
- [3] Picking up other champions is the CONDITION under which this frame is a extremely bad freeze frame. (resolution: literal_explicit; relation: CONDITION)
  - cite[4382:4451] byte-exact: 'when you pick up other champions this is a extremely bad freeze frame'
- [4] The habit (leaving position early) PREVENTS landing maybe two more ticks of the queue. (resolution: context_resolved; relation: PREVENTS)
  - cite[4272:4374] byte-exact: "it's a habit you don't want to have because you could just be here and land maybe two more ticks right"
- [5] 'if this was Synindra Q or something else' RESULTS in 'maybe you miss everything'; the substituted ability is unresolved (ASR-corrupted surface form). (resolution: unresolved; relation: RESULT)
  - cite[4460:4526] byte-exact: 'if this was Synindra Q or something else maybe you miss everything'
- [6] The elliptical subject of 'is to proc scorch and comet' REFERS_TO 'the main point of your queue' (Mel's queue). (resolution: context_resolved; relation: REFERS_TO)
  - cite[4205:4267] byte-exact: 'the main point of your queue is to proc scorch and comet right'

#### uncertainty_unresolved (2 items)
- [1] ASR-corrupted champion surface form 'Synindra' in 'if this was Synindra Q or something else'; the supplied lexical vocabulary contains no matching canonical champion or alias, so the intended champion/ability binding remains unresolved. (resolution: unresolved)
  - cite[4460:4526] byte-exact: 'if this was Synindra Q or something else maybe you miss everything'
- [2] Garbled connective span 'right and on when you pick up other champions': 'and on' is an unrecoverable ASR/discourse artifact; only the temporal condition content is recoverable. (resolution: unresolved)
  - cite[4369:4451] byte-exact: 'right and on when you pick up other champions this is a extremely bad freeze frame'

#### supporting_source_spans (4 items)
- [1] Establishes the queue's purpose and the habit warning. (resolution: literal_explicit)
  - cite[4205:4307] byte-exact: "the main point of your queue is to proc scorch and comet right but it's a habit you don't want to have"
  - span [4205:4307]
- [2] Establishes the alternative of landing more ticks by staying in position. (resolution: literal_explicit)
  - cite[4308:4374] byte-exact: 'because you could just be here and land maybe two more ticks right'
  - span [4308:4374]
- [3] Establishes the other-champions condition and the freeze-frame evaluation. (resolution: literal_explicit)
  - cite[4382:4451] byte-exact: 'when you pick up other champions this is a extremely bad freeze frame'
  - span [4382:4451]
- [4] Establishes the counterfactual miss-everything consequence. (resolution: literal_explicit)
  - cite[4460:4526] byte-exact: 'if this was Synindra Q or something else maybe you miss everything'
  - span [4460:4526]

Raw response binding A: `881d29233dd0aca29740807b305516fdcfa66c92d6fa0825156475d54334cbd7`
Raw response binding B: `0065766a9ec6819c60fc3c256b5769e94a9f0550202438d1005b13f3e7c0178f`

---

## TARGET p2k:case:0007 (selection rank 7)

### Exact Bronze

```text
now next time they give you this all in here on the freeze then you would ignite twoo right so why not do it now when they're going to probably potion and maybe Nami will heal the sa you know like I don't know if she has W
```

Location in full transcript: [9536:9758]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (3 items)
- [1] "you" - the coached player being addressed (Lucian ADC per supplied metadata) (resolution: context_resolved)
  - cite[14:58] byte-exact: 'they give you this all in here on the freeze'
  - cite[59:91] byte-exact: 'then you would ignite twoo right'
- [2] "they" / "they're" - the opposing player(s) in the lane (resolution: context_resolved)
  - cite[0:58] byte-exact: 'now next time they give you this all in here on the freeze'
  - cite[113:150] byte-exact: "when they're going to probably potion"
- [3] Nami - champion named literally; team affiliation is not stated in the supplied material (resolution: literal_explicit)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'

#### reference_bindings (5 items)
- [1] "this" binds to the all-in situation occurring on the freeze that is being reviewed (resolution: context_resolved)
  - cite[24:58] byte-exact: 'you this all in here on the freeze'
- [2] "it" in "why not do it now" binds to the just-described ignite play during the all-in (resolution: context_resolved)
  - cite[92:112] byte-exact: 'so why not do it now'
  - cite[59:91] byte-exact: 'then you would ignite twoo right'
- [3] "she" binds to Nami (nearest named female champion entity, who is said to possibly heal) (resolution: context_resolved)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'
  - cite[197:222] byte-exact: "I don't know if she has W"
- [4] "the sa" - corrupted noun phrase; intended referent cannot be recovered from supplied material (resolution: unresolved)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'
- [5] "W" binds to an ability key belonging to "she" (Nami) via "she has W"; W is a listed ability key in the supplied vocabulary (resolution: vocabulary_supported)
  - cite[197:222] byte-exact: "I don't know if she has W"

#### abilities_resources (3 items)
- [1] Ignite - summoner spell referenced as something the coached player would use; Ignite appears in the supplied summoner_spells vocabulary (resolution: vocabulary_supported)
  - cite[59:91] byte-exact: 'then you would ignite twoo right'
- [2] W - ability key referenced for Nami via "she has W"; W is listed in the supplied ability_keys vocabulary and Nami's W slot exists in the supplied champion_abilities data (resolution: vocabulary_supported)
  - cite[197:222] byte-exact: "I don't know if she has W"
- [3] potion - consumable item/resource the opponents are expected to use (used verbally: "probably potion") (resolution: literal_explicit)
  - cite[113:150] byte-exact: "when they're going to probably potion"

#### events_actions (3 items)
- [1] The opponents give the coached player an all-in opportunity while the wave is frozen; framed as a recurring scenario ("next time they give you this") (resolution: literal_explicit)
  - cite[0:58] byte-exact: 'now next time they give you this all in here on the freeze'
  - cite[45:58] byte-exact: 'on the freeze'
- [2] Anticipated action: the opponents are expected to use a potion ("going to probably potion") (resolution: literal_explicit)
  - cite[113:150] byte-exact: "when they're going to probably potion"
- [3] Anticipated possible action: Nami may heal (target noun phrase "the sa" corrupted; referent unrecoverable) (resolution: literal_explicit)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'

#### states (2 items)
- [1] A freeze (frozen wave state) is present in the lane at the reviewed moment ("on the freeze") (resolution: context_resolved)
  - cite[45:58] byte-exact: 'on the freeze'
- [2] Whether Nami currently has her W available is unknown - the speaker explicitly does not know (resolution: unresolved)
  - cite[197:222] byte-exact: "I don't know if she has W"

#### conditions (2 items)
- [1] Condition for the ignite advice: next time the opponents again give an all-in opportunity on the freeze (resolution: literal_explicit)
  - cite[0:58] byte-exact: 'now next time they give you this all in here on the freeze'
- [2] Condition attached to doing it now: the moment when the opponents are probably going to potion (resolution: literal_explicit)
  - cite[92:150] byte-exact: "so why not do it now when they're going to probably potion"

#### recommended_advice (2 items)
- [1] When this all-in-on-the-freeze situation happens again ("next time"), the player would use Ignite as well (final token corrupted: "twoo", plausibly "too") (resolution: context_resolved)
  - cite[0:91] byte-exact: 'now next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] Coach urges doing it (the ignite/all-in play) now rather than waiting, posed rhetorically as "why not do it now" (resolution: context_resolved)
  - cite[92:112] byte-exact: 'so why not do it now'

#### consequences_outcomes (2 items)
- [1] Expected near-term outcome: the opponents will probably use a potion (resolution: literal_explicit)
  - cite[113:150] byte-exact: "when they're going to probably potion"
- [2] Possible outcome: Nami may heal the entity referred to by the corrupted phrase "the sa" (resolution: literal_explicit)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'

#### explicit_relationships (9 items)
- [1] ACTOR: "they" (opponents) -> the all-in event given to the player on the freeze (resolution: literal_explicit; relation: ACTOR)
  - cite[14:58] byte-exact: 'they give you this all in here on the freeze'
- [2] ACTOR: "you" (coached player) -> Ignite usage in the all-in scenario (resolution: context_resolved; relation: ACTOR)
  - cite[59:91] byte-exact: 'then you would ignite twoo right'
- [3] ACTOR: "they" (opponents) -> potion usage (resolution: literal_explicit; relation: ACTOR)
  - cite[113:150] byte-exact: "when they're going to probably potion"
- [4] ACTOR: Nami -> healing event (resolution: literal_explicit; relation: ACTOR)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'
- [5] TARGET: "the sa" (corrupted referent) <- the healing event performed by Nami (resolution: unresolved; relation: TARGET)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'
- [6] REFERS_TO: "she" -> Nami (resolution: context_resolved; relation: REFERS_TO)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'
  - cite[197:222] byte-exact: "I don't know if she has W"
- [7] USES/possession: "she" (Nami) <-> her W ability; possession asserted conditionally since the speaker does not know if it is held (resolution: context_resolved; relation: USES)
  - cite[197:222] byte-exact: "I don't know if she has W"
- [8] CONDITION: recurrence of the all-in-on-freeze situation conditions using Ignite ("then you would ignite ...)") (resolution: context_resolved; relation: CONDITION)
  - cite[0:91] byte-exact: 'now next time they give you this all in here on the freeze then you would ignite twoo right'
- [9] CONDITION: imminent opponent potion usage conditions the proposal to do it now (resolution: literal_explicit; relation: CONDITION)
  - cite[92:150] byte-exact: "so why not do it now when they're going to probably potion"

#### uncertainty_unresolved (4 items)
- [1] ASR-corrupted token "twoo" immediately after "ignite"; no supplied alias or vocabulary rule licenses a repair (plausible reading "too" remains unverified), so recorded unresolved (resolution: unresolved)
  - cite[64:91] byte-exact: 'you would ignite twoo right'
- [2] Truncated/corrupted noun phrase "the sa" after "heal"; intended referent cannot be recovered from supplied material (resolution: unresolved)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'
- [3] Availability of Nami's W cannot be determined; the speaker states he does not know if she has it (resolution: unresolved)
  - cite[197:222] byte-exact: "I don't know if she has W"
- [4] Nami's team affiliation (ally vs opponent) is never stated in the passage; grouping with the opponents' potion usage suggests but does not establish she opposes the player (resolution: unresolved)
  - cite[151:182] byte-exact: 'and maybe Nami will heal the sa'

#### supporting_source_spans (5 items)
- [1] Span covering the conditional ignite advice for the recurring all-in-on-freeze scenario (resolution: literal_explicit)
  - cite[0:91] byte-exact: 'now next time they give you this all in here on the freeze then you would ignite twoo right'
  - span [0:91]
- [2] Span covering the coach's challenge to perform the play immediately (resolution: literal_explicit)
  - cite[92:112] byte-exact: 'so why not do it now'
  - span [92:112]
- [3] Span covering the predicted opponent potion usage conditioning the timing (resolution: literal_explicit)
  - cite[113:150] byte-exact: "when they're going to probably potion"
  - span [113:150]
- [4] Span covering the possible Nami heal with its corrupted target phrase (resolution: literal_explicit)
  - cite[155:182] byte-exact: 'maybe Nami will heal the sa'
  - span [155:182]
- [5] Span covering the speaker's stated uncertainty about Nami's W availability (resolution: literal_explicit)
  - cite[197:222] byte-exact: "I don't know if she has W"
  - span [197:222]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (4 items)
- [1] "you" – the coached player reviewing the replay (Lucian ADC per supplied metadata), addressed by the coach. (resolution: context_resolved)
  - cite[9536:9594] byte-exact: 'now next time they give you this all in here on the freeze'
- [2] "they" – the opposing bot-lane side (enemy players/champions) who would 'give' the all-in on the freeze. (resolution: context_resolved)
  - cite[9540:9594] byte-exact: 'next time they give you this all in here on the freeze'
- [3] Nami – enemy champion explicitly named; expected to possibly heal her ally. (resolution: literal_explicit)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'
- [4] "the sa" – an enemy champion allied with Nami (the heal target); surface form 'sa' is ASR-corrupted and its champion identity is not recoverable from the supplied material. (resolution: unresolved)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'

#### reference_bindings (8 items)
- [1] Pronoun "you" binds to the reviewed player (the Lucian ADC being coached). (resolution: context_resolved)
  - cite[9540:9594] byte-exact: 'next time they give you this all in here on the freeze'
- [2] Pronoun "they" in 'next time they give you' binds to the enemy bot-lane duo. (resolution: context_resolved)
  - cite[9536:9594] byte-exact: 'now next time they give you this all in here on the freeze'
- [3] Pronoun "they" in 'when they're going to probably potion' binds to the same enemy bot-lane duo. (resolution: context_resolved)
  - cite[9649:9686] byte-exact: "when they're going to probably potion"
- [4] Deictic "this all in" refers to the all-in opportunity currently under discussion on the frozen wave (anchored to the reviewed footage). (resolution: context_resolved)
  - cite[9564:9594] byte-exact: 'this all in here on the freeze'
- [5] Deictic "here" refers to the current lane/freeze position shown in the replay at this moment of the review. (resolution: context_resolved)
  - cite[9564:9594] byte-exact: 'this all in here on the freeze'
- [6] Pronoun "it" in 'so why not do it now' refers to using Ignite in the current all-in. (resolution: context_resolved)
  - cite[9628:9648] byte-exact: 'so why not do it now'
- [7] Reference "the sa" cannot be fully resolved: ASR-corrupted name of Nami's ally; only its team-side binding (enemy ally of Nami) is supported. (resolution: unresolved)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'
- [8] Pronoun "she" in 'I don't know if she has W' binds to Nami, the nearest female-champion antecedent; confirmed by the immediately following coach confirmation that she does have it. (resolution: context_resolved)
  - cite[9691:9758] byte-exact: "maybe Nami will heal the sa you know like I don't know if she has W"
  - cite[9759:9851] byte-exact: 'up but she should yeah she has dou up see so now she just healed more and you do less damage'

#### abilities_resources (4 items)
- [1] Ignite – summoner spell belonging to the coached player ('you'), referenced as the action he would/could use ('ignite'). (resolution: literal_explicit)
  - cite[9595:9627] byte-exact: 'then you would ignite twoo right'
  - cite[9628:9648] byte-exact: 'so why not do it now'
- [2] Potion – consumable resource expected to be used by the enemies ('they') during/after the fight. (resolution: literal_explicit)
  - cite[9649:9686] byte-exact: "when they're going to probably potion"
- [3] W – ability slot owned by Nami; its current availability is expressed as uncertain ('I don't know if she has W'). Ownership resolved via pronoun 'she' = Nami. (resolution: context_resolved)
  - cite[9691:9758] byte-exact: "maybe Nami will heal the sa you know like I don't know if she has W"
- [4] Healing by Nami directed at her ally ('heal the sa'); stated as a verb action, not tied in the passage to a specific named ability. (resolution: literal_explicit)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'

#### events_actions (3 items)
- [1] Anticipated conditional event: next time the enemies allow the all-in opportunity on the frozen wave, the player would also use Ignite (surface token 'twoo' corrupted). (resolution: literal_explicit)
  - cite[9536:9627] byte-exact: 'now next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] Expected enemy response if the fight happens now: they will probably use potions. (resolution: literal_explicit)
  - cite[9649:9686] byte-exact: "when they're going to probably potion"
- [3] Possible event raised by the coach: Nami may heal 'the sa'. (resolution: literal_explicit)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'

#### states (3 items)
- [1] Wave state during the discussed scenario: the minion wave is held as a freeze ('on the freeze'). (resolution: literal_explicit)
  - cite[9564:9594] byte-exact: 'this all in here on the freeze'
- [2] Uncertain resource state: whether Nami currently has her W available is unknown ('I don't know if she has W'); left unresolved within the passage. (resolution: unresolved)
  - cite[9733:9758] byte-exact: "I don't know if she has W"
- [3] Present-moment state implied by the rhetorical question: an all-in opportunity exists right now that could be converted with Ignite ('do it now'). (resolution: context_resolved)
  - cite[9628:9648] byte-exact: 'so why not do it now'

#### conditions (3 items)
- [1] 'next time they give you this all in here on the freeze' – condition under which the hypothetical future Ignite usage holds (if the enemies again grant the all-in on the freeze). (resolution: literal_explicit)
  - cite[9536:9594] byte-exact: 'now next time they give you this all in here on the freeze'
- [2] 'when they're going to probably potion' – timing condition framing the decision to use Ignite now rather than later. (resolution: literal_explicit)
  - cite[9628:9686] byte-exact: "so why not do it now when they're going to probably potion"
- [3] 'maybe Nami will heal the sa' – possible enemy-sustain condition attached to fighting now. (resolution: literal_explicit)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'

#### recommended_advice (2 items)
- [1] Coach recommends using Ignite now in the current all-in instead of saving it for a hypothetical future all-in on the freeze. (resolution: literal_explicit)
  - cite[9595:9648] byte-exact: 'then you would ignite twoo right so why not do it now'
- [2] Supporting reasoning given by the coach for spending Ignite immediately: holding it has no point when an insane trade with First Strike is happening anyway (immediately preceding sentence). (resolution: context_resolved)
  - cite[9410:9517] byte-exact: "there's no point to hold your ignite here when you're going to do an insane trade with First Strike as well"

#### consequences_outcomes (3 items)
- [1] If the fight is taken now, the enemies are expected to respond by using potions. (resolution: literal_explicit)
  - cite[9649:9686] byte-exact: "when they're going to probably potion"
- [2] If the fight is taken now, possibly Nami heals her ally, mitigating the damage dealt. (resolution: literal_explicit)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'
- [3] Implied outcome of waiting versus acting now: enemy sustain responses (potions/heals) reduce the value of the Ignite/trade, which is why the coach argues for igniting now; made explicit by the coach's follow-up that after the heal 'you do less damage'. (resolution: context_resolved)
  - cite[9649:9718] byte-exact: "when they're going to probably potion and maybe Nami will heal the sa"
  - cite[9782:9851] byte-exact: 'she has dou up see so now she just healed more and you do less damage'

#### explicit_relationships (8 items)
- [1] The coached player ('you') USES Ignite in the anticipated all-in scenario. (resolution: vocabulary_supported; relation: USES)
  - cite[9595:9627] byte-exact: 'then you would ignite twoo right'
- [2] Event 'next time they give you this all in here on the freeze' is the CONDITION for the event 'you would ignite too'. (resolution: literal_explicit; relation: CONDITION)
  - cite[9536:9627] byte-exact: 'now next time they give you this all in here on the freeze then you would ignite twoo right'
- [3] 'it' REFERS_TO using Ignite in the current moment. (resolution: context_resolved; relation: REFERS_TO)
  - cite[9628:9648] byte-exact: 'so why not do it now'
- [4] 'they' ACTOR of the anticipated event of using potions. (resolution: literal_explicit; relation: ACTOR)
  - cite[9649:9686] byte-exact: "when they're going to probably potion"
- [5] Nami ACTOR of the possible healing event. (resolution: literal_explicit; relation: ACTOR)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'
- [6] 'the sa' TARGET of Nami's healing event. (resolution: unresolved; relation: TARGET)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'
- [7] 'she' REFERS_TO Nami (owner of the uncertain W availability). (resolution: context_resolved; relation: REFERS_TO)
  - cite[9691:9758] byte-exact: "maybe Nami will heal the sa you know like I don't know if she has W"
- [8] The expectation that enemies potion AFFECTS the decision of whether to spend Ignite now. (resolution: context_resolved; relation: AFFECTS)
  - cite[9628:9686] byte-exact: "so why not do it now when they're going to probably potion"

#### uncertainty_unresolved (4 items)
- [1] ASR-corrupted token 'twoo' in 'then you would ignite twoo right'; plausibly 'too' but not recoverable from the supplied vocabulary or context; recorded unresolved. (resolution: unresolved)
  - cite[9595:9627] byte-exact: 'then you would ignite twoo right'
- [2] ASR-corrupted champion reference 'the sa': identity unrecoverable; bound only contextually as an enemy ally of Nami. (resolution: unresolved)
  - cite[9691:9718] byte-exact: 'maybe Nami will heal the sa'
- [3] Deictic items 'this all in' and double 'here' depend on replay visuals not present in the text; resolution to the current freeze/all-in situation is contextual and approximate. (resolution: unresolved)
  - cite[9564:9594] byte-exact: 'this all in here on the freeze'
- [4] Coach's own uncertainty preserved: he states he does not know whether Nami's W is available ('I don't know if she has W'). (resolution: unresolved)
  - cite[9733:9758] byte-exact: "I don't know if she has W"

#### supporting_source_spans (3 items)
- [1] Primary span: the full target passage as it occurs verbatim in the transcript. (resolution: literal_explicit)
  - cite[9536:9758] byte-exact: 'now next time they give you this all in here on the freeze then you would ignite twoo right so why not do it now when...'
  - span [9536:9758]
- [2] Preceding-context span motivating the advice to spend Ignite immediately. (resolution: context_resolved)
  - cite[9410:9517] byte-exact: "there's no point to hold your ignite here when you're going to do an insane trade with First Strike as well"
  - span [9410:9517]
- [3] Following-context span confirming Nami had the W/heal available and that the heal reduced damage dealt. (resolution: context_resolved)
  - cite[9759:9851] byte-exact: 'up but she should yeah she has dou up see so now she just healed more and you do less damage'
  - span [9759:9851]

Raw response binding A: `b4474cafa372ca511e7df85601c866e331eb17b82a53eb578603f7c1cc2f9a07`
Raw response binding B: `d2d52ae42c7e641aa80d8b07a398d951b7f9d01d0cae0f17bea4ad881046fb53`

---

## TARGET p2k:case:0008 (selection rank 8)

### Exact Bronze

```text
than Riven. >> So that's already a good reason to ban it. But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure. Yeah, I might have to do that because against Riven like if
```

Location in full transcript: [75938:76169]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (2 items)
- [1] Riven, champion referenced in the ban discussion and in the student's stated matchup (resolution: literal_explicit)
  - cite[0:11] byte-exact: 'than Riven.'
  - cite[210:231] byte-exact: 'against Riven like if'
- [2] "you" / "I": the coached player being advised by the coach about the ban decision (resolution: context_resolved)
  - cite[81:131] byte-exact: "you feel like you're not able to progress the lane"
  - cite[172:231] byte-exact: 'Yeah, I might have to do that because against Riven like if'

#### reference_bindings (5 items)
- [1] "it" in "already a good reason to ban it" binds to Riven, supported by the student tying the contemplated ban to playing against Riven (resolution: context_resolved)
  - cite[18:56] byte-exact: "that's already a good reason to ban it"
  - cite[202:231] byte-exact: 'because against Riven like if'
- [2] "it" in "you should ban it for sure" binds to Riven under the same contextual support (resolution: context_resolved)
  - cite[133:170] byte-exact: 'then yeah, you should ban it for sure'
  - cite[210:231] byte-exact: 'against Riven like if'
- [3] "that" in "I might have to do that" binds to banning Riven as just advised (resolution: context_resolved)
  - cite[172:231] byte-exact: 'Yeah, I might have to do that because against Riven like if'
  - cite[144:170] byte-exact: 'you should ban it for sure'
- [4] "that" in "on top of that" refers to an additional reason or fact stated earlier, whose content lies before the supplied passage and cannot be recovered (resolution: unresolved)
  - cite[58:131] byte-exact: "But if on top of that, you feel like you're not able to progress the lane"
- [5] "that's" in "So that's already a good reason to ban it" refers to the immediately preceding statement, truncated before this passage; only its comparison involving Riven is supplied (resolution: unresolved)
  - cite[15:57] byte-exact: "So that's already a good reason to ban it."
  - cite[0:11] byte-exact: 'than Riven.'

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (2 items)
- [1] Perceived lane state under discussion: the player feeling unable to progress the lane (raised hypothetically within the advice) (resolution: literal_explicit)
  - cite[81:131] byte-exact: "you feel like you're not able to progress the lane"
- [2] The player is facing or expects to face Riven in lane, which motivates the ban consideration (resolution: context_resolved)
  - cite[202:231] byte-exact: 'because against Riven like if'

#### conditions (2 items)
- [1] Condition for the firm ban advice: on top of the previously stated reason, the player feels unable to progress the lane (resolution: literal_explicit)
  - cite[58:131] byte-exact: "But if on top of that, you feel like you're not able to progress the lane"
- [2] A prior reason was already established before this passage began ('already'), forming part of the grounds for banning (resolution: context_resolved)
  - cite[15:57] byte-exact: "So that's already a good reason to ban it."
  - cite[0:57] byte-exact: "than Riven. >> So that's already a good reason to ban it."

#### recommended_advice (2 items)
- [1] Coach advises: if, in addition to the existing reason, the player feels unable to progress the lane, they should ban Riven for sure (resolution: literal_explicit)
  - cite[58:170] byte-exact: "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure"
- [2] Coach states the previously made point is already by itself a good reason to ban (resolution: literal_explicit)
  - cite[15:57] byte-exact: "So that's already a good reason to ban it."

#### consequences_outcomes (2 items)
- [1] Feared negative outcome raised in the condition: the player not being able to progress the lane in the matchup (resolution: literal_explicit)
  - cite[81:131] byte-exact: "you feel like you're not able to progress the lane"
- [2] Possible outcome of accepting the advice: the student would carry out the ban, which he acknowledges he may have to do (resolution: literal_explicit)
  - cite[172:231] byte-exact: 'Yeah, I might have to do that because against Riven like if'

#### explicit_relationships (5 items)
- [1] CONDITION(feeling unable to progress the lane, banning Riven): the 'ban it for sure' recommendation holds if the player additionally feels unable to progress the lane (resolution: vocabulary_supported; relation: CONDITION)
  - cite[58:170] byte-exact: "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure"
- [2] TARGET(ban action, Riven): the champion to be banned is Riven (resolution: vocabulary_supported; relation: TARGET)
  - cite[144:170] byte-exact: 'you should ban it for sure'
  - cite[210:231] byte-exact: 'against Riven like if'
- [3] REFERS_TO("it" in "ban it", Riven) (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[18:56] byte-exact: "that's already a good reason to ban it"
  - cite[202:231] byte-exact: 'because against Riven like if'
- [4] REFERS_TO("that" in "I might have to do that", the advised ban of Riven) (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[172:231] byte-exact: 'Yeah, I might have to do that because against Riven like if'
  - cite[144:170] byte-exact: 'you should ban it for sure'
- [5] CAUSES(previously stated point, ban consideration): the coach calls the earlier point already a good reason to ban (resolution: vocabulary_supported; relation: CAUSES)
  - cite[15:57] byte-exact: "So that's already a good reason to ban it."

#### uncertainty_unresolved (3 items)
- [1] Passage opens mid-comparison with "than Riven."; what was compared to Riven, and the full statement the coach calls 'already a good reason', lie outside the supplied passage (resolution: unresolved)
  - cite[0:57] byte-exact: "than Riven. >> So that's already a good reason to ban it."
- [2] Student's utterance cuts off after "like if"; whatever condition or reasoning follows the "if" is unrecoverable from the supplied material (resolution: unresolved)
  - cite[202:231] byte-exact: 'because against Riven like if'
- [3] Binding of "on top of that" cannot be completed: it points to an additional earlier reason whose content is not supplied beyond the truncated fragment (resolution: unresolved)
  - cite[58:131] byte-exact: "But if on top of that, you feel like you're not able to progress the lane"

#### supporting_source_spans (3 items)
- [1] Coach: the earlier point is already sufficient reason to ban (resolution: literal_explicit)
  - cite[15:57] byte-exact: "So that's already a good reason to ban it."
  - span [15:57]
- [2] Conditional advice: if also unable to progress the lane, ban it for sure (resolution: literal_explicit)
  - cite[58:171] byte-exact: "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure."
  - span [58:171]
- [3] Student agreement citing the matchup against Riven, truncated mid-sentence (resolution: literal_explicit)
  - cite[172:231] byte-exact: 'Yeah, I might have to do that because against Riven like if'
  - span [172:231]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (4 items)
- [1] Riven (champion named in the comparison and in the student's stated reason) (resolution: literal_explicit)
  - cite[75938:75949] byte-exact: 'than Riven.'
  - cite[76140:76169] byte-exact: 'because against Riven like if'
- [2] Ambessa (champion; referent of 'it'/'that's' in the ban recommendation, bound via the immediately preceding source sentence 'Ambessa is by far better champion than Riven.') (resolution: context_resolved)
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'
  - cite[75956:75994] byte-exact: "that's already a good reason to ban it"
- [3] The coached player (student being advised; addressee 'you' and speaker 'I' in the reply; Fiora top player per metadata) (resolution: context_resolved)
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'
  - cite[76110:76169] byte-exact: 'Yeah, I might have to do that because against Riven like if'
- [4] The coach (advice giver issuing the ban recommendation) (resolution: context_resolved)
  - cite[75956:75994] byte-exact: "that's already a good reason to ban it"
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'

#### reference_bindings (7 items)
- [1] 'that's' in 'So that's already a good reason to ban it' -> the immediately preceding claim that Ambessa is by far a better champion than Riven (resolution: context_resolved)
  - cite[75956:75994] byte-exact: "that's already a good reason to ban it"
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'
  - cite[75938:75949] byte-exact: 'than Riven.'
- [2] 'it' in 'to ban it' -> Ambessa (resolution: context_resolved)
  - cite[75973:75994] byte-exact: 'good reason to ban it'
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'
- [3] 'it' in 'you should ban it for sure' -> Ambessa (resolution: context_resolved)
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'
- [4] 'you' in 'you feel like you're not able to progress the lane' -> the coached player (resolution: context_resolved)
  - cite[76019:76069] byte-exact: "you feel like you're not able to progress the lane"
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'
- [5] 'you' in 'you should ban it for sure' -> the coached player (resolution: context_resolved)
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'
- [6] 'I' in 'Yeah, I might have to do that' -> the coached player (student replying to the advice) (resolution: context_resolved)
  - cite[76110:76169] byte-exact: 'Yeah, I might have to do that because against Riven like if'
- [7] 'that' in 'I might have to do that' -> banning Ambessa (the action just recommended) (resolution: context_resolved)
  - cite[76118:76139] byte-exact: 'might have to do that'
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'

#### abilities_resources (1 items)
- [1] Champion ban (draft resource/action), directed at Ambessa (resolution: literal_explicit)
  - cite[75973:75994] byte-exact: 'good reason to ban it'
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'

#### events_actions (0 items)

(none)

#### states (0 items)

(none)

#### conditions (2 items)
- [1] Condition for the strengthened ban recommendation: if, on top of Ambessa being the better champion, the player feels unable to progress the lane (resolution: literal_explicit)
  - cite[76000:76069] byte-exact: "if on top of that, you feel like you're not able to progress the lane"
- [2] Truncated causal/conditional clause closing the passage: the student says he might ban Ambessa 'because against Riven like if', completed immediately after the target span as 'they all in me' (resolution: context_resolved)
  - cite[76140:76169] byte-exact: 'because against Riven like if'
  - cite[76170:76184] byte-exact: 'they all in me'

#### recommended_advice (2 items)
- [1] Coach: ban Ambessa - her being by far a better champion than Riven is already a good reason to ban her (resolution: literal_explicit)
  - cite[75956:75994] byte-exact: "that's already a good reason to ban it"
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'
  - cite[75938:75949] byte-exact: 'than Riven.'
- [2] Coach: if on top of that you feel like you're not able to progress the lane, then you should ban Ambessa for sure (resolution: literal_explicit)
  - cite[76000:76069] byte-exact: "if on top of that, you feel like you're not able to progress the lane"
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'

#### consequences_outcomes (2 items)
- [1] Outcome of the advice: the student says he might have to do it (ban Ambessa) (resolution: literal_explicit)
  - cite[76110:76169] byte-exact: 'Yeah, I might have to do that because against Riven like if'
- [2] Conditional consequence: feeling unable to progress the lane makes banning Ambessa clearly warranted ('then yeah, you should ban it for sure') (resolution: literal_explicit)
  - cite[76000:76069] byte-exact: "if on top of that, you feel like you're not able to progress the lane"
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'

#### explicit_relationships (6 items)
- [1] 'it' (object of 'to ban it') refers to Ambessa (resolution: context_resolved; relation: REFERS_TO)
  - cite[75973:75994] byte-exact: 'good reason to ban it'
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'
- [2] 'that's' refers to the claim that Ambessa is by far a better champion than Riven (resolution: context_resolved; relation: REFERS_TO)
  - cite[75956:75994] byte-exact: "that's already a good reason to ban it"
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'
  - cite[75938:75949] byte-exact: 'than Riven.'
- [3] 'that' (in 'I might have to do that') refers to banning Ambessa (resolution: context_resolved; relation: REFERS_TO)
  - cite[76118:76139] byte-exact: 'might have to do that'
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'
- [4] 'you feel like you're not able to progress the lane' is the condition under which 'you should ban it for sure' holds (resolution: literal_explicit; relation: CONDITION)
  - cite[76000:76069] byte-exact: "if on top of that, you feel like you're not able to progress the lane"
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'
- [5] Playing against Riven (with the truncated trigger 'if they all in me') is the stated cause of the student considering the Ambessa ban (resolution: context_resolved; relation: CAUSES)
  - cite[76140:76169] byte-exact: 'because against Riven like if'
  - cite[76170:76184] byte-exact: 'they all in me'
  - cite[76110:76169] byte-exact: 'Yeah, I might have to do that because against Riven like if'
- [6] Ambessa is the target of the recommended ban action (resolution: context_resolved; relation: TARGET)
  - cite[75973:75994] byte-exact: 'good reason to ban it'
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'

#### uncertainty_unresolved (2 items)
- [1] The target passage ends mid-sentence at 'because against Riven like if'; the conditional remains open within the target passage, and its continuation ('they all in me...') is recoverable only from text beyond the target span (resolution: unresolved)
  - cite[76140:76169] byte-exact: 'because against Riven like if'
  - cite[76170:76184] byte-exact: 'they all in me'
- [2] The passage opens mid-sentence at 'than Riven.'; the compared subject (Ambessa, described as 'by far better champion') precedes the target span and is recovered from the immediately preceding source sentence (resolution: context_resolved)
  - cite[75938:75949] byte-exact: 'than Riven.'
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'

#### supporting_source_spans (7 items)
- [1] 'Ambessa is by far better champion' (preceding context establishing the comparison) (resolution: context_resolved)
  - cite[75904:75937] byte-exact: 'Ambessa is by far better champion'
  - span [75904:75937]
- [2] 'than Riven.' (resolution: literal_explicit)
  - cite[75938:75949] byte-exact: 'than Riven.'
  - span [75938:75949]
- [3] 'that's already a good reason to ban it' (resolution: literal_explicit)
  - cite[75956:75994] byte-exact: "that's already a good reason to ban it"
  - span [75956:75994]
- [4] 'if on top of that, you feel like you're not able to progress the lane' (resolution: literal_explicit)
  - cite[76000:76069] byte-exact: "if on top of that, you feel like you're not able to progress the lane"
  - span [76000:76069]
- [5] 'then yeah, you should ban it for sure' (resolution: literal_explicit)
  - cite[76071:76108] byte-exact: 'then yeah, you should ban it for sure'
  - span [76071:76108]
- [6] 'Yeah, I might have to do that because against Riven like if' (resolution: literal_explicit)
  - cite[76110:76169] byte-exact: 'Yeah, I might have to do that because against Riven like if'
  - span [76110:76169]
- [7] 'they all in me' (continuation completing the truncated final clause, adjacent to the target span) (resolution: context_resolved)
  - cite[76170:76184] byte-exact: 'they all in me'
  - span [76170:76184]

Raw response binding A: `32dd35747f31b0b33f006855e57510a7785a802291142f412739da6dda6053c9`
Raw response binding B: `afd91119f2779ba673b9c7cdf00cdcdc38115bb41b83fb12493de045ab60a2e1`

---

## TARGET p2k:case:0009 (selection rank 9)

### Exact Bronze

```text
then it's bot. Yes? Yeah. She's not a bot, because well, obviously, she's not killing Poppy. She's not on mid, because wave equals ward. But you don't play around waves being your ward. So, you chase. If Mel is showing on mid, this could be a good flip,
```

Location in full transcript: [56538:56791]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (4 items)
- [1] An unnamed female enemy champion referred to as 'she', whose lane whereabouts are being deduced; her identity is not given in the passage (resolution: unresolved)
  - cite[26:41] byte-exact: "She's not a bot"
  - cite[93:109] byte-exact: "She's not on mid"
- [2] Poppy, named as the champion she is statedly not killing (resolution: literal_explicit)
  - cite[68:91] byte-exact: "she's not killing Poppy"
- [3] Mel, named as the champion whose possibly showing on mid is raised (resolution: literal_explicit)
  - cite[201:225] byte-exact: 'If Mel is showing on mid'
- [4] 'you': the coached player being addressed and advised (supplied metadata frames the session as Jinx ADC coaching) (resolution: context_resolved)
  - cite[186:200] byte-exact: 'So, you chase.'
  - cite[137:185] byte-exact: "But you don't play around waves being your ward."

#### reference_bindings (5 items)
- [1] 'it' in 'then it's bot' binds to the tracked female champion's whereabouts (the same entity later called 'she'); the champion herself remains unidentified (resolution: context_resolved)
  - cite[0:14] byte-exact: "then it's bot."
- [2] 'She's' in 'She's not a bot' binds anaphorically to the tracked female enemy champion under discussion; her specific identity is unresolved (resolution: unresolved)
  - cite[26:42] byte-exact: "She's not a bot,"
- [3] 'she's' in 'she's not killing Poppy' binds to the same tracked female enemy champion as the immediately preceding 'She' (resolution: context_resolved)
  - cite[57:91] byte-exact: "obviously, she's not killing Poppy"
- [4] 'your' in 'waves being your ward' binds to the coached player addressed as 'you' (resolution: context_resolved)
  - cite[163:184] byte-exact: 'waves being your ward'
- [5] 'this' in 'this could be a good flip' refers to the prospective opportunity arising if Mel is showing on mid, in the chase scenario just urged (resolution: context_resolved)
  - cite[201:253] byte-exact: 'If Mel is showing on mid, this could be a good flip,'

#### abilities_resources (1 items)
- [1] A ward invoked figuratively: the coach speaks of waves serving as 'your ward' while stating you don't play around that; no actual ward placement is described (resolution: literal_explicit)
  - cite[163:184] byte-exact: 'waves being your ward'

#### events_actions (4 items)
- [1] Deduction delivered aloud: the tracked champion's whereabouts are concluded to be bot (resolution: literal_explicit)
  - cite[0:14] byte-exact: "then it's bot."
- [2] Negated event: she is not killing Poppy, offered as the obvious ground for ruling out bot (resolution: literal_explicit)
  - cite[57:91] byte-exact: "obviously, she's not killing Poppy"
- [3] Hypothetical event raised under an if-clause: Mel is showing on mid (resolution: literal_explicit)
  - cite[201:225] byte-exact: 'If Mel is showing on mid'
- [4] Stated general behavior claim: you don't play around waves being your ward (resolution: literal_explicit)
  - cite[137:185] byte-exact: "But you don't play around waves being your ward."

#### states (2 items)
- [1] Location state asserted by elimination: the tracked champion is taken to be (heading) bot (resolution: literal_explicit)
  - cite[0:14] byte-exact: "then it's bot."
- [2] Negated location state: she is not on mid (resolution: literal_explicit)
  - cite[93:109] byte-exact: "She's not on mid"

#### conditions (3 items)
- [1] Condition for the flip assessment: if Mel is showing on mid, the situation could be a good flip (resolution: literal_explicit)
  - cite[201:225] byte-exact: 'If Mel is showing on mid'
- [2] Stated premise 'wave equals ward' under which 'She's not on mid' is inferred (resolution: literal_explicit)
  - cite[111:135] byte-exact: 'because wave equals ward'
- [3] Stated premise that she's obviously not killing Poppy, under which 'She's not a bot' is inferred (resolution: literal_explicit)
  - cite[43:91] byte-exact: "because well, obviously, she's not killing Poppy"

#### recommended_advice (1 items)
- [1] Directive to the student: chase (following the deduction that she is bot) (resolution: literal_explicit)
  - cite[186:200] byte-exact: 'So, you chase.'

#### consequences_outcomes (1 items)
- [1] Potential outcome: if Mel is showing on mid, the situation could be a good flip (resolution: literal_explicit)
  - cite[227:253] byte-exact: 'this could be a good flip,'

#### explicit_relationships (8 items)
- [1] 'it' (in 'then it's bot') REFERS_TO the tracked female champion ('she') whose location is being deduced (resolution: context_resolved; relation: REFERS_TO)
  - cite[0:14] byte-exact: "then it's bot."
  - cite[26:41] byte-exact: "She's not a bot"
- [2] 'this' (in 'this could be a good flip') REFERS_TO the chase opportunity conditioned on Mel showing on mid (resolution: context_resolved; relation: REFERS_TO)
  - cite[201:253] byte-exact: 'If Mel is showing on mid, this could be a good flip,'
- [3] Premise 'she's not killing Poppy' CAUSES the deduction 'She's not a bot' (resolution: literal_explicit; relation: CAUSES)
  - cite[26:92] byte-exact: "She's not a bot, because well, obviously, she's not killing Poppy."
- [4] Premise 'wave equals ward' CAUSES the deduction 'She's not on mid' (resolution: literal_explicit; relation: CAUSES)
  - cite[93:136] byte-exact: "She's not on mid, because wave equals ward."
- [5] Condition 'If Mel is showing on mid' is the CONDITION for the outcome 'this could be a good flip' (resolution: literal_explicit; relation: CONDITION)
  - cite[201:253] byte-exact: 'If Mel is showing on mid, this could be a good flip,'
- [6] 'you' (the coached player) is the ACTOR of the urged action 'you chase' (resolution: literal_explicit; relation: ACTOR)
  - cite[186:200] byte-exact: 'So, you chase.'
- [7] The tracked female champion is the ACTOR of the negated action of killing Poppy (the action is explicitly denied) (resolution: literal_explicit; relation: ACTOR)
  - cite[68:91] byte-exact: "she's not killing Poppy"
- [8] Poppy is the TARGET of the negated action 'she's not killing Poppy' (the killing is explicitly denied) (resolution: literal_explicit; relation: TARGET)
  - cite[57:91] byte-exact: "obviously, she's not killing Poppy"

#### uncertainty_unresolved (7 items)
- [1] Identity of the female champion tracked as 'she'/'it' is never given; she cannot be bound to any named champion from the supplied material (resolution: unresolved)
  - cite[26:41] byte-exact: "She's not a bot"
  - cite[0:14] byte-exact: "then it's bot."
- [2] Tension between 'then it's bot' and the following 'She's not a bot': whether the latter means 'she is not [at/going] bot' or reflects ASR-corrupted speech cannot be determined from the supplied text (resolution: unresolved)
  - cite[0:42] byte-exact: "then it's bot. Yes? Yeah. She's not a bot,"
- [3] Whether 'Mel' denotes the same entity as the tracked 'she' or a different champion is not determinable from the passage; no conflation is made (resolution: unresolved)
  - cite[201:225] byte-exact: 'If Mel is showing on mid'
  - cite[93:109] byte-exact: "She's not on mid"
- [4] Compressed elliptical phrasing 'wave equals ward' is preserved verbatim; any fuller intended meaning beyond the stated words is unrecoverable (resolution: unresolved)
  - cite[119:135] byte-exact: 'wave equals ward'
- [5] 'But you don't play around waves being your ward.' is ambiguous between a prescriptive instruction and a descriptive remark and may be ASR-garbled; recorded as stated without repair (resolution: unresolved)
  - cite[137:185] byte-exact: "But you don't play around waves being your ward."
- [6] Speaker turns for 'Yes? Yeah.' cannot be attributed to coach versus student from the supplied text (resolution: unresolved)
  - cite[15:25] byte-exact: 'Yes? Yeah.'
- [7] The slang sense of 'flip' (e.g., a risky gamble) is not defined in the supplied material; only the literal words 'a good flip' are supported (resolution: unresolved)
  - cite[227:253] byte-exact: 'this could be a good flip,'

#### supporting_source_spans (7 items)
- [1] Opening deduction that the tracked champion's location is bot (resolution: literal_explicit)
  - cite[0:14] byte-exact: "then it's bot."
  - span [0:14]
- [2] Confirmation exchange between speakers (resolution: literal_explicit)
  - cite[15:25] byte-exact: 'Yes? Yeah.'
  - span [15:25]
- [3] Elimination of bot with its stated premise (resolution: literal_explicit)
  - cite[26:92] byte-exact: "She's not a bot, because well, obviously, she's not killing Poppy."
  - span [26:92]
- [4] Elimination of mid with its stated premise (resolution: literal_explicit)
  - cite[93:136] byte-exact: "She's not on mid, because wave equals ward."
  - span [93:136]
- [5] Correction about playing around waves as ward (resolution: literal_explicit)
  - cite[137:185] byte-exact: "But you don't play around waves being your ward."
  - span [137:185]
- [6] Directive to chase (resolution: literal_explicit)
  - cite[186:200] byte-exact: 'So, you chase.'
  - span [186:200]
- [7] Conditional flip assessment closing the passage (resolution: literal_explicit)
  - cite[201:253] byte-exact: 'If Mel is showing on mid, this could be a good flip,'
  - span [201:253]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (3 items)
- [1] Mel (enemy champion; subject of the location deduction over mid, Baron, and bot) (resolution: literal_explicit)
  - cite[56739:56791] byte-exact: 'If Mel is showing on mid, this could be a good flip,'
  - cite[56248:56272] byte-exact: 'Well, where else is Mel?'
- [2] Poppy (champion cited as evidence in the bot elimination: Mel is not killing Poppy) (resolution: literal_explicit)
  - cite[56564:56630] byte-exact: "She's not a bot, because well, obviously, she's not killing Poppy."
- [3] The coached player ('you', the Jinx player), who does not play around waves as wards and chases (resolution: context_resolved)
  - cite[56675:56738] byte-exact: "But you don't play around waves being your ward. So, you chase."

#### reference_bindings (4 items)
- [1] 'She' in "She's not a bot" and "She's not on mid" binds to Mel, who is being located among three options (mid, Baron, bot) (resolution: context_resolved)
  - cite[56564:56630] byte-exact: "She's not a bot, because well, obviously, she's not killing Poppy."
  - cite[56456:56490] byte-exact: 'Mel only has three options, right?'
- [2] 'it' in "then it's bot" binds to the third enumerated option for Mel's location (after mid and Baron), i.e., bot lane (resolution: context_resolved)
  - cite[56534:56552] byte-exact: "And then it's bot."
  - cite[56456:56490] byte-exact: 'Mel only has three options, right?'
- [3] 'you' in "you don't play around waves being your ward" and in "you chase" binds to the coached player (resolution: context_resolved)
  - cite[56675:56738] byte-exact: "But you don't play around waves being your ward. So, you chase."
- [4] 'this' in "this could be a good flip" binds to the prospective engagement/gamble against Mel in the situation where she shows on mid (resolution: context_resolved)
  - cite[56739:56791] byte-exact: 'If Mel is showing on mid, this could be a good flip,'
  - cite[56825:56891] byte-exact: 'But Mel is not showing on the mid wave, so she has to be on Baron.'

#### abilities_resources (1 items)
- [1] Waves functioning as wards (vision resource): the 'wave equals ward' principle, and the player's failure to use waves as his ward (resolution: literal_explicit)
  - cite[56631:56674] byte-exact: "She's not on mid, because wave equals ward."
  - cite[56675:56723] byte-exact: "But you don't play around waves being your ward."

#### events_actions (1 items)
- [1] The player chases (action stated as following from not playing around waves as wards) (resolution: literal_explicit)
  - cite[56675:56738] byte-exact: "But you don't play around waves being your ward. So, you chase."

#### states (3 items)
- [1] Mel is not at bot, evidenced by her not killing Poppy (resolution: context_resolved)
  - cite[56564:56630] byte-exact: "She's not a bot, because well, obviously, she's not killing Poppy."
- [2] Mel is not on mid, evidenced by the wave-equals-ward principle given the new mid wave (resolution: context_resolved)
  - cite[56631:56674] byte-exact: "She's not on mid, because wave equals ward."
  - cite[56335:56358] byte-exact: "There's a new mid wave."
- [3] Mel is not killing Poppy (ongoing non-kill observation used as elimination evidence) (resolution: literal_explicit)
  - cite[56564:56630] byte-exact: "She's not a bot, because well, obviously, she's not killing Poppy."

#### conditions (1 items)
- [1] Condition: Mel is showing on mid (antecedent of the flip evaluation) (resolution: literal_explicit)
  - cite[56739:56791] byte-exact: 'If Mel is showing on mid, this could be a good flip,'

#### recommended_advice (1 items)
- [1] Implied recommendation: play around waves being your ward (use waves as vision), which the coach states the player currently does not do; consistent with the adjacent explicit advice 'Use waves as vision.' (resolution: context_resolved)
  - cite[56675:56723] byte-exact: "But you don't play around waves being your ward."
  - cite[56314:56334] byte-exact: 'Use waves as vision.'

#### consequences_outcomes (3 items)
- [1] If Mel is showing on mid, this could be a good flip (potentially favorable gamble/fight) (resolution: literal_explicit)
  - cite[56739:56791] byte-exact: 'If Mel is showing on mid, this could be a good flip,'
- [2] Consequence of not playing around waves being your ward: you chase (resolution: context_resolved)
  - cite[56675:56738] byte-exact: "But you don't play around waves being your ward. So, you chase."
- [3] Outcome completing the elimination begun in the target passage: with bot and mid ruled out, Mel has to be on Baron (resolution: context_resolved)
  - cite[56825:56891] byte-exact: 'But Mel is not showing on the mid wave, so she has to be on Baron.'
  - cite[56456:56490] byte-exact: 'Mel only has three options, right?'

#### explicit_relationships (5 items)
- [1] 'She' (subject of 'not a bot' and 'not on mid') REFERS_TO Mel (resolution: context_resolved; relation: REFERS_TO)
  - cite[56631:56674] byte-exact: "She's not on mid, because wave equals ward."
  - cite[56248:56272] byte-exact: 'Well, where else is Mel?'
- [2] Mel showing on mid is the CONDITION for 'this could be a good flip' (resolution: literal_explicit; relation: CONDITION)
  - cite[56739:56791] byte-exact: 'If Mel is showing on mid, this could be a good flip,'
- [3] Not playing around waves as wards CAUSES the player to chase (resolution: context_resolved; relation: CAUSES)
  - cite[56675:56738] byte-exact: "But you don't play around waves being your ward. So, you chase."
- [4] The observation 'she's not killing Poppy' NEGATES the event of Mel killing Poppy (basis for ruling out bot) (resolution: literal_explicit; relation: NEGATES)
  - cite[56564:56630] byte-exact: "She's not a bot, because well, obviously, she's not killing Poppy."
- [5] Waves acting as wards ENABLES ruling Mel out of mid (her absence from the mid wave shows she is not there) (resolution: context_resolved; relation: ENABLES)
  - cite[56631:56674] byte-exact: "She's not on mid, because wave equals ward."

#### uncertainty_unresolved (2 items)
- [1] Surface form "She's not a bot" is ambiguous between literal 'she is not a bot (AI)' and contextual 'she is not at bot (lane)'; the surrounding enumeration (mid, Baron, bot) supports the location reading, but any ASR corruption of e.g. 'at bot'/'on bot' into 'a bot' cannot be verified from the supplied material (resolution: unresolved)
  - cite[56564:56630] byte-exact: "She's not a bot, because well, obviously, she's not killing Poppy."
- [2] "So, you chase." is ambiguous between a description of the player's actual behavior (criticism) and a prescription ('therefore you chase'); the supplied material does not fully disambiguate these readings (resolution: unresolved)
  - cite[56675:56738] byte-exact: "But you don't play around waves being your ward. So, you chase."

#### supporting_source_spans (4 items)
- [1] Core span covering the bot and mid eliminations and their stated evidence (resolution: literal_explicit)
  - cite[56534:56674] byte-exact: "And then it's bot. Yes? Yeah. She's not a bot, because well, obviously, she's not killing Poppy. She's not on mid, be..."
  - span [56534:56674]
- [2] Span covering the failure to use waves as wards and the resulting chase (resolution: literal_explicit)
  - cite[56675:56738] byte-exact: "But you don't play around waves being your ward. So, you chase."
  - span [56675:56738]
- [3] Span covering the conditional flip evaluation closing the target passage (resolution: literal_explicit)
  - cite[56739:56791] byte-exact: 'If Mel is showing on mid, this could be a good flip,'
  - span [56739:56791]
- [4] Adjacent-context spans resolving 'She' to Mel and completing the elimination with Baron as the remaining option (resolution: context_resolved)
  - cite[56248:56272] byte-exact: 'Well, where else is Mel?'
  - cite[56456:56490] byte-exact: 'Mel only has three options, right?'
  - cite[56825:56891] byte-exact: 'But Mel is not showing on the mid wave, so she has to be on Baron.'
  - span [56248:56891]

Raw response binding A: `9d6fbad6911d2ebec7c81d30fde02f3a1931f036f60229971caa078cfaefbfa1`
Raw response binding B: `339ee522f1ade08695cfaff6718623551b618d76e04f2fab319787ad675fa84e`

---

## TARGET p2k:case:0010 (selection rank 10)

### Exact Bronze

```text
yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to get push then you could get it then you could win MH but you're not meant to win this you know but
```

Location in full transcript: [23114:23320]

Metadata supplied: video_title, champion, role, game, description

### Condition A — Isolated Bronze

#### actors_entities (1 items)
- [1] 'you' - the player being coached (addressee of the passage) (resolution: context_resolved)
  - cite[0:30] byte-exact: "yeah and now you don't win but"

#### reference_bindings (3 items)
- [1] 'you' binds to the player being coached (resolution: context_resolved)
  - cite[9:26] byte-exact: "now you don't win"
- [2] 'it' in 'then you could get it' binds to the push (resolution: context_resolved)
  - cite[96:138] byte-exact: 'an angle to get push then you could get it'
- [3] 'this' in 'not meant to win this' binds to the current game/match under discussion (resolution: context_resolved)
  - cite[172:202] byte-exact: 'not meant to win this you know'

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (2 items)
- [1] Current state at 'now': you do not win / have not won (resolution: literal_explicit)
  - cite[0:30] byte-exact: "yeah and now you don't win but"
- [2] Assessed expectation for the current game: you are not meant to win it (resolution: literal_explicit)
  - cite[161:206] byte-exact: "but you're not meant to win this you know but"

#### conditions (2 items)
- [1] The single way that you can win is conditioned on you doing/getting the push (resolution: literal_explicit)
  - cite[31:81] byte-exact: 'the one way that you can win is if you do get push'
- [2] If there's an angle to get push, then you could get it, and then you could win (resolution: literal_explicit)
  - cite[82:116] byte-exact: "so if there's an angle to get push"
  - cite[117:160] byte-exact: 'then you could get it then you could win MH'

#### recommended_advice (1 items)
- [1] If there is an angle to get push, get/take it - pursue the push as the stated path to winning (resolution: literal_explicit)
  - cite[85:138] byte-exact: "if there's an angle to get push then you could get it"

#### consequences_outcomes (2 items)
- [1] If you do get the push, you could win (resolution: literal_explicit)
  - cite[63:81] byte-exact: 'if you do get push'
  - cite[139:160] byte-exact: 'then you could win MH'
- [2] Because getting push is stated as the one way to win, without getting push you cannot/don't win (resolution: context_resolved)
  - cite[31:81] byte-exact: 'the one way that you can win is if you do get push'

#### explicit_relationships (6 items)
- [1] 'if you do get push' is the condition under which 'the one way that you can win' holds (resolution: literal_explicit; relation: CONDITION)
  - cite[31:81] byte-exact: 'the one way that you can win is if you do get push'
- [2] 'if there's an angle to get push' is the condition for 'then you could get it' (resolution: literal_explicit; relation: CONDITION)
  - cite[85:138] byte-exact: "if there's an angle to get push then you could get it"
- [3] getting the push ('you could get it') results in 'you could win' (resolution: literal_explicit; relation: RESULT)
  - cite[117:160] byte-exact: 'then you could get it then you could win MH'
- [4] 'it' in 'you could get it' refers to the push (resolution: context_resolved; relation: REFERS_TO)
  - cite[96:138] byte-exact: 'an angle to get push then you could get it'
- [5] 'you' is the actor who would do/get the push (resolution: context_resolved; relation: ACTOR)
  - cite[63:81] byte-exact: 'if you do get push'
- [6] 'you' is the actor who could win (resolution: context_resolved; relation: ACTOR)
  - cite[139:160] byte-exact: 'then you could win MH'

#### uncertainty_unresolved (2 items)
- [1] 'MH' - unintelligible token, likely an ASR artifact of a backchannel acknowledgment; unrecoverable from the supplied material (resolution: unresolved)
  - cite[154:164] byte-exact: 'win MH but'
- [2] 'push' in 'get push' - what is being pushed (wave vs objective) is not specified in the supplied passage; intended referent cannot be determined from source alone (resolution: unresolved)
  - cite[66:84] byte-exact: 'you do get push so'

#### supporting_source_spans (3 items)
- [1] Span supporting the current not-winning state and the sole stated winning condition (resolution: literal_explicit)
  - cite[0:30] byte-exact: "yeah and now you don't win but"
  - cite[31:81] byte-exact: 'the one way that you can win is if you do get push'
  - span [0:81]
- [2] Span supporting the conditional chain from an angle to get push through possibly winning (resolution: literal_explicit)
  - cite[82:160] byte-exact: "so if there's an angle to get push then you could get it then you could win MH"
  - span [82:160]
- [3] Span supporting the assessed expectation that the current game is not meant to be won (resolution: literal_explicit)
  - cite[161:206] byte-exact: "but you're not meant to win this you know but"
  - span [161:206]

### Condition B — Full Transcript
(B citations resolve against the FULL ordered transcript; byte-exactness was verified mechanically at import time.)

### Condition B — structured extraction

#### actors_entities (4 items)
- [1] The coached player ('you'), an ADC player who plays Varus in the matchup under discussion. (resolution: context_resolved)
  - cite[22939:22974] byte-exact: "sometimes you're playing your varus"
  - cite[23119:23140] byte-exact: "and now you don't win"
- [2] Varus, the champion the coached player is playing in this bot-lane matchup. (resolution: context_resolved)
  - cite[22939:22974] byte-exact: "sometimes you're playing your varus"
- [3] Blitzcrank (surface form 'bliss crank' immediately before the target passage), the enemy champion that makes the matchup a losing one. (resolution: context_resolved)
  - cite[23073:23113] byte-exact: 'against a bliss crank now with a kalista'
  - cite[29006:29028] byte-exact: 'blitzcrank misses hook'
- [4] Kalista, the enemy champion paired with Blitzcrank in the matchup bound to 'this'. (resolution: context_resolved)
  - cite[23099:23113] byte-exact: 'with a kalista'

#### reference_bindings (4 items)
- [1] 'you' throughout the target passage binds to the coached player (the student ADC playing Varus). (resolution: context_resolved)
  - cite[22939:22974] byte-exact: "sometimes you're playing your varus"
  - cite[23119:23140] byte-exact: "and now you don't win"
- [2] 'this' in 'you're not meant to win this' binds to the bot-lane matchup of the player's Varus versus Blitzcrank and Kalista described immediately before the target passage. (resolution: context_resolved)
  - cite[23275:23316] byte-exact: "but you're not meant to win this you know"
  - cite[23073:23113] byte-exact: 'against a bliss crank now with a kalista'
  - cite[22939:22974] byte-exact: "sometimes you're playing your varus"
- [3] 'it' in 'then you could get it' binds to the push (the wave push whose angle is under discussion). (resolution: context_resolved)
  - cite[23196:23271] byte-exact: "so if there's an angle to get push then you could get it then you could win"
- [4] 'push' / 'get push' binds to obtaining wave push (lane priority and control), consistent with the coach's earlier statement that Varus wins lane by having push and control. (resolution: context_resolved)
  - cite[23177:23195] byte-exact: 'if you do get push'
  - cite[3497:3541] byte-exact: 'you just win Lane by having push and control'

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (2 items)
- [1] Current state: the player does not win in this situation ('now you don't win'). (resolution: literal_explicit)
  - cite[23123:23195] byte-exact: "now you don't win but the one way that you can win is if you do get push"
- [2] Standing expectation for this matchup: it is not one the player is meant to win. (resolution: literal_explicit)
  - cite[23275:23316] byte-exact: "but you're not meant to win this you know"

#### conditions (2 items)
- [1] The one way the player can win is conditional on getting push ('if you do get push'); implicitly, without push the player does not win. (resolution: literal_explicit)
  - cite[23145:23195] byte-exact: 'the one way that you can win is if you do get push'
- [2] Conditional: if there is an angle to get push, then the player could take it and could win. (resolution: literal_explicit)
  - cite[23196:23271] byte-exact: "so if there's an angle to get push then you could get it then you could win"

#### recommended_advice (1 items)
- [1] If there is an angle to get push, take it, because getting push is the one way to win this matchup. (resolution: literal_explicit)
  - cite[23145:23195] byte-exact: 'the one way that you can win is if you do get push'
  - cite[23196:23271] byte-exact: "so if there's an angle to get push then you could get it then you could win"

#### consequences_outcomes (3 items)
- [1] Outcome of getting push: the player could win ('then you could win'). (resolution: literal_explicit)
  - cite[23231:23271] byte-exact: 'then you could get it then you could win'
- [2] Implied outcome without push: the player does not win this matchup, since getting push is stated as the one way to win. (resolution: context_resolved)
  - cite[23123:23195] byte-exact: "now you don't win but the one way that you can win is if you do get push"
- [3] Expected outcome: the player is not meant to win this matchup. (resolution: literal_explicit)
  - cite[23275:23316] byte-exact: "but you're not meant to win this you know"

#### explicit_relationships (6 items)
- [1] Winning this matchup REQUIRES getting push. (resolution: vocabulary_supported; relation: REQUIRES)
  - cite[23145:23195] byte-exact: 'the one way that you can win is if you do get push'
- [2] There being an angle to get push is the CONDITION under which the player could take it and could win. (resolution: literal_explicit; relation: CONDITION)
  - cite[23196:23271] byte-exact: "so if there's an angle to get push then you could get it then you could win"
- [3] Getting push RESULTS IN the possibility of winning. (resolution: vocabulary_supported; relation: RESULT)
  - cite[23231:23271] byte-exact: 'then you could get it then you could win'
- [4] Getting push ENABLES winning, as push is stated to be the one way the player can win. (resolution: vocabulary_supported; relation: ENABLES)
  - cite[23174:23271] byte-exact: "is if you do get push so if there's an angle to get push then you could get it then you could win"
- [5] The coached player is the ACTOR who would get push and win. (resolution: context_resolved; relation: ACTOR)
  - cite[23196:23271] byte-exact: "so if there's an angle to get push then you could get it then you could win"
- [6] 'this' REFERS_TO the Varus versus Blitzcrank-and-Kalista bot-lane matchup. (resolution: context_resolved; relation: REFERS_TO)
  - cite[23275:23316] byte-exact: "but you're not meant to win this you know"
  - cite[23073:23113] byte-exact: 'against a bliss crank now with a kalista'

#### uncertainty_unresolved (3 items)
- [1] 'MH' in the target passage is a backchannel/acknowledgment token with no entity referent resolvable from the supplied material. (resolution: unresolved)
  - cite[23258:23278] byte-exact: 'you could win MH but'
- [2] Adjacent-context surface form 'jna' (describing the player's side/partner in the matchup framing) is ASR-corrupted and cannot be repaired from the supplied vocabulary or context. (resolution: unresolved)
  - cite[22993:23003] byte-exact: 'like a jna'
- [3] The spelling 'bliss crank' is ASR-corrupted; its binding to Blitzcrank relies on the later literal mention 'blitzcrank misses hook' rather than any supplied alias rule, so residual uncertainty is recorded. (resolution: unresolved)
  - cite[23073:23113] byte-exact: 'against a bliss crank now with a kalista'
  - cite[29006:29028] byte-exact: 'blitzcrank misses hook'

#### supporting_source_spans (3 items)
- [1] Target-passage span covering the current non-winning state and the single winning condition of getting push. (resolution: literal_explicit)
  - cite[23123:23195] byte-exact: "now you don't win but the one way that you can win is if you do get push"
  - span [23123:23195]
- [2] Target-passage span covering the angle-to-push condition, the possible win, and the expectation that this matchup is not meant to be won. (resolution: literal_explicit)
  - cite[23196:23320] byte-exact: "so if there's an angle to get push then you could get it then you could win MH but you're not meant to win this you k..."
  - span [23196:23320]
- [3] Immediately preceding context establishing the matchup participants needed to bind 'you', 'push', and 'this'. (resolution: context_resolved)
  - cite[22931:23003] byte-exact: "because sometimes you're playing your varus and then you have like a jna"
  - cite[23042:23113] byte-exact: 'and then sometimes you will be against a bliss crank now with a kalista'
  - span [22931:23113]

Raw response binding A: `f233191cc840edd7f75b51e295a553c2d38948d580929f1fd5d9a6ddb413eefb`
Raw response binding B: `e2b7c553748c929dbc250ab80671d47d83bf86c3d1722759d85b1f4d973318bb`
