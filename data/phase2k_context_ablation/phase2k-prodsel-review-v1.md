# Phase 2K Production-Model Selection — Review Packet

Same frozen 10-target benchmark and scoring contract as the completed Phase 2K ablation. Baseline reference: OX = 0x Alpha full-context (109/110). Reviewer note: scoring remains AI-based.

---

## Condition P (model opencode-go/deepseek-v4-pro)

### TARGET p2k:case:0001

#### actors_entities (3 items)
- [1] I (the coach speaking) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [2] you (the player being coached) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [3] Karthus (the champion under discussion; context says carus) (resolution: context_resolved)
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'

#### reference_bindings (12 items)
- [1] you -> the player being coached (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [2] I in 'I do' -> the player being coached (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [3] I in 'I would' -> the coach (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [4] this -> the Exhaust/Smite and rune setup under discussion (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [5] flash -> Flash (summoner spell) (resolution: literal_explicit)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [6] exhaust -> Exhaust (summoner spell) (resolution: literal_explicit)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [7] SM -> Smite (summoner spell) (resolution: vocabulary_supported)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [8] Last Stand -> Last Stand (rune) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [9] Harvest -> Dark Harvest (rune) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [10] cheap shot -> Cheap Shot (rune) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [11] ultimate Hunter -> Ultimate Hunter (rune) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [12] eyeball -> Eyeball Collection (rune) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### abilities_resources (8 items)
- [1] Flash (summoner spell; recommended by the coach) (resolution: literal_explicit)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [2] Exhaust (summoner spell; part of the player's current setup) (resolution: literal_explicit)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [3] SM (Smite; summoner spell; part of the player's current setup) (resolution: vocabulary_supported)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [4] Last Stand (rune the player loses by not taking it) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [5] Harvest (Dark Harvest rune the player loses by not taking it) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [6] cheap shot (Cheap Shot rune the player loses by not taking it) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [7] Ultimate Hunter (rune the player probably has) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [8] Eyeball Collection (rune the player probably has, from 'eyeball') (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### events_actions (2 items)
- [1] The coach states he would play with Flash rather than Exhaust/Smite (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [2] The player confirms he has Ultimate Hunter and Eyeball (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### states (6 items)
- [1] the player is going Exhaust and Smite (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [2] the player has less fighting Power (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [3] the player is losing Last Stand (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [4] the player is losing Harvest (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [5] the player is losing cheap shot (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [6] the player has Ultimate Hunter and Eyeball (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### conditions (1 items)
- [1] you're going exhaust SM (the condition under which the player has less fighting power) (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### recommended_advice (1 items)
- [1] Play with Flash rather than Exhaust/Smite (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### consequences_outcomes (2 items)
- [1] Going Exhaust/Smite results in less fighting power (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [2] Going Exhaust/Smite means losing Last Stand, Harvest, and cheap shot (resolution: context_resolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### explicit_relationships (5 items)
- [1] you (the player) uses Exhaust and Smite (resolution: context_resolved; relation: USES)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [2] I (the coach) uses Flash in the recommended setup (resolution: context_resolved; relation: USES)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [3] going Exhaust/Smite causes less fighting power (resolution: context_resolved; relation: CAUSES)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [4] losing Last Stand, Harvest, and cheap shot causes less fighting power (resolution: context_resolved; relation: CAUSES)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [5] playing with Flash rather than Exhaust/Smite negates the Exhaust/Smite setup (resolution: context_resolved; relation: NEGATES)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### uncertainty_unresolved (1 items)
- [1] SM (ASR-corrupted token; context suggests Smite) (resolution: unresolved)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."

#### supporting_source_spans (2 items)
- [1] play with flash because like you're going exhaust SM but you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot because you probably have ultimate Hunter eyeball right yeah I do right that doesn't mean this is bad like I would (resolution: literal_explicit)
  - cite[4942:5222] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM but you have less fighting Power because you're losing..."
- [2] for carus you are never allowed to go Smite and exhaust again (resolution: literal_explicit)
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'

### TARGET p2k:case:0002

#### actors_entities (2 items)
- [1] atrox (resolution: literal_explicit)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'
- [2] darus (resolution: vocabulary_supported)
  - cite[30531:30762] (citations verified byte-exact at import): "look at atrox atrox has 11 CS after this if at he's giving a freeb right because this if yeah if atrox T is..."

#### reference_bindings (4 items)
- [1] you (resolution: context_resolved)
  - cite[30649:30668] (citations verified byte-exact at import): 'do you really think'
- [2] I (resolution: context_resolved)
  - cite[30738:30762] (citations verified byte-exact at import): "because I don't think so"
- [3] this darus (resolution: context_resolved)
  - cite[30531:30762] (citations verified byte-exact at import): "look at atrox atrox has 11 CS after this if at he's giving a freeb right because this if yeah if atrox T is..."
- [4] he's giving a freeb (resolution: unresolved)
  - cite[30578:30597] (citations verified byte-exact at import): "he's giving a freeb"

#### abilities_resources (2 items)
- [1] T (Teleport) (resolution: context_resolved)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
- [2] clo armor (resolution: literal_explicit)
  - cite[30675:30697] (citations verified byte-exact at import): 'with clo armor full HP'

#### events_actions (1 items)
- [1] atrox has 11 CS after this (resolution: literal_explicit)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'

#### states (2 items)
- [1] atrox has 11 CS (resolution: literal_explicit)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'
- [2] atrox has clo armor and full HP (resolution: literal_explicit)
  - cite[30675:30697] (citations verified byte-exact at import): 'with clo armor full HP'

#### conditions (2 items)
- [1] if atrox T is back here (resolution: literal_explicit)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
- [2] if at he's giving a freeb (resolution: unresolved)
  - cite[30572:30597] (citations verified byte-exact at import): "if at he's giving a freeb"

#### recommended_advice (0 items)

(none)

#### consequences_outcomes (2 items)
- [1] atrox with clo armor full HP will beat this darus probably not right because I don't think so (resolution: literal_explicit)
  - cite[30649:30762] (citations verified byte-exact at import): "do you really think atrox with clo armor full HP will beat this darus probably not right because I don't th..."
- [2] he's giving a freeb (resolution: unresolved)
  - cite[30578:30597] (citations verified byte-exact at import): "he's giving a freeb"

#### explicit_relationships (3 items)
- [1] atrox uses T (resolution: context_resolved; relation: USES)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
- [2] if atrox T is back here, then evaluate whether atrox with clo armor full HP will beat this darus (resolution: context_resolved; relation: CONDITION)
  - cite[30531:30762] (citations verified byte-exact at import): "look at atrox atrox has 11 CS after this if at he's giving a freeb right because this if yeah if atrox T is..."
- [3] atrox with clo armor full HP does not beat this darus (resolution: context_resolved; relation: NEGATES)
  - cite[30531:30762] (citations verified byte-exact at import): "look at atrox atrox has 11 CS after this if at he's giving a freeb right because this if yeah if atrox T is..."

#### uncertainty_unresolved (2 items)
- [1] freeb (resolution: unresolved)
  - cite[30583:30597] (citations verified byte-exact at import): 'giving a freeb'
- [2] if at (resolution: unresolved)
  - cite[30572:30597] (citations verified byte-exact at import): "if at he's giving a freeb"

#### supporting_source_spans (1 items)
- [1] look at atrox atrox has 11 CS after this if at he's giving a freeb right because this if yeah if atrox T is back here do you really think atrox with clo armor full HP will beat this darus probably not right because I don't think so (resolution: literal_explicit)
  - cite[30531:30762] (citations verified byte-exact at import): "look at atrox atrox has 11 CS after this if at he's giving a freeb right because this if yeah if atrox T is..."

### TARGET p2k:case:0003

#### actors_entities (3 items)
- [1] Fiora (resolution: literal_explicit)
  - cite[13322:13359] (citations verified byte-exact at import): 'I would always go ignite versus Fiora'
- [2] Camille (resolution: literal_explicit)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
- [3] I (the speaker/coach) (resolution: context_resolved)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'

#### reference_bindings (3 items)
- [1] she (Fiora) (resolution: literal_explicit)
  - cite[13354:13377] (citations verified byte-exact at import): 'Fiora because she heals'
- [2] I (the speaker/coach) (resolution: context_resolved)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [3] you (the addressed player/student) (resolution: context_resolved)
  - cite[13502:13524] (citations verified byte-exact at import): 'when you need heal cut'

#### abilities_resources (5 items)
- [1] ignite (resolution: literal_explicit)
  - cite[13337:13359] (citations verified byte-exact at import): 'go ignite versus Fiora'
- [2] flash (resolution: literal_explicit)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
- [3] TP (resolution: literal_explicit)
  - cite[13452:13477] (citations verified byte-exact at import): 'go flash TP in most cases'
- [4] heal cut (resolution: literal_explicit)
  - cite[13502:13532] (citations verified byte-exact at import): 'when you need heal cut is fine'
- [5] Fiora's healing (she heals) (resolution: context_resolved)
  - cite[13368:13401] (citations verified byte-exact at import): 'she heals more than [\xa0__\xa0] window'

#### events_actions (0 items)

(none)

#### states (1 items)
- [1] Fiora heals more than [ __ ] window (resolution: unresolved)
  - cite[13368:13401] (citations verified byte-exact at import): 'she heals more than [\xa0__\xa0] window'

#### conditions (3 items)
- [1] versus Fiora (resolution: literal_explicit)
  - cite[13337:13359] (citations verified byte-exact at import): 'go ignite versus Fiora'
- [2] in most cases (resolution: literal_explicit)
  - cite[13452:13477] (citations verified byte-exact at import): 'go flash TP in most cases'
- [3] when you need heal cut (resolution: literal_explicit)
  - cite[13502:13532] (citations verified byte-exact at import): 'when you need heal cut is fine'

#### recommended_advice (3 items)
- [1] always go ignite versus Fiora (resolution: literal_explicit)
  - cite[13322:13359] (citations verified byte-exact at import): 'I would always go ignite versus Fiora'
- [2] go flash TP in most cases (resolution: literal_explicit)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [3] ignite when you need heal cut is fine (resolution: literal_explicit)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'

#### consequences_outcomes (1 items)
- [1] ignite when you need heal cut is fine (resolution: literal_explicit)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'

#### explicit_relationships (3 items)
- [1] she refers to Fiora (resolution: literal_explicit; relation: REFERS_TO)
  - cite[13354:13377] (citations verified byte-exact at import): 'Fiora because she heals'
- [2] Fiora heals more than [ __ ] window (resolution: unresolved; relation: CAUSES)
  - cite[13368:13401] (citations verified byte-exact at import): 'she heals more than [\xa0__\xa0] window'
- [3] you need heal cut (resolution: literal_explicit; relation: CONDITION)
  - cite[13502:13532] (citations verified byte-exact at import): 'when you need heal cut is fine'

#### uncertainty_unresolved (1 items)
- [1] she heals more than [ __ ] window (resolution: unresolved)
  - cite[13368:13401] (citations verified byte-exact at import): 'she heals more than [\xa0__\xa0] window'

#### supporting_source_spans (1 items)
- [1] Target passage summoner spell advice (resolution: literal_explicit)
  - cite[13315:13540] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window but I think flash is...'

### TARGET p2k:case:0004

#### actors_entities (3 items)
- [1] Brier (Briar) (resolution: context_resolved)
  - cite[54345:54425] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go because you still want to invade"
- [2] Talia (Taliyah) (resolution: context_resolved)
  - cite[54524:54566] (citations verified byte-exact at import): 'I hope Talia does not 2v one you right but'
- [3] you (the coached Aatrox top laner) (resolution: context_resolved)
  - cite[54524:54566] (citations verified byte-exact at import): 'I hope Talia does not 2v one you right but'
  - cite[54345:54425] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go because you still want to invade"

#### reference_bindings (5 items)
- [1] you -> the coached Aatrox top laner (resolution: context_resolved)
  - cite[54345:54425] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go because you still want to invade"
- [2] Brier -> Briar (resolution: context_resolved)
  - cite[54345:54425] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go because you still want to invade"
- [3] Talia -> Taliyah (resolution: context_resolved)
  - cite[54524:54566] (citations verified byte-exact at import): 'I hope Talia does not 2v one you right but'
- [4] it -> the invade/invasion (resolution: context_resolved)
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'
- [5] I -> the coach/speaker (resolution: context_resolved)
  - cite[54524:54566] (citations verified byte-exact at import): 'I hope Talia does not 2v one you right but'

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (0 items)

(none)

#### conditions (2 items)
- [1] If Brier does not win the 1v1, you should go. (resolution: literal_explicit)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
- [2] Just because Brier loses to does not mean you should not invade. (resolution: literal_explicit)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"

#### recommended_advice (2 items)
- [1] If Brier does not win the 1v1, go because you still want to invade. (resolution: literal_explicit)
  - cite[54345:54425] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go because you still want to invade"
- [2] Do not decide not to invade just because Brier loses the 1v1; you could make it winning. (resolution: literal_explicit)
  - cite[54426:54517] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade cuz you could make it winning"

#### consequences_outcomes (2 items)
- [1] Going can make the invade winning. (resolution: literal_explicit)
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'
- [2] The coach hopes Talia does not 2v1 the player. (resolution: literal_explicit)
  - cite[54524:54566] (citations verified byte-exact at import): 'I hope Talia does not 2v one you right but'

#### explicit_relationships (2 items)
- [1] Brier not winning the 1v1 is the condition for you to go. (resolution: literal_explicit; relation: CONDITION)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
- [2] Brier losing the 1v1 does not negate the reason to invade. (resolution: literal_explicit; relation: NEGATES)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"

#### uncertainty_unresolved (2 items)
- [1] Brier loses to (missing opponent/object) (resolution: unresolved)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"
- [2] 2v one (ASR corruption/nonstandard spelling of 2v1) (resolution: unresolved)
  - cite[54524:54566] (citations verified byte-exact at import): 'I hope Talia does not 2v one you right but'

#### supporting_source_spans (1 items)
- [1] move then you should go if Brier doesn't win one one then you should go because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it winning right I hope Talia does not 2v one you right but (resolution: literal_explicit)
  - cite[54321:54566] (citations verified byte-exact at import): "move then you should go if Brier doesn't win one one then you should go because you still want to invade ju..."

### TARGET p2k:case:0005

#### actors_entities (3 items)
- [1] Syndra (resolution: context_resolved)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [2] Veigar (player) (resolution: context_resolved)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [3] wave (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'

#### reference_bindings (4 items)
- [1] she → Syndra (resolution: context_resolved)
  - cite[55339:55408] (citations verified byte-exact at import): 'then does she get to farm no no she loses the whole wave then yes yes'
- [2] her → Syndra (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [3] you → Veigar (player) (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [4] we → unresolved (resolution: unresolved)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'

#### abilities_resources (4 items)
- [1] Syndra's Q (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] Syndra's E (resolution: context_resolved)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'
- [3] HP (resolution: literal_explicit)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'
- [4] whole wave (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'

#### events_actions (4 items)
- [1] she loses the whole wave (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'
- [2] she uses Q (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [3] she uses E (resolution: context_resolved)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'
- [4] you lose like 100 HP 150 (resolution: literal_explicit)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'

#### states (3 items)
- [1] she has no spell (resolution: context_resolved)
  - cite[55526:55556] (citations verified byte-exact at import): 'but then she has no spell yeah'
- [2] you lose like 100 HP 150 (resolution: literal_explicit)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'
- [3] she loses the whole wave (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'

#### conditions (2 items)
- [1] now that she uses Q (resolution: literal_explicit)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] if she uses E (resolution: context_resolved)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'

#### recommended_advice (1 items)
- [1] you should run at her now that she uses Q (resolution: literal_explicit)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"

#### consequences_outcomes (3 items)
- [1] she loses the whole wave (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'
- [2] you lose like 100 HP 150 (resolution: literal_explicit)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'
- [3] she has no spell (resolution: context_resolved)
  - cite[55526:55556] (citations verified byte-exact at import): 'but then she has no spell yeah'

#### explicit_relationships (7 items)
- [1] she → Syndra (resolution: context_resolved; relation: REFERS_TO)
  - cite[55339:55408] (citations verified byte-exact at import): 'then does she get to farm no no she loses the whole wave then yes yes'
- [2] her → Syndra (resolution: context_resolved; relation: REFERS_TO)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [3] you → Veigar (player) (resolution: context_resolved; relation: REFERS_TO)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [4] she uses Q (resolution: context_resolved; relation: USES)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [5] she uses E (resolution: context_resolved; relation: USES)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'
- [6] if she uses E (resolution: context_resolved; relation: CONDITION)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'
- [7] you lose like 100 HP 150 and she has no spell (resolution: context_resolved; relation: RESULT)
  - cite[55465:55556] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no spell yeah'

#### uncertainty_unresolved (1 items)
- [1] we (resolution: unresolved)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'

#### supporting_source_spans (2 items)
- [1] target passage span (resolution: literal_explicit)
  - cite[55339:55577] (citations verified byte-exact at import): "then does she get to farm no no she loses the whole wave then yes yes so that's why you should run at her n..."
- [2] Syndra matchup context (resolution: literal_explicit)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"

### TARGET p2k:case:0006

#### actors_entities (3 items)
- [1] you (resolution: context_resolved)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
- [2] Mel (resolution: context_resolved)
  - cite[4189:4200] (citations verified byte-exact at import): 'you are Mel'
- [3] other champions (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### reference_bindings (4 items)
- [1] you refers to the player being coached (resolution: context_resolved)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
- [2] it in it's a habit refers to the habit of stepping back or leaving the good position after farming (resolution: context_resolved)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
- [3] this in this is a extremely bad freeze frame refers to the freeze-frame situation caused by the bad habit (resolution: context_resolved)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
- [4] this in if this was refers to the same bad freeze-frame situation or ability use being shown (resolution: context_resolved)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### abilities_resources (2 items)
- [1] your queue (resolution: context_resolved)
  - cite[4223:4261] (citations verified byte-exact at import): 'your queue is to proc scorch and comet'
- [2] scorch and comet (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### events_actions (1 items)
- [1] you could just be here and land maybe two more ticks (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### states (2 items)
- [1] you could just be here (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
- [2] this is a extremely bad freeze frame (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### conditions (2 items)
- [1] when you pick up other champions (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
- [2] if this was (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### recommended_advice (2 items)
- [1] it's a habit you don't want to have (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
- [2] you could just be here and land maybe two more ticks (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### consequences_outcomes (1 items)
- [1] when you pick up other champions this is a extremely bad freeze frame (resolution: literal_explicit)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### explicit_relationships (1 items)
- [1] when you pick up other champions this is a extremely bad freeze frame (resolution: literal_explicit; relation: CONDITION)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### uncertainty_unresolved (1 items)
- [1] target passage cuts off at if this was, leaving the alternative ability condition unfinished (resolution: unresolved)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

#### supporting_source_spans (1 items)
- [1] Target passage and immediately preceding clause (resolution: literal_explicit)
  - cite[4171:4261] (citations verified byte-exact at import): 'because of course you are Mel and the main point of your queue is to proc scorch and comet'
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."

### TARGET p2k:case:0007

#### actors_entities (4 items)
- [1] Nami (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [2] Samira ("sa") (resolution: context_resolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [3] you (Lucian/player being coached) (resolution: context_resolved)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'
- [4] they (enemy bot lane, Samira and Nami) (resolution: context_resolved)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'

#### reference_bindings (8 items)
- [1] they → enemy bot lane (Samira and Nami) (resolution: context_resolved)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'
- [2] you → player/student (Lucian) (resolution: context_resolved)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'
- [3] this all in → the all-in/engage the enemy gives (resolution: context_resolved)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'
- [4] it → using Ignite now (resolution: context_resolved)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"
- [5] they → enemy bot lane (Samira and Nami) (resolution: context_resolved)
  - cite[9649:9718] (citations verified byte-exact at import): "when they're going to probably potion and maybe Nami will heal the sa"
- [6] the sa → Samira (resolution: context_resolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [7] she → Nami (resolution: context_resolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [8] I → coach/speaker (resolution: context_resolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### abilities_resources (3 items)
- [1] Ignite (resolution: literal_explicit)
  - cite[9600:9621] (citations verified byte-exact at import): 'you would ignite twoo'
- [2] Nami's W (resolution: context_resolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [3] potion (resolution: literal_explicit)
  - cite[9649:9686] (citations verified byte-exact at import): "when they're going to probably potion"

#### events_actions (4 items)
- [1] the enemy gives you an all-in on the freeze (resolution: literal_explicit)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'
- [2] you ignite them (resolution: literal_explicit)
  - cite[9600:9621] (citations verified byte-exact at import): 'you would ignite twoo'
- [3] the enemies are going to potion (resolution: literal_explicit)
  - cite[9649:9686] (citations verified byte-exact at import): "when they're going to probably potion"
- [4] Nami heals Samira (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'

#### states (2 items)
- [1] the wave is in a freeze (resolution: literal_explicit)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'
- [2] whether Nami has W available is uncertain (resolution: unresolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### conditions (2 items)
- [1] if they give you this all-in here on the freeze, then you would ignite (resolution: literal_explicit)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'
- [2] when they are going to potion and maybe Nami will heal Samira (resolution: literal_explicit)
  - cite[9649:9718] (citations verified byte-exact at import): "when they're going to probably potion and maybe Nami will heal the sa"

#### recommended_advice (1 items)
- [1] why not do it now when they're going to probably potion and maybe Nami will heal the sa (resolution: literal_explicit)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"

#### consequences_outcomes (2 items)
- [1] you would ignite them on a later all-in on the freeze (resolution: literal_explicit)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'
- [2] the enemies will probably potion and maybe Nami will heal Samira (resolution: literal_explicit)
  - cite[9649:9718] (citations verified byte-exact at import): "when they're going to probably potion and maybe Nami will heal the sa"

#### explicit_relationships (3 items)
- [1] you use Ignite (resolution: literal_explicit; relation: USES)
  - cite[9600:9621] (citations verified byte-exact at import): 'you would ignite twoo'
- [2] Nami heals Samira (resolution: literal_explicit; relation: AFFECTS)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [3] the enemy all-in on the freeze is a condition for you to ignite (resolution: literal_explicit; relation: CONDITION)
  - cite[9540:9621] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo'

#### uncertainty_unresolved (1 items)
- [1] whether Nami has W available (resolution: unresolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### supporting_source_spans (2 items)
- [1] Target passage (resolution: literal_explicit)
  - cite[9536:9758] (citations verified byte-exact at import): 'now next time they give you this all in here on the freeze then you would ignite twoo right so why not do i...'
- [2] Context for Nami healing Samira (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'

### TARGET p2k:case:0008

#### actors_entities (3 items)
- [1] Ambessa (resolution: literal_explicit)
  - cite[75904:75948] (citations verified byte-exact at import): 'Ambessa is by far better champion than Riven'
- [2] Riven (resolution: literal_explicit)
  - cite[76148:76169] (citations verified byte-exact at import): 'against Riven like if'
- [3] the student/player (you) (resolution: context_resolved)
  - cite[76019:76069] (citations verified byte-exact at import): "you feel like you're not able to progress the lane"

#### reference_bindings (5 items)
- [1] that -> Ambessa being a better champion than Riven (resolution: context_resolved)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [2] it -> Ambessa (resolution: context_resolved)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [3] you -> the student/player (resolution: context_resolved)
  - cite[76019:76069] (citations verified byte-exact at import): "you feel like you're not able to progress the lane"
- [4] it -> Ambessa (in 'ban it for sure') (resolution: context_resolved)
  - cite[76082:76108] (citations verified byte-exact at import): 'you should ban it for sure'
- [5] I -> the student/player (resolution: context_resolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

#### abilities_resources (0 items)

(none)

#### events_actions (2 items)
- [1] ban it for sure (resolution: context_resolved)
  - cite[76082:76108] (citations verified byte-exact at import): 'you should ban it for sure'
- [2] I might have to do that (resolution: context_resolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

#### states (2 items)
- [1] Ambessa is by far better champion than Riven (resolution: literal_explicit)
  - cite[75904:75948] (citations verified byte-exact at import): 'Ambessa is by far better champion than Riven'
- [2] you feel like you're not able to progress the lane (resolution: literal_explicit)
  - cite[76019:76069] (citations verified byte-exact at import): "you feel like you're not able to progress the lane"

#### conditions (1 items)
- [1] if on top of that, you feel like you're not able to progress the lane (resolution: literal_explicit)
  - cite[76000:76069] (citations verified byte-exact at import): "if on top of that, you feel like you're not able to progress the lane"

#### recommended_advice (1 items)
- [1] you should ban it for sure (resolution: literal_explicit)
  - cite[76082:76108] (citations verified byte-exact at import): 'you should ban it for sure'

#### consequences_outcomes (2 items)
- [1] So that's already a good reason to ban it. (resolution: literal_explicit)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [2] I might have to do that (resolution: literal_explicit)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

#### explicit_relationships (4 items)
- [1] it refers to Ambessa in 'ban it' (resolution: context_resolved; relation: REFERS_TO)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [2] you refers to the student/player (resolution: context_resolved; relation: REFERS_TO)
  - cite[76019:76069] (citations verified byte-exact at import): "you feel like you're not able to progress the lane"
- [3] Feeling unable to progress the lane conditions the advice to ban Ambessa (resolution: literal_explicit; relation: CONDITION)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
- [4] The advice to ban Ambessa for sure is the result of the condition (resolution: literal_explicit; relation: RESULT)
  - cite[76082:76108] (citations verified byte-exact at import): 'you should ban it for sure'

#### uncertainty_unresolved (1 items)
- [1] The student's reason stops at an unresolved/truncated condition: 'against Riven like if' (resolution: unresolved)
  - cite[76148:76169] (citations verified byte-exact at import): 'against Riven like if'

#### supporting_source_spans (4 items)
- [1] Ambessa is by far better champion than Riven (resolution: literal_explicit)
  - cite[75904:75948] (citations verified byte-exact at import): 'Ambessa is by far better champion than Riven'
- [2] So that's already a good reason to ban it. (resolution: literal_explicit)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [3] But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure. (resolution: literal_explicit)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
- [4] Yeah, I might have to do that because against Riven like if (resolution: literal_explicit)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

### TARGET p2k:case:0009

#### actors_entities (3 items)
- [1] Mel (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [2] Poppy (resolution: literal_explicit)
  - cite[56606:56630] (citations verified byte-exact at import): "she's not killing Poppy."
- [3] you (player being coached) (resolution: context_resolved)
  - cite[56724:56738] (citations verified byte-exact at import): 'So, you chase.'

#### reference_bindings (7 items)
- [1] it → Mel (resolution: context_resolved)
  - cite[56538:56563] (citations verified byte-exact at import): "then it's bot. Yes? Yeah."
- [2] She → Mel (in 'She's not a bot') (resolution: context_resolved)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
- [3] She → Mel (in 'She's not killing Poppy') (resolution: context_resolved)
  - cite[56606:56630] (citations verified byte-exact at import): "she's not killing Poppy."
- [4] She → Mel (in 'She's not on mid') (resolution: context_resolved)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [5] your → player being coached (resolution: context_resolved)
  - cite[56675:56723] (citations verified byte-exact at import): "But you don't play around waves being your ward."
- [6] you → player being coached (in 'you chase') (resolution: context_resolved)
  - cite[56724:56738] (citations verified byte-exact at import): 'So, you chase.'
- [7] this → chasing Mel when she shows mid (resolution: context_resolved)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### abilities_resources (2 items)
- [1] wave (resolution: literal_explicit)
  - cite[56649:56674] (citations verified byte-exact at import): 'because wave equals ward.'
- [2] ward (resolution: literal_explicit)
  - cite[56675:56723] (citations verified byte-exact at import): "But you don't play around waves being your ward."

#### events_actions (3 items)
- [1] You chase (resolution: literal_explicit)
  - cite[56724:56738] (citations verified byte-exact at import): 'So, you chase.'
- [2] Mel is not killing Poppy (resolution: literal_explicit)
  - cite[56606:56630] (citations verified byte-exact at import): "she's not killing Poppy."
- [3] you don't play around waves being your ward (resolution: literal_explicit)
  - cite[56675:56723] (citations verified byte-exact at import): "But you don't play around waves being your ward."

#### states (3 items)
- [1] Mel is not bot (resolution: literal_explicit)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
- [2] Mel is not on mid (resolution: literal_explicit)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [3] Wave equals ward (resolution: literal_explicit)
  - cite[56649:56674] (citations verified byte-exact at import): 'because wave equals ward.'

#### conditions (1 items)
- [1] If Mel is showing on mid (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### recommended_advice (0 items)

(none)

#### consequences_outcomes (3 items)
- [1] She's not a bot (resolution: literal_explicit)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
- [2] She's not on mid (resolution: literal_explicit)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [3] this could be a good flip (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### explicit_relationships (4 items)
- [1] Mel not killing Poppy causes the conclusion that Mel is not bot (resolution: literal_explicit; relation: CAUSES)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
- [2] Wave equals ward causes the conclusion that Mel is not on mid (resolution: literal_explicit; relation: CAUSES)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [3] Mel showing on mid is a condition for this being a good flip (resolution: literal_explicit; relation: CONDITION)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [4] She refers to Mel (resolution: context_resolved; relation: REFERS_TO)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."

#### uncertainty_unresolved (1 items)
- [1] this could be a good flip (speaker uncertainty: 'could' marks a possibility) (resolution: unresolved)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### supporting_source_spans (1 items)
- [1] then it's bot. Yes? Yeah. She's not a bot, because well, obviously, she's not killing Poppy. She's not on mid, because wave equals ward. But you don't play around waves being your ward. So, you chase. If Mel is showing on mid, this could be a good flip, (resolution: literal_explicit)
  - cite[56538:56791] (citations verified byte-exact at import): "then it's bot. Yes? Yeah. She's not a bot, because well, obviously, she's not killing Poppy. She's not on m..."

### TARGET p2k:case:0010

#### actors_entities (2 items)
- [1] you (the player being coached, playing Varus) (resolution: context_resolved)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [2] this (the Blitzcrank/Kalista matchup/lane) (resolution: context_resolved)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### reference_bindings (3 items)
- [1] you -> the player being coached (Varus player) (resolution: context_resolved)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [2] this -> the matchup/lane against Blitzcrank and Kalista (resolution: context_resolved)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [3] it -> push (resolution: context_resolved)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### abilities_resources (0 items)

(none)

#### events_actions (3 items)
- [1] you do get push (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [2] you could get it (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [3] you could win (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### states (3 items)
- [1] now you don't win (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [2] there's an angle to get push (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [3] you're not meant to win this (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### conditions (2 items)
- [1] if you do get push (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [2] if there's an angle to get push (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### recommended_advice (1 items)
- [1] if there's an angle to get push then you could get it then you could win (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### consequences_outcomes (2 items)
- [1] you could win (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [2] you're not meant to win this (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### explicit_relationships (2 items)
- [1] getting push enables winning (resolution: context_resolved; relation: ENABLES)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [2] an angle to get push is a condition for getting push and winning (resolution: context_resolved; relation: CONDITION)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### uncertainty_unresolved (1 items)
- [1] MH (resolution: unresolved)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

#### supporting_source_spans (1 items)
- [1] yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to get push then you could get it then you could win MH but you're not meant to win this you know but (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."

---

## Condition F (model opencode-go/deepseek-v4-flash)

### TARGET p2k:case:0001

#### actors_entities (4 items)
- [1] you (the player/student being coached) (resolution: context_resolved)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
- [2] I (the student/player who says 'I do') (resolution: context_resolved)
  - cite[5158:5179] (citations verified byte-exact at import): 'right yeah I do right'
- [3] I (the coach who would play with Flash) (resolution: context_resolved)
  - cite[4893:4965] (citations verified byte-exact at import): 'I would always play First Strike but but I would play with flash because'
- [4] Karthus (source surface 'carus') (resolution: context_resolved)
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'

#### reference_bindings (7 items)
- [1] 'you' refers to the player/student being coached (resolution: context_resolved)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
- [2] 'I' in 'I do' refers to the student/player responding (resolution: context_resolved)
  - cite[5158:5179] (citations verified byte-exact at import): 'right yeah I do right'
- [3] 'I' in 'I would' refers to the coach (resolution: context_resolved)
  - cite[4893:4965] (citations verified byte-exact at import): 'I would always play First Strike but but I would play with flash because'
- [4] 'SM' refers to Smite (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [5] 'Harvest' refers to Dark Harvest (resolution: context_resolved)
  - cite[5061:5082] (citations verified byte-exact at import): "you're losing Harvest"
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'
- [6] 'that' in 'that doesn't mean this is bad' has no resolvable antecedent in the target passage (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"
- [7] 'this' in 'that doesn't mean this is bad' has no clearly resolvable antecedent in the target passage (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"

#### abilities_resources (8 items)
- [1] Flash (summoner spell) (resolution: literal_explicit)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
- [2] Exhaust (summoner spell) (resolution: literal_explicit)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
- [3] Smite (summoner spell; source surface 'SM') (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [4] Last Stand (rune) (resolution: literal_explicit)
  - cite[5036:5060] (citations verified byte-exact at import): "you're losing Last Stand"
- [5] Dark Harvest (rune; source surface 'Harvest') (resolution: context_resolved)
  - cite[5061:5082] (citations verified byte-exact at import): "you're losing Harvest"
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'
- [6] Cheap Shot (rune) (resolution: literal_explicit)
  - cite[5083:5107] (citations verified byte-exact at import): "you're losing cheap shot"
- [7] Ultimate Hunter (rune) (resolution: literal_explicit)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [8] Eyeball (rune; source surface 'eyeball') (resolution: literal_explicit)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'

#### events_actions (5 items)
- [1] The player is going Exhaust and Smite. (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [2] The player has less fighting power. (resolution: literal_explicit)
  - cite[4999:5027] (citations verified byte-exact at import): 'you have less fighting Power'
- [3] The player is losing Last Stand, Harvest, and Cheap Shot. (resolution: literal_explicit)
  - cite[5036:5107] (citations verified byte-exact at import): "you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [4] The player probably has Ultimate Hunter and Eyeball. (resolution: literal_explicit)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [5] The student/player confirms with 'I do' that they have the runes. (resolution: context_resolved)
  - cite[5158:5179] (citations verified byte-exact at import): 'right yeah I do right'

#### states (4 items)
- [1] The player is going Exhaust and Smite. (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
- [2] The player has less fighting power. (resolution: literal_explicit)
  - cite[4999:5027] (citations verified byte-exact at import): 'you have less fighting Power'
- [3] The player is losing Last Stand, Harvest, and Cheap Shot. (resolution: literal_explicit)
  - cite[5036:5107] (citations verified byte-exact at import): "you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [4] The player probably has Ultimate Hunter and Eyeball. (resolution: literal_explicit)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'

#### conditions (1 items)
- [1] When the player is going Exhaust/Smite, they have less fighting power. (resolution: context_resolved)
  - cite[4971:5027] (citations verified byte-exact at import): "you're going exhaust SM but you have less fighting Power"

#### recommended_advice (2 items)
- [1] Play with Flash. (resolution: literal_explicit)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
- [2] The coach would play with Flash. (resolution: context_resolved)
  - cite[4893:4965] (citations verified byte-exact at import): 'I would always play First Strike but but I would play with flash because'

#### consequences_outcomes (3 items)
- [1] The player has less fighting power. (resolution: literal_explicit)
  - cite[4999:5107] (citations verified byte-exact at import): "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [2] The player is losing Last Stand, Harvest, and Cheap Shot. (resolution: literal_explicit)
  - cite[5036:5107] (citations verified byte-exact at import): "you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [3] That does not mean this is bad. (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"

#### explicit_relationships (8 items)
- [1] The player is the actor who goes Exhaust and Smite. (resolution: context_resolved; relation: ACTOR)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
- [2] The player is the actor addressed by 'play with flash'. (resolution: context_resolved; relation: ACTOR)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
- [3] The player uses Exhaust and Smite. (resolution: context_resolved; relation: USES)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [4] Losing Last Stand, Harvest, and Cheap Shot causes less fighting power. (resolution: literal_explicit; relation: CAUSES)
  - cite[4999:5107] (citations verified byte-exact at import): "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [5] If the player is going Exhaust/Smite, the player has less fighting power. (resolution: context_resolved; relation: CONDITION)
  - cite[4971:5027] (citations verified byte-exact at import): "you're going exhaust SM but you have less fighting Power"
- [6] 'SM' refers to Smite. (resolution: context_resolved; relation: REFERS_TO)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [7] 'Harvest' refers to Dark Harvest. (resolution: context_resolved; relation: REFERS_TO)
  - cite[5061:5082] (citations verified byte-exact at import): "you're losing Harvest"
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'
- [8] That does not mean this is bad. (resolution: unresolved; relation: NEGATES)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"

#### uncertainty_unresolved (6 items)
- [1] 'SM' is not spelled out in the target; context supports resolving it to Smite. (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [2] The antecedent of 'that' in 'that doesn't mean this is bad' is not resolvable from the target passage. (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"
- [3] The referent of 'this' in 'that doesn't mean this is bad' is not clearly resolvable from the target passage. (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"
- [4] The target passage ends with the incomplete phrase 'like I would'; the intended continuation is not in the target passage. (resolution: unresolved)
  - cite[5180:5222] (citations verified byte-exact at import): "that doesn't mean this is bad like I would"
- [5] The source says the player 'probably' has Ultimate Hunter and Eyeball, expressing uncertainty. (resolution: literal_explicit)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [6] 'Harvest' is written without 'Dark' in the target; context suggests Dark Harvest. (resolution: context_resolved)
  - cite[5061:5082] (citations verified byte-exact at import): "you're losing Harvest"
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'

#### supporting_source_spans (10 items)
- [1] play with flash because like you're going exhaust SM (resolution: literal_explicit)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
- [2] you're going exhaust SM but you have less fighting Power (resolution: literal_explicit)
  - cite[4971:5027] (citations verified byte-exact at import): "you're going exhaust SM but you have less fighting Power"
- [3] you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot (resolution: literal_explicit)
  - cite[4999:5107] (citations verified byte-exact at import): "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [4] you're losing Last Stand you're losing Harvest you're losing cheap shot (resolution: literal_explicit)
  - cite[5036:5107] (citations verified byte-exact at import): "you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [5] you probably have ultimate Hunter eyeball (resolution: literal_explicit)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [6] right yeah I do right (resolution: literal_explicit)
  - cite[5158:5179] (citations verified byte-exact at import): 'right yeah I do right'
- [7] that doesn't mean this is bad like I would (resolution: literal_explicit)
  - cite[5180:5222] (citations verified byte-exact at import): "that doesn't mean this is bad like I would"
- [8] I would always play First Strike but but I would play with flash because (resolution: literal_explicit)
  - cite[4893:4965] (citations verified byte-exact at import): 'I would always play First Strike but but I would play with flash because'
- [9] for carus you are never allowed to go Smite and exhaust again (resolution: literal_explicit)
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [10] you should at least go dark Harvest (resolution: literal_explicit)
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'

### TARGET p2k:case:0002

#### actors_entities (2 items)
- [1] atrox (Aatrox) (resolution: vocabulary_supported)
  - cite[30531:30571] (citations verified byte-exact at import): 'look at atrox atrox has 11 CS after this'
- [2] darus (Darius) (resolution: vocabulary_supported)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'

#### reference_bindings (6 items)
- [1] he ('he' in 'if at he's giving a freeb') (resolution: unresolved)
  - cite[30572:30597] (citations verified byte-exact at import): "if at he's giving a freeb"
- [2] this ('after this') (resolution: unresolved)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'
- [3] this ('this darus') -> darus (Darius) (resolution: context_resolved)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [4] you ('do you really think') -> addressee/student being coached (resolution: context_resolved)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [5] I ('I don't think so') -> coach/speaker (resolution: context_resolved)
  - cite[30738:30762] (citations verified byte-exact at import): "because I don't think so"
- [6] this ('because this if yeah') (resolution: unresolved)
  - cite[30604:30624] (citations verified byte-exact at import): 'because this if yeah'

#### abilities_resources (4 items)
- [1] 11 CS (Aatrox's) (resolution: context_resolved)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'
- [2] full HP (Aatrox's) (resolution: context_resolved)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [3] clo armor (Cloth Armor?, Aatrox's item) (resolution: unresolved)
  - cite[30680:30689] (citations verified byte-exact at import): 'clo armor'
- [4] T (Teleport; Aatrox's summoner spell) (resolution: context_resolved)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'

#### events_actions (2 items)
- [1] if atrox T is back here (Aatrox returns via Teleport) (resolution: context_resolved)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
- [2] he's giving a freeb (resolution: unresolved)
  - cite[30572:30597] (citations verified byte-exact at import): "if at he's giving a freeb"

#### states (2 items)
- [1] atrox (Aatrox) has 11 CS after this (resolution: literal_explicit)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'
- [2] atrox (Aatrox) with clo armor full HP (hypothetical state) (resolution: literal_explicit)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'

#### conditions (3 items)
- [1] if atrox T is back here (resolution: context_resolved)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
- [2] if at he's giving a freeb (resolution: unresolved)
  - cite[30572:30597] (citations verified byte-exact at import): "if at he's giving a freeb"
- [3] with clo armor full HP (condition attached to Aatrox in matchup question) (resolution: literal_explicit)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'

#### recommended_advice (2 items)
- [1] Look at atrox: atrox has 11 CS after this; if at he's giving a freeb. (resolution: unresolved)
  - cite[30531:30603] (citations verified byte-exact at import): "look at atrox atrox has 11 CS after this if at he's giving a freeb right"
- [2] Consider whether atrox with clo armor full HP will beat this darus; it is probably not right. (resolution: literal_explicit)
  - cite[30649:30762] (citations verified byte-exact at import): "do you really think atrox with clo armor full HP will beat this darus probably not right because I don't th..."

#### consequences_outcomes (2 items)
- [1] atrox with clo armor full HP probably will not beat this darus (resolution: literal_explicit)
  - cite[30649:30762] (citations verified byte-exact at import): "do you really think atrox with clo armor full HP will beat this darus probably not right because I don't th..."
- [2] he's giving a freeb (unresolved consequence) (resolution: unresolved)
  - cite[30572:30597] (citations verified byte-exact at import): "if at he's giving a freeb"

#### explicit_relationships (4 items)
- [1] if atrox T is back here conditions the question of whether atrox with clo armor full HP will beat this darus (resolution: context_resolved; relation: CONDITION)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [2] the proposition that atrox with clo armor full HP will beat this darus is negated ('probably not right') (resolution: literal_explicit; relation: NEGATES)
  - cite[30649:30762] (citations verified byte-exact at import): "do you really think atrox with clo armor full HP will beat this darus probably not right because I don't th..."
- [3] this darus refers to darus (Darius) (resolution: context_resolved; relation: REFERS_TO)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [4] atrox is the target of the coach's directive 'look at atrox' (resolution: literal_explicit; relation: TARGET)
  - cite[30531:30571] (citations verified byte-exact at import): 'look at atrox atrox has 11 CS after this'

#### uncertainty_unresolved (4 items)
- [1] freeb ('he's giving a freeb') (resolution: unresolved)
  - cite[30572:30597] (citations verified byte-exact at import): "if at he's giving a freeb"
- [2] if at he's giving a freeb right ('at' fragment and 'he' antecedent) (resolution: unresolved)
  - cite[30572:30597] (citations verified byte-exact at import): "if at he's giving a freeb"
- [3] clo armor (possible Cloth Armor, not spelled out) (resolution: unresolved)
  - cite[30680:30689] (citations verified byte-exact at import): 'clo armor'
- [4] because this if yeah (deictic/filler) (resolution: unresolved)
  - cite[30604:30624] (citations verified byte-exact at import): 'because this if yeah'

#### supporting_source_spans (3 items)
- [1] look at atrox atrox has 11 CS after this if at he's giving a freeb right (resolution: literal_explicit)
  - cite[30531:30603] (citations verified byte-exact at import): "look at atrox atrox has 11 CS after this if at he's giving a freeb right"
- [2] because this if yeah if atrox T is back here do you really think atrox with clo armor full HP will beat this darus (resolution: literal_explicit)
  - cite[30604:30718] (citations verified byte-exact at import): 'because this if yeah if atrox T is back here do you really think atrox with clo armor full HP will beat thi...'
- [3] probably not right because I don't think so (resolution: literal_explicit)
  - cite[30719:30762] (citations verified byte-exact at import): "probably not right because I don't think so"

### TARGET p2k:case:0003

#### actors_entities (4 items)
- [1] Fiora (champion) (resolution: literal_explicit)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [2] Camille (champion) (resolution: literal_explicit)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
- [3] Coach/speaker ('I') (resolution: context_resolved)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [4] Player being coached ('you') (resolution: context_resolved)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'

#### reference_bindings (3 items)
- [1] 'I' refers to the coach giving the advice. (resolution: context_resolved)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [2] 'she' in 'she heals more than [ __ ] window' refers to Fiora. (resolution: context_resolved)
  - cite[13360:13401] (citations verified byte-exact at import): 'because she heals more than [\xa0__\xa0] window'
- [3] 'you' in 'when you need heal cut' refers to the player being coached. (resolution: context_resolved)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'

#### abilities_resources (9 items)
- [1] Ignite (summoner spell) (resolution: literal_explicit)
  - cite[13330:13359] (citations verified byte-exact at import): 'always go ignite versus Fiora'
- [2] Flash (summoner spell) (resolution: literal_explicit)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
- [3] TP (Teleport) (resolution: vocabulary_supported)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [4] Heal cut (resolution: literal_explicit)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'
- [5] Fiora's healing (resolution: context_resolved)
  - cite[13360:13401] (citations verified byte-exact at import): 'because she heals more than [\xa0__\xa0] window'
- [6] Executioner's (executioners) (resolution: literal_explicit)
  - cite[13541:13583] (citations verified byte-exact at import): "you can't buy executioners on this channel"
- [7] Trinity (resolution: literal_explicit)
  - cite[13619:13663] (citations verified byte-exact at import): 'you need Trinity and you need Hydra too much'
- [8] Hydra (resolution: literal_explicit)
  - cite[13619:13663] (citations verified byte-exact at import): 'you need Trinity and you need Hydra too much'
- [9] Camille E + Flash combo (resolution: context_resolved)
  - cite[13749:13793] (citations verified byte-exact at import): 'otherwise always flash cuz e flash is too op'

#### events_actions (0 items)

(none)

#### states (4 items)
- [1] Fiora heals more than [ __ ] window (unresolved comparison). (resolution: unresolved)
  - cite[13360:13401] (citations verified byte-exact at import): 'because she heals more than [\xa0__\xa0] window'
- [2] Camille cannot buy Executioner's on this channel. (resolution: literal_explicit)
  - cite[13541:13583] (citations verified byte-exact at import): "you can't buy executioners on this channel"
- [3] Camille needs Trinity and Hydra too much to buy heal cut like a normal champion. (resolution: literal_explicit)
  - cite[13619:13663] (citations verified byte-exact at import): 'you need Trinity and you need Hydra too much'
  - cite[13667:13712] (citations verified byte-exact at import): "you can't buy heal cut like a normal Champion"
- [4] Flash is considered broken on Camille; E-flash is considered too strong. (resolution: literal_explicit)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
  - cite[13749:13793] (citations verified byte-exact at import): 'otherwise always flash cuz e flash is too op'

#### conditions (6 items)
- [1] Going Ignite is recommended versus Fiora. (resolution: literal_explicit)
  - cite[13330:13359] (citations verified byte-exact at import): 'always go ignite versus Fiora'
- [2] Fiora healing more than [ __ ] window is the reason/condition for going Ignite versus Fiora. (resolution: unresolved)
  - cite[13360:13401] (citations verified byte-exact at import): 'because she heals more than [\xa0__\xa0] window'
- [3] In most cases is the condition for going Flash TP. (resolution: literal_explicit)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [4] When you need heal cut is the condition under which Ignite is fine. (resolution: literal_explicit)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'
- [5] If you need heal cut, Ignite; otherwise, always Flash. (resolution: literal_explicit)
  - cite[13721:13793] (citations verified byte-exact at import): 'ignite if you need heal cut otherwise always flash cuz e flash is too op'
- [6] Because Camille cannot buy Executioner's, Ignite is the heal-cut option. (resolution: context_resolved)
  - cite[13541:13583] (citations verified byte-exact at import): "you can't buy executioners on this channel"
  - cite[13721:13793] (citations verified byte-exact at import): 'ignite if you need heal cut otherwise always flash cuz e flash is too op'

#### recommended_advice (4 items)
- [1] Always go Ignite versus Fiora. (resolution: literal_explicit)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [2] Go Flash TP in most cases. (resolution: literal_explicit)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [3] Ignite is fine when you need heal cut. (resolution: literal_explicit)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'
- [4] Use Ignite if you need heal cut; otherwise always Flash. (resolution: literal_explicit)
  - cite[13721:13793] (citations verified byte-exact at import): 'ignite if you need heal cut otherwise always flash cuz e flash is too op'

#### consequences_outcomes (3 items)
- [1] Taking Ignite covers the need for heal cut. (resolution: context_resolved)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'
- [2] Because Camille cannot buy Executioner's, Ignite is the way to get heal cut. (resolution: context_resolved)
  - cite[13541:13583] (citations verified byte-exact at import): "you can't buy executioners on this channel"
  - cite[13721:13793] (citations verified byte-exact at import): 'ignite if you need heal cut otherwise always flash cuz e flash is too op'
- [3] Choosing Flash TP in most cases includes Flash, which is broken on Camille. (resolution: context_resolved)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'

#### explicit_relationships (11 items)
- [1] The coach ('I') is the actor who would go Ignite versus Fiora. (resolution: context_resolved; relation: ACTOR)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [2] Fiora is the target of the Ignite choice ('versus Fiora'). (resolution: literal_explicit; relation: TARGET)
  - cite[13330:13359] (citations verified byte-exact at import): 'always go ignite versus Fiora'
- [3] The Camille player is advised to use Flash and TP. (resolution: context_resolved; relation: USES)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [4] Fiora healing more than [ __ ] window causes the recommendation to go Ignite versus Fiora. (resolution: unresolved; relation: CAUSES)
  - cite[13330:13359] (citations verified byte-exact at import): 'always go ignite versus Fiora'
  - cite[13360:13401] (citations verified byte-exact at import): 'because she heals more than [\xa0__\xa0] window'
- [5] Versus Fiora is the condition for always going Ignite. (resolution: literal_explicit; relation: CONDITION)
  - cite[13330:13359] (citations verified byte-exact at import): 'always go ignite versus Fiora'
- [6] In most cases is the condition for going Flash TP. (resolution: literal_explicit; relation: CONDITION)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [7] When you need heal cut is the condition under which Ignite is fine. (resolution: literal_explicit; relation: CONDITION)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'
- [8] Ignite enables heal cut when it is needed. (resolution: context_resolved; relation: ENABLES)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'
- [9] Camille's inability to buy Executioner's causes Ignite to be used for heal cut. (resolution: context_resolved; relation: CAUSES)
  - cite[13541:13583] (citations verified byte-exact at import): "you can't buy executioners on this channel"
  - cite[13721:13793] (citations verified byte-exact at import): 'ignite if you need heal cut otherwise always flash cuz e flash is too op'
- [10] Flash being broken / E-flash being too strong causes Flash to be the default otherwise. (resolution: literal_explicit; relation: CAUSES)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
  - cite[13749:13793] (citations verified byte-exact at import): 'otherwise always flash cuz e flash is too op'
- [11] 'she' in 'she heals more than [ __ ] window' refers to Fiora. (resolution: context_resolved; relation: REFERS_TO)
  - cite[13360:13401] (citations verified byte-exact at import): 'because she heals more than [\xa0__\xa0] window'

#### uncertainty_unresolved (3 items)
- [1] The '[ __ ]' in 'heals more than [ __ ] window' is an ASR corruption; the comparison target is unrecoverable. (resolution: unresolved)
  - cite[13378:13401] (citations verified byte-exact at import): 'more than [\xa0__\xa0] window'
- [2] The target passage ends with 'because'; the isolated target span is incomplete, though the following transcript span completes the reason. (resolution: unresolved)
  - cite[13487:13540] (citations verified byte-exact at import): 'I think ignite when you need heal cut is fine because'
- [3] 'this channel' in 'on this channel' may be an ASR variant of 'this champ', but the supplied text is not corrected. (resolution: unresolved)
  - cite[13541:13583] (citations verified byte-exact at import): "you can't buy executioners on this channel"

#### supporting_source_spans (2 items)
- [1] Target passage span: Ignite versus Fiora; Flash TP in most cases; Ignite for heal cut. (resolution: literal_explicit)
  - cite[13315:13540] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window but I think flash is...'
- [2] Continuation span: Camille cannot buy Executioner's, needs Trinity/Hydra, so Ignite if heal cut and otherwise Flash. (resolution: literal_explicit)
  - cite[13541:13793] (citations verified byte-exact at import): "you can't buy executioners on this channel yeah you need other items too much you need Trinity and you need..."

### TARGET p2k:case:0004

#### actors_entities (6 items)
- [1] Brier (resolution: literal_explicit)
  - cite[54345:54373] (citations verified byte-exact at import): "if Brier doesn't win one one"
- [2] Briar (allied jungler) (resolution: vocabulary_supported)
  - cite[54345:54373] (citations verified byte-exact at import): "if Brier doesn't win one one"
  - cite[50530:50561] (citations verified byte-exact at import): 'your Brier is right next to you'
- [3] Talia (resolution: literal_explicit)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
- [4] Taliyah (enemy jungler) (resolution: vocabulary_supported)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
  - cite[54177:54203] (citations verified byte-exact at import): 'if Talia is invading Brier'
- [5] you (the coached player) (resolution: context_resolved)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
  - cite[489:541] (citations verified byte-exact at import): 'you also play the gar quite a bit yourself too right'
- [6] I (the coach) (resolution: context_resolved)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
  - cite[969:1030] (citations verified byte-exact at import): 'I think that conquer on atrox is obviously really really good'

#### reference_bindings (6 items)
- [1] 'you' binds to the coached player/student being given advice (resolution: context_resolved)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
  - cite[54401:54425] (citations verified byte-exact at import): 'you still want to invade'
  - cite[489:541] (citations verified byte-exact at import): 'you also play the gar quite a bit yourself too right'
- [2] 'I' binds to the coach speaking (resolution: context_resolved)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
  - cite[969:1030] (citations verified byte-exact at import): 'I think that conquer on atrox is obviously really really good'
- [3] 'it' in 'make it winning' binds to the invade (resolution: context_resolved)
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'
  - cite[54401:54425] (citations verified byte-exact at import): 'you still want to invade'
- [4] 'Brier' binds to Briar, the allied jungler (resolution: vocabulary_supported)
  - cite[54345:54373] (citations verified byte-exact at import): "if Brier doesn't win one one"
  - cite[50530:50561] (citations verified byte-exact at import): 'your Brier is right next to you'
- [5] 'Talia' binds to Taliyah, the enemy jungler (resolution: vocabulary_supported)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
  - cite[54177:54203] (citations verified byte-exact at import): 'if Talia is invading Brier'
- [6] 'one one' refers to a 1v1 fight; the opponent is not named in the target passage (resolution: unresolved)
  - cite[54345:54373] (citations verified byte-exact at import): "if Brier doesn't win one one"

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (0 items)

(none)

#### conditions (3 items)
- [1] If Brier doesn't win one one, then you should go. (resolution: literal_explicit)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
- [2] Even if Brier loses, you still want to invade. (resolution: context_resolved)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"
  - cite[54401:54425] (citations verified byte-exact at import): 'you still want to invade'
- [3] Because you could make the invade winning, you should still invade. (resolution: context_resolved)
  - cite[54401:54425] (citations verified byte-exact at import): 'you still want to invade'
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'

#### recommended_advice (3 items)
- [1] You should go if Brier doesn't win the 1v1. (resolution: context_resolved)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
- [2] You still want to invade. (resolution: literal_explicit)
  - cite[54401:54425] (citations verified byte-exact at import): 'you still want to invade'
- [3] Brier losing does not mean you should not invade. (resolution: literal_explicit)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"

#### consequences_outcomes (3 items)
- [1] You could make the invade winning. (resolution: context_resolved)
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'
- [2] Brier losing does not result in you not invading. (resolution: literal_explicit)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"
- [3] If Brier doesn't win one one, the advised outcome is that you go. (resolution: literal_explicit)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"

#### explicit_relationships (6 items)
- [1] Brier not winning one one is a condition for you to go. (resolution: literal_explicit; relation: CONDITION)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
- [2] Brier losing negates the inference that you should not invade. (resolution: literal_explicit; relation: NEGATES)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"
- [3] The possibility of making the invade winning is the reason you still want to invade. (resolution: context_resolved; relation: CAUSES)
  - cite[54401:54425] (citations verified byte-exact at import): 'you still want to invade'
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'
- [4] Taliyah's 2v1 would target you. (resolution: context_resolved; relation: TARGET)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
- [5] Brier is the actor who does not win one one. (resolution: literal_explicit; relation: ACTOR)
  - cite[54345:54373] (citations verified byte-exact at import): "if Brier doesn't win one one"
- [6] It in make it winning refers to the invade. (resolution: context_resolved; relation: REFERS_TO)
  - cite[54401:54425] (citations verified byte-exact at import): 'you still want to invade'
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'

#### uncertainty_unresolved (4 items)
- [1] The object of 'Brier loses to' is omitted; the target does not say whom Brier loses to. (resolution: unresolved)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"
- [2] '2v one' is a nonstandard rendering of '2v1'. (resolution: unresolved)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
- [3] 'one one' is a nonstandard rendering of '1v1'. (resolution: unresolved)
  - cite[54345:54373] (citations verified byte-exact at import): "if Brier doesn't win one one"
- [4] 'move' at the start of the target passage is a fragment; the subject of 'move' is outside the target passage and not resolved. (resolution: unresolved)
  - cite[54321:54344] (citations verified byte-exact at import): 'move then you should go'

#### supporting_source_spans (6 items)
- [1] move then you should go if Brier doesn't win one one then you should go because you still want to invade just because Brier loses to doesn't mean you shouldn't invade cuz you could make it winning right I hope Talia does not 2v one you right but (resolution: literal_explicit)
  - cite[54321:54566] (citations verified byte-exact at import): "move then you should go if Brier doesn't win one one then you should go because you still want to invade ju..."
- [2] if Brier doesn't win one one (resolution: literal_explicit)
  - cite[54345:54373] (citations verified byte-exact at import): "if Brier doesn't win one one"
- [3] you still want to invade (resolution: literal_explicit)
  - cite[54401:54425] (citations verified byte-exact at import): 'you still want to invade'
- [4] just because Brier loses to doesn't mean you shouldn't invade (resolution: literal_explicit)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"
- [5] cuz you could make it winning (resolution: literal_explicit)
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'
- [6] I hope Talia does not 2v one you (resolution: literal_explicit)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'

### TARGET p2k:case:0005

#### actors_entities (3 items)
- [1] Syndra (enemy mid laner; referred to as 'she'/'her' in the target) (resolution: context_resolved)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] the coached player (the mid laner being addressed as 'you') (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [3] the whole wave (the minion wave Syndra would lose) (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'

#### reference_bindings (3 items)
- [1] 'she' and 'her' → Syndra (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
- [2] 'you' (in 'you should run' and 'you lose') → the coached player (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
- [3] 'we' in 'right we then you lose' → unresolved (possibly non-referential or ASR artifact) (resolution: unresolved)
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'

#### abilities_resources (4 items)
- [1] Q (Syndra's Q / Dark Sphere) (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] E (Syndra's E / Scatter the Weak) (resolution: context_resolved)
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
- [3] spell (generic ability reference; in context the E she has just used) (resolution: context_resolved)
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'
- [4] HP (health resource): lose like 100 HP 150 (resolution: literal_explicit)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'

#### events_actions (3 items)
- [1] She uses Q (the trigger for running at her) (resolution: literal_explicit)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] She uses E (hypothetical in the 'if' clause) (resolution: literal_explicit)
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
- [3] She loses the whole wave (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'

#### states (3 items)
- [1] Syndra does not get to farm (the whole wave is lost to her) (resolution: literal_explicit)
  - cite[55339:55408] (citations verified byte-exact at import): 'then does she get to farm no no she loses the whole wave then yes yes'
- [2] You lose about 100-150 HP (resolution: literal_explicit)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'
- [3] Syndra has no spell available after using E (resolution: context_resolved)
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'

#### conditions (2 items)
- [1] Now that she uses Q is the condition under which you should run at her. (resolution: literal_explicit)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] If she uses E, the conditional expectation is that you lose about 100-150 HP and she has no spell. (resolution: literal_explicit)
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'

#### recommended_advice (1 items)
- [1] You should run at her now that she uses Q. (resolution: literal_explicit)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"

#### consequences_outcomes (3 items)
- [1] She loses the whole wave. (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'
- [2] You lose about 100-150 HP. (resolution: literal_explicit)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'
- [3] Then she has no spell. (resolution: literal_explicit)
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'

#### explicit_relationships (5 items)
- [1] Syndra uses Q. (resolution: context_resolved; relation: USES)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] Syndra uses E. (resolution: context_resolved; relation: USES)
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
- [3] Syndra using Q is the condition for running at her. (resolution: context_resolved; relation: CONDITION)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [4] If Syndra uses E, the result is you lose about 100-150 HP and then she has no spell. (resolution: context_resolved; relation: RESULT)
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'
- [5] The stated outcome in the scenario is that she loses the whole wave. (resolution: literal_explicit; relation: RESULT)
  - cite[55339:55408] (citations verified byte-exact at import): 'then does she get to farm no no she loses the whole wave then yes yes'

#### uncertainty_unresolved (2 items)
- [1] 'we' in 'right we then you lose' is unresolved; possibly non-referential or ASR artifact. (resolution: unresolved)
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
- [2] The surrounding word 'way' in 'I'm way versus syndra' is not verifiably a champion name; no champion binding is asserted for the player's champion. (resolution: unresolved)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"

#### supporting_source_spans (5 items)
- [1] I'm way versus syndra (resolution: literal_explicit)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [2] then does she get to farm no no she loses the whole wave then yes yes (resolution: literal_explicit)
  - cite[55339:55408] (citations verified byte-exact at import): 'then does she get to farm no no she loses the whole wave then yes yes'
- [3] so that's why you should run at her now that she uses Q (resolution: literal_explicit)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [4] because if she uses e right we then you lose like 100 HP 150 but then she has no (resolution: literal_explicit)
  - cite[55465:55545] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no'
- [5] but then she has no spell (resolution: literal_explicit)
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'

### TARGET p2k:case:0006

#### actors_entities (3 items)
- [1] you (the player being coached) (resolution: context_resolved)
  - cite[4189:4471] (citations verified byte-exact at import): "you are Mel and the main point of your queue is to proc scorch and comet right but it's a habit you don't w..."
- [2] Mel (the champion the player is playing) (resolution: literal_explicit)
  - cite[4189:4200] (citations verified byte-exact at import): 'you are Mel'
- [3] other champions (generic set; no specific champions named) (resolution: literal_explicit)
  - cite[4382:4414] (citations verified byte-exact at import): 'when you pick up other champions'

#### reference_bindings (6 items)
- [1] 'you' resolves to the player being coached (resolution: context_resolved)
  - cite[4189:4471] (citations verified byte-exact at import): "you are Mel and the main point of your queue is to proc scorch and comet right but it's a habit you don't w..."
- [2] 'your' in 'your queue' resolves to Mel's Q ability (resolution: context_resolved)
  - cite[4189:4471] (citations verified byte-exact at import): "you are Mel and the main point of your queue is to proc scorch and comet right but it's a habit you don't w..."
- [3] 'it' in 'it's a habit' resolves to the habit of not staying in position to land more Q ticks (resolution: context_resolved)
  - cite[4268:4374] (citations verified byte-exact at import): "but it's a habit you don't want to have because you could just be here and land maybe two more ticks right"
- [4] 'this' in 'this is a extremely bad freeze frame' resolves to the current freeze frame being reviewed (resolution: context_resolved)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'
- [5] 'this' in 'if this was' is unresolved because the target passage cuts off (resolution: unresolved)
  - cite[4452:4471] (citations verified byte-exact at import): 'because if this was'
- [6] 'other champions' is a generic reference, not bound to specific champions (resolution: unresolved)
  - cite[4382:4414] (citations verified byte-exact at import): 'when you pick up other champions'

#### abilities_resources (2 items)
- [1] Mel's Q ability, referred to in the source as 'your queue' (resolution: vocabulary_supported)
  - cite[4189:4471] (citations verified byte-exact at import): "you are Mel and the main point of your queue is to proc scorch and comet right but it's a habit you don't w..."
- [2] Scorch and Comet (the procs named 'scorch and comet'; source does not label them as runes) (resolution: vocabulary_supported)
  - cite[4245:4261] (citations verified byte-exact at import): 'scorch and comet'

#### events_actions (3 items)
- [1] The player's Q is meant to proc Scorch and Comet. (resolution: literal_explicit)
  - cite[4209:4267] (citations verified byte-exact at import): 'main point of your queue is to proc scorch and comet right'
- [2] The player could stay here and land maybe two more ticks. (resolution: literal_explicit)
  - cite[4316:4374] (citations verified byte-exact at import): 'you could just be here and land maybe two more ticks right'
- [3] The player picks up other champions. (resolution: literal_explicit)
  - cite[4382:4414] (citations verified byte-exact at import): 'when you pick up other champions'

#### states (2 items)
- [1] The current freeze frame is extremely bad. (resolution: literal_explicit)
  - cite[4415:4451] (citations verified byte-exact at import): 'this is a extremely bad freeze frame'
- [2] The player has a habit the coach says they do not want to have. (resolution: context_resolved)
  - cite[4268:4307] (citations verified byte-exact at import): "but it's a habit you don't want to have"

#### conditions (2 items)
- [1] When the player picks up other champions, this freeze frame is extremely bad. (resolution: literal_explicit)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'
- [2] An incomplete condition begins with 'if this was'; the comparison is not completed in the target passage. (resolution: unresolved)
  - cite[4452:4471] (citations verified byte-exact at import): 'because if this was'

#### recommended_advice (1 items)
- [1] Avoid the habit; instead be here and land maybe two more ticks. (resolution: context_resolved)
  - cite[4268:4374] (citations verified byte-exact at import): "but it's a habit you don't want to have because you could just be here and land maybe two more ticks right"

#### consequences_outcomes (2 items)
- [1] If the player keeps the habit, they miss the extra Q ticks they could land by being here. (resolution: context_resolved)
  - cite[4285:4374] (citations verified byte-exact at import): "you don't want to have because you could just be here and land maybe two more ticks right"
- [2] For other champions, the same freeze frame is extremely bad; the consequence after 'if this was' is cut off in the target passage. (resolution: unresolved)
  - cite[4382:4471] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame because if this was'

#### explicit_relationships (3 items)
- [1] Mel's Q is used to proc Scorch and Comet. (resolution: vocabulary_supported; relation: USES)
  - cite[4209:4267] (citations verified byte-exact at import): 'main point of your queue is to proc scorch and comet right'
- [2] Picking up other champions is a condition under which this freeze frame is extremely bad. (resolution: literal_explicit; relation: CONDITION)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'
- [3] Being here enables the player to land maybe two more Q ticks. (resolution: context_resolved; relation: ENABLES)
  - cite[4316:4374] (citations verified byte-exact at import): 'you could just be here and land maybe two more ticks right'

#### uncertainty_unresolved (3 items)
- [1] The marked passage is a fragment beginning mid-sentence at 'is to proc...' and ending at 'because if this was'; the intended comparison after 'if this was' is not named in the marked passage. (resolution: unresolved)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
- [2] The identity of 'other champions' is unspecified; no specific champion names are recoverable from the target passage. (resolution: unresolved)
  - cite[4382:4414] (citations verified byte-exact at import): 'when you pick up other champions'
- [3] The source spells 'your queue'; it is interpreted as Mel's Q from context, but the letters 'Q' are not present in the quoted span. (resolution: vocabulary_supported)
  - cite[4209:4267] (citations verified byte-exact at import): 'main point of your queue is to proc scorch and comet right'

#### supporting_source_spans (2 items)
- [1] Target passage span, including the immediately preceding context that identifies Mel as the champion. (resolution: context_resolved)
  - cite[4189:4471] (citations verified byte-exact at import): "you are Mel and the main point of your queue is to proc scorch and comet right but it's a habit you don't w..."
- [2] Additional supporting source spans for the conditional bad-freeze-frame clause and the better-position option. (resolution: literal_explicit)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'
  - cite[4316:4374] (citations verified byte-exact at import): 'you could just be here and land maybe two more ticks right'

### TARGET p2k:case:0007

#### actors_entities (4 items)
- [1] the coached player ('you'), the Lucian ADC being coached (resolution: context_resolved)
  - cite[8100:8169] (citations verified byte-exact at import): "you're Lucian Milo they're saami the lane that get push here is happy"
  - cite[9600:9627] (citations verified byte-exact at import): 'you would ignite twoo right'
- [2] the enemy bot lane ('they') that gives the all-in on the freeze (resolution: context_resolved)
  - cite[9550:9594] (citations verified byte-exact at import): 'they give you this all in here on the freeze'
- [3] Nami, the enemy support who may heal the sa (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [4] sa (Samira), the enemy ADC who is expected to potion and is the target of Nami's possible heal (resolution: context_resolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
  - cite[8119:8169] (citations verified byte-exact at import): "they're saami the lane that get push here is happy"

#### reference_bindings (10 items)
- [1] 'I' in 'I don't know if she has W' -> the coach/speaker (resolution: context_resolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [2] 'you' (in 'they give you this all in here', 'you would ignite twoo right', 'why not do it now') -> the coached player (Lucian ADC) (resolution: context_resolved)
  - cite[9550:9594] (citations verified byte-exact at import): 'they give you this all in here on the freeze'
  - cite[9600:9627] (citations verified byte-exact at import): 'you would ignite twoo right'
  - cite[9631:9648] (citations verified byte-exact at import): 'why not do it now'
- [3] 'they' (in 'they give you this all in here on the freeze') -> the enemy bot lane (resolution: context_resolved)
  - cite[9550:9594] (citations verified byte-exact at import): 'they give you this all in here on the freeze'
- [4] 'they' (in 'they're going to probably potion') -> the sa/Samira, likely the champion who will use the potion (resolution: context_resolved)
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"
- [5] 'this all in' -> the all-in on the freeze (resolution: context_resolved)
  - cite[9550:9594] (citations verified byte-exact at import): 'they give you this all in here on the freeze'
- [6] 'here' -> the freeze/lane position being discussed (resolution: context_resolved)
  - cite[9569:9594] (citations verified byte-exact at import): 'all in here on the freeze'
- [7] 'it' in 'why not do it now' -> using Ignite (resolution: context_resolved)
  - cite[9631:9648] (citations verified byte-exact at import): 'why not do it now'
- [8] 'the sa' -> Samira (enemy ADC) (resolution: context_resolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
  - cite[8119:8169] (citations verified byte-exact at import): "they're saami the lane that get push here is happy"
- [9] 'she' in 'if she has W' -> Nami (resolution: context_resolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [10] 'you' in discourse filler 'you know' -> the listener/addressee (non-referential discourse use) (resolution: context_resolved)
  - cite[9691:9758] (citations verified byte-exact at import): "maybe Nami will heal the sa you know like I don't know if she has W"

#### abilities_resources (3 items)
- [1] Ignite (summoner spell) belonging to the coached player (resolution: context_resolved)
  - cite[9600:9627] (citations verified byte-exact at import): 'you would ignite twoo right'
- [2] Potion (consumable item the enemy is expected to use) (resolution: context_resolved)
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"
- [3] Nami's W (Ebb and Flow), the heal/ability whose availability is uncertain (resolution: vocabulary_supported)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### events_actions (4 items)
- [1] Predicted/hypothetical: next time the enemy bot lane gives you this all-in on the freeze. (resolution: context_resolved)
  - cite[9540:9594] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze'
- [2] Predicted/hypothetical: you would ignite too in that next all-in (source surface 'twoo'). (resolution: context_resolved)
  - cite[9595:9627] (citations verified byte-exact at import): 'then you would ignite twoo right'
- [3] Predicted: the enemy (likely the sa/Samira) is probably going to use a potion. (resolution: context_resolved)
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"
- [4] Predicted/uncertain: Nami may heal the sa (Samira). (resolution: context_resolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'

#### states (1 items)
- [1] The wave/all-in context is on the freeze ('here on the freeze'). (resolution: literal_explicit)
  - cite[9569:9594] (citations verified byte-exact at import): 'all in here on the freeze'

#### conditions (2 items)
- [1] Next time they give you this all-in on the freeze, then you would ignite too. (resolution: context_resolved)
  - cite[9540:9627] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] Now is the time to ignite because they are probably going to potion and Nami may heal the sa. (resolution: context_resolved)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"

#### recommended_advice (1 items)
- [1] Ignite now rather than waiting for the next all-in on the freeze. (resolution: context_resolved)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"

#### consequences_outcomes (3 items)
- [1] If you wait until the next all-in on the freeze, your Ignite is used at that later time. (resolution: context_resolved)
  - cite[9540:9627] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] Igniting now applies it before the enemy's likely potion and Nami's possible heal of the sa. (resolution: context_resolved)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"
- [3] If Nami has W, she may heal the sa (Samira). (resolution: context_resolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### explicit_relationships (7 items)
- [1] The enemy's all-in on the freeze is the condition for 'you would ignite too'. (resolution: context_resolved; relation: CONDITION)
  - cite[9540:9627] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] The expected potion and Nami's possible heal are the reason/condition for igniting now. (resolution: context_resolved; relation: CONDITION)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"
- [3] Nami's heal affects/heals the sa (Samira). (resolution: context_resolved; relation: AFFECTS)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [4] The coached player uses Ignite. (resolution: context_resolved; relation: USES)
  - cite[9600:9627] (citations verified byte-exact at import): 'you would ignite twoo right'
- [5] Nami uses W to heal the sa (Samira), if W is available. (resolution: vocabulary_supported; relation: USES)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [6] Igniting now occurs before the enemy's expected potion and Nami's possible heal. (resolution: context_resolved; relation: BEFORE)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"
- [7] 'they' refers to the enemy bot lane. (resolution: context_resolved; relation: REFERS_TO)
  - cite[9550:9594] (citations verified byte-exact at import): 'they give you this all in here on the freeze'

#### uncertainty_unresolved (5 items)
- [1] The token 'twoo' is an ASR surface (likely 'too'): 'ignite twoo right'. (resolution: unresolved)
  - cite[9595:9627] (citations verified byte-exact at import): 'then you would ignite twoo right'
- [2] The speaker is uncertain whether Nami has W ('I don't know if she has W'). (resolution: unresolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [3] The potion is only 'probably' going to happen; it is not asserted as certain. (resolution: literal_explicit)
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"
- [4] The speaker says 'maybe Nami will heal the sa', so the heal is presented as uncertain, not certain. (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [5] 'sa' is a clipped/ASR form; the target passage itself does not spell the champion name, though elsewhere 'saami' supports Samira. (resolution: unresolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
  - cite[8119:8169] (citations verified byte-exact at import): "they're saami the lane that get push here is happy"

#### supporting_source_spans (4 items)
- [1] Entire target passage span from the condition transcript. (resolution: literal_explicit)
  - cite[9536:9758] (citations verified byte-exact at import): 'now next time they give you this all in here on the freeze then you would ignite twoo right so why not do i...'
- [2] Hypothetical freeze all-in and ignite-too support. (resolution: literal_explicit)
  - cite[9540:9627] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right'
- [3] Ignite-now-before-sustain support. (resolution: literal_explicit)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"
- [4] Uncertainty support for Nami's W. (resolution: literal_explicit)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

### TARGET p2k:case:0008

#### actors_entities (3 items)
- [1] Ambessa (the champion referred to by 'it') (resolution: context_resolved)
  - cite[75887:75948] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven"
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"
- [2] Riven (resolution: literal_explicit)
  - cite[76148:76169] (citations verified byte-exact at import): 'against Riven like if'
- [3] the coached player (addressed as 'you'/'I') (resolution: context_resolved)
  - cite[76019:76069] (citations verified byte-exact at import): "you feel like you're not able to progress the lane"
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

#### reference_bindings (6 items)
- [1] 'that's' in 'So that's' refers to Ambessa being by far better champion than Riven (resolution: context_resolved)
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"
  - cite[75887:75948] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven"
- [2] 'it' in 'ban it' / 'ban it for sure' refers to Ambessa (resolution: context_resolved)
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"
  - cite[76082:76108] (citations verified byte-exact at import): 'you should ban it for sure'
- [3] 'that' in 'on top of that' refers to Ambessa being by far better champion than Riven (resolution: context_resolved)
  - cite[76000:76108] (citations verified byte-exact at import): "if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure"
  - cite[75887:75948] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven"
- [4] 'you' refers to the coached player (resolution: context_resolved)
  - cite[76019:76069] (citations verified byte-exact at import): "you feel like you're not able to progress the lane"
- [5] 'I' in 'I might have to do that' refers to the coached player (resolution: context_resolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'
- [6] 'that' in 'do that' refers to banning Ambessa (resolution: context_resolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (1 items)
- [1] The player feels unable to progress the lane. (resolution: literal_explicit)
  - cite[76019:76069] (citations verified byte-exact at import): "you feel like you're not able to progress the lane"

#### conditions (2 items)
- [1] If the player feels unable to progress the lane, then the player should ban Ambessa for sure. (resolution: context_resolved)
  - cite[76000:76108] (citations verified byte-exact at import): "if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure"
- [2] Ambessa being by far better champion than Riven is already a good reason to ban her. (resolution: context_resolved)
  - cite[75887:75948] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven"
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"

#### recommended_advice (2 items)
- [1] You should ban Ambessa. (resolution: context_resolved)
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"
  - cite[76082:76108] (citations verified byte-exact at import): 'you should ban it for sure'
- [2] If you are not able to progress the lane, you should ban Ambessa for sure. (resolution: context_resolved)
  - cite[76000:76108] (citations verified byte-exact at import): "if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure"

#### consequences_outcomes (1 items)
- [1] Because Ambessa is by far better champion than Riven, there is already a good reason to ban her. (resolution: context_resolved)
  - cite[75887:75948] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven"
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"

#### explicit_relationships (4 items)
- [1] 'it' refers to Ambessa (resolution: context_resolved; relation: REFERS_TO)
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"
  - cite[76082:76108] (citations verified byte-exact at import): 'you should ban it for sure'
- [2] Ambessa being by far better champion than Riven is the reason there is already a good reason to ban her. (resolution: context_resolved; relation: CAUSES)
  - cite[75887:75948] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven"
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"
- [3] Feeling unable to progress the lane is the condition for the advice to ban Ambessa for sure. (resolution: context_resolved; relation: CONDITION)
  - cite[76000:76108] (citations verified byte-exact at import): "if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure"
- [4] 'that's' refers to Ambessa being by far better champion than Riven (resolution: context_resolved; relation: REFERS_TO)
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"
  - cite[75887:75948] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven"

#### uncertainty_unresolved (1 items)
- [1] The target passage ends with 'against Riven like if'; the student's if-clause is cut off before its conclusion, so that continuation is unresolved. (resolution: unresolved)
  - cite[76148:76169] (citations verified byte-exact at import): 'against Riven like if'

#### supporting_source_spans (1 items)
- [1] Source spans jointly supporting the target-passage extraction. (resolution: literal_explicit)
  - cite[75887:75948] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven"
  - cite[75953:75994] (citations verified byte-exact at import): "So that's already a good reason to ban it"
  - cite[76000:76108] (citations verified byte-exact at import): "if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for sure"
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

### TARGET p2k:case:0009

#### actors_entities (3 items)
- [1] Mel (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [2] Poppy (resolution: literal_explicit)
  - cite[56595:56629] (citations verified byte-exact at import): "obviously, she's not killing Poppy"
- [3] you (coached player) (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."

#### reference_bindings (11 items)
- [1] 'it' in 'then it's bot' -> Mel's third possible location (bot lane) (resolution: context_resolved)
  - cite[56456:56563] (citations verified byte-exact at import): "Mel only has three options, right? Yeah. So, one is mid, two is Baron, right? And then it's bot. Yes? Yeah."
- [2] 'bot' in 'She's not a bot' -> bottom lane (the third possible location) (resolution: context_resolved)
  - cite[56456:56563] (citations verified byte-exact at import): "Mel only has three options, right? Yeah. So, one is mid, two is Baron, right? And then it's bot. Yes? Yeah."
- [3] 'She' in 'She's not a bot' -> Mel (resolution: context_resolved)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [4] 'she' in 'she's not killing Poppy' -> Mel (resolution: context_resolved)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [5] 'She' in 'She's not on mid' -> Mel (resolution: context_resolved)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [6] 'mid' in 'She's not on mid' -> mid lane (resolution: context_resolved)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [7] 'you' in 'you don't play' -> coached player (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [8] 'your' in 'your ward' -> coached player's metaphorical ward (the wave) (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [9] 'waves' in 'waves being your ward' -> minion waves used as vision/ward (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [10] 'you' in 'you chase' -> coached player (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [11] 'this' in 'this could be a good flip' -> the situation of Mel showing on mid (resolution: context_resolved)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### abilities_resources (2 items)
- [1] ward (resource/vision concept; source uses waves as the ward) (resolution: literal_explicit)
  - cite[56657:56673] (citations verified byte-exact at import): 'wave equals ward'
  - cite[56701:56722] (citations verified byte-exact at import): 'waves being your ward'
- [2] wave (minion wave used as a ward/vision) (resolution: literal_explicit)
  - cite[56657:56673] (citations verified byte-exact at import): 'wave equals ward'
  - cite[56701:56722] (citations verified byte-exact at import): 'waves being your ward'

#### events_actions (4 items)
- [1] Mel is not killing Poppy (resolution: literal_explicit)
  - cite[56595:56629] (citations verified byte-exact at import): "obviously, she's not killing Poppy"
- [2] the coached player does not play around waves being their ward (resolution: context_resolved)
  - cite[56675:56723] (citations verified byte-exact at import): "But you don't play around waves being your ward."
- [3] the coached player chases (resolution: context_resolved)
  - cite[56724:56738] (citations verified byte-exact at import): 'So, you chase.'
- [4] Mel shows on mid (conditional possibility) (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### states (6 items)
- [1] Mel is not at bot lane (resolution: literal_explicit)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
- [2] Mel is not on mid (resolution: literal_explicit)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [3] wave equals ward (the mid wave reveals whether someone is there) (resolution: literal_explicit)
  - cite[56657:56673] (citations verified byte-exact at import): 'wave equals ward'
- [4] the coached player is not playing around waves as their ward (resolution: context_resolved)
  - cite[56675:56723] (citations verified byte-exact at import): "But you don't play around waves being your ward."
- [5] the coached player chases (resolution: context_resolved)
  - cite[56724:56738] (citations verified byte-exact at import): 'So, you chase.'
- [6] if Mel is showing on mid, this could be a good flip (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### conditions (1 items)
- [1] If Mel is showing on mid, this could be a good flip (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### recommended_advice (1 items)
- [1] Play around waves being your ward instead of chasing (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."

#### consequences_outcomes (4 items)
- [1] Because Mel is not killing Poppy, she is concluded to be not at bot (resolution: literal_explicit)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
- [2] Because wave equals ward, Mel is not on mid (resolution: literal_explicit)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [3] Not playing around waves as a ward results in chasing (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [4] If Mel shows on mid, a good flip is possible (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### explicit_relationships (4 items)
- [1] 'She' in 'She's not a bot' refers to Mel (resolution: context_resolved; relation: REFERS_TO)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [2] wave equals ward supports the inference that Mel is not on mid (resolution: literal_explicit; relation: CAUSES)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [3] Not playing around waves being your ward causes the coached player to chase (resolution: context_resolved; relation: CAUSES)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [4] Mel showing on mid is a condition for a good flip (resolution: literal_explicit; relation: CONDITION)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### uncertainty_unresolved (1 items)
- [1] The object of 'you chase' is not stated in the target passage (resolution: unresolved)
  - cite[56724:56738] (citations verified byte-exact at import): 'So, you chase.'

#### supporting_source_spans (2 items)
- [1] Target passage source span (resolution: literal_explicit)
  - cite[56538:56791] (citations verified byte-exact at import): "then it's bot. Yes? Yeah. She's not a bot, because well, obviously, she's not killing Poppy. She's not on m..."
- [2] Preceding context identifying Mel's three location options (resolution: literal_explicit)
  - cite[56456:56563] (citations verified byte-exact at import): "Mel only has three options, right? Yeah. So, one is mid, two is Baron, right? And then it's bot. Yes? Yeah."

### TARGET p2k:case:0010

#### actors_entities (4 items)
- [1] you (the coached player, playing Varus) (resolution: context_resolved)
  - cite[22949:22974] (citations verified byte-exact at import): "you're playing your varus"
- [2] Varus (the champion the coached player is playing) (resolution: context_resolved)
  - cite[22949:22974] (citations verified byte-exact at import): "you're playing your varus"
- [3] Blitzcrank (surface form 'a bliss crank') (resolution: context_resolved)
  - cite[23073:23113] (citations verified byte-exact at import): 'against a bliss crank now with a kalista'
- [4] Kalista (resolution: context_resolved)
  - cite[23073:23113] (citations verified byte-exact at import): 'against a bliss crank now with a kalista'

#### reference_bindings (3 items)
- [1] 'you' in the target passage refers to the coached Varus player. (resolution: context_resolved)
  - cite[22949:22974] (citations verified byte-exact at import): "you're playing your varus"
  - cite[23114:23140] (citations verified byte-exact at import): "yeah and now you don't win"
- [2] 'it' in 'you could get it' refers to 'push'. (resolution: context_resolved)
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"
- [3] 'this' in 'you're not meant to win this' refers to the matchup against Blitzcrank and Kalista. (resolution: context_resolved)
  - cite[23073:23113] (citations verified byte-exact at import): 'against a bliss crank now with a kalista'
  - cite[23279:23307] (citations verified byte-exact at import): "you're not meant to win this"

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (2 items)
- [1] Push (lane push) is the obtainable wave state named as the way to win. (resolution: literal_explicit)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
- [2] There is an angle/opportunity to get push in this situation. (resolution: literal_explicit)
  - cite[23199:23230] (citations verified byte-exact at import): "if there's an angle to get push"

#### conditions (2 items)
- [1] If you get push, you can win. (resolution: literal_explicit)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
- [2] If there is an angle to get push, then you can get push. (resolution: literal_explicit)
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"

#### recommended_advice (1 items)
- [1] Get push if there is an angle, because it is the one way you can win. (resolution: literal_explicit)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"

#### consequences_outcomes (3 items)
- [1] If you get push, you can win. (resolution: literal_explicit)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
- [2] If there is an angle to get push, you could get push and then you could win. (resolution: literal_explicit)
  - cite[23199:23271] (citations verified byte-exact at import): "if there's an angle to get push then you could get it then you could win"
- [3] Right now you do not win; you are not meant to win this matchup. (resolution: literal_explicit)
  - cite[23114:23140] (citations verified byte-exact at import): "yeah and now you don't win"
  - cite[23279:23307] (citations verified byte-exact at import): "you're not meant to win this"

#### explicit_relationships (6 items)
- [1] you (the Varus player) is the actor who could win by getting push. (resolution: context_resolved; relation: ACTOR)
  - cite[22949:22974] (citations verified byte-exact at import): "you're playing your varus"
  - cite[23114:23140] (citations verified byte-exact at import): "yeah and now you don't win"
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
- [2] getting push enables winning. (resolution: literal_explicit; relation: ENABLES)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
- [3] having an angle to get push enables getting push. (resolution: literal_explicit; relation: ENABLES)
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"
- [4] winning is presented as the result of getting push. (resolution: literal_explicit; relation: RESULT)
  - cite[23199:23271] (citations verified byte-exact at import): "if there's an angle to get push then you could get it then you could win"
- [5] 'it' refers to 'push'. (resolution: context_resolved; relation: REFERS_TO)
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"
- [6] 'this' refers to the Blitzcrank/Kalista matchup. (resolution: context_resolved; relation: REFERS_TO)
  - cite[23073:23113] (citations verified byte-exact at import): 'against a bliss crank now with a kalista'
  - cite[23279:23307] (citations verified byte-exact at import): "you're not meant to win this"

#### uncertainty_unresolved (1 items)
- [1] The token 'MH' in 'you could win MH but' is unrecoverable from the supplied transcript; it may be a filler or transcription artifact, not a meaningful game term. (resolution: unresolved)
  - cite[23268:23307] (citations verified byte-exact at import): "win MH but you're not meant to win this"

#### supporting_source_spans (3 items)
- [1] The full marked target passage. (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
- [2] Preceding context identifying 'you' as the Varus player. (resolution: context_resolved)
  - cite[22949:22974] (citations verified byte-exact at import): "you're playing your varus"
- [3] Preceding context identifying the Blitzcrank/Kalista matchup referred to by 'this'. (resolution: context_resolved)
  - cite[23073:23113] (citations verified byte-exact at import): 'against a bliss crank now with a kalista'

---

## Condition FV (model opencode-go/deepseek-v4-flash)

### TARGET p2k:case:0001

Selected candidate: candidate_2 (presentation order: candidate_3, candidate_4, candidate_5, candidate_1, candidate_2)

Verifier rationale: Candidate 2 best preserves the target passage's source-supported semantics. It captures the central causal/reasoning chain: Flash is recommended because the player is going Exhaust + Smite, which yields less fighting power due to losing Last Stand, Harvest, and Cheap Shot, linked to likely Ultimate Hunter/Eyeball and the student's confirmation. It also makes supported contextual resolutions (Smite, Dark Harvest, 'that') while honestly leaving genuinely uncertain referents unresolved ('this', full Eyeball Collection name) and correctly notes that the target ends mid-clause at 'like I would'. Other candidates either over-resolve ambiguous references, import out-of-span content, omit a key causal link, or misassign a speaker reference.

#### actors_entities (3 items)
- [1] carus (Karthus) - the champion whose rune/summoner setup is under discussion (resolution: vocabulary_supported)
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [2] the coached player (surface 'you' and student 'I' in 'yeah I do') - the Karthus player being coached (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[5158:5173] (citations verified byte-exact at import): 'right yeah I do'
- [3] the coach (surface 'I' in 'I would play with flash' / 'like I would') - the person giving the advice (resolution: context_resolved)
  - cite[4934:4957] (citations verified byte-exact at import): 'I would play with flash'

#### reference_bindings (8 items)
- [1] 'you' / 'you're' -> the coached Karthus player (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [2] 'I' in 'I would play with flash' / 'like I would' -> the coach (resolution: context_resolved)
  - cite[4934:4957] (citations verified byte-exact at import): 'I would play with flash'
- [3] 'I' in 'yeah I do' -> the coached player (student) confirming the rune setup (resolution: context_resolved)
  - cite[5158:5173] (citations verified byte-exact at import): 'right yeah I do'
- [4] 'SM' -> Smite summoner spell (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [5] 'Harvest' -> Dark Harvest rune (resolution: context_resolved)
  - cite[5061:5082] (citations verified byte-exact at import): "you're losing Harvest"
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'
- [6] 'ultimate Hunter eyeball' -> Ultimate Hunter rune and Eyeball Collection rune; 'eyeball' full name is not explicit in the target (resolution: unresolved)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [7] 'that' in 'that doesn't mean this is bad' -> the fact that the player has Ultimate Hunter and Eyeball (resolution: context_resolved)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"
- [8] 'this' in 'that doesn't mean this is bad' -> likely the current Exhaust+Smite/rune setup, but the target does not explicitly name the referent (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"

#### abilities_resources (8 items)
- [1] flash (summoner spell Flash) (resolution: literal_explicit)
  - cite[4942:4957] (citations verified byte-exact at import): 'play with flash'
- [2] exhaust (summoner spell Exhaust) (resolution: literal_explicit)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
- [3] Smite (summoner spell; surface 'SM') (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [4] Last Stand (rune) (resolution: literal_explicit)
  - cite[5036:5060] (citations verified byte-exact at import): "you're losing Last Stand"
- [5] Dark Harvest (rune; surface 'Harvest') (resolution: context_resolved)
  - cite[5061:5082] (citations verified byte-exact at import): "you're losing Harvest"
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'
- [6] Cheap Shot (rune; source lowercase 'cheap shot') (resolution: literal_explicit)
  - cite[5083:5107] (citations verified byte-exact at import): "you're losing cheap shot"
- [7] Ultimate Hunter (rune; source lowercase 'ultimate Hunter') (resolution: literal_explicit)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [8] Eyeball (rune; probably Eyeball Collection, full name not explicit) (resolution: unresolved)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'

#### events_actions (2 items)
- [1] The coached player is going Exhaust + Smite. (resolution: context_resolved)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
- [2] The coached player is losing Last Stand, Dark Harvest, and Cheap Shot in the rune setup. (resolution: context_resolved)
  - cite[5036:5107] (citations verified byte-exact at import): "you're losing Last Stand you're losing Harvest you're losing cheap shot"

#### states (3 items)
- [1] The player has less fighting power. (resolution: literal_explicit)
  - cite[4999:5027] (citations verified byte-exact at import): 'you have less fighting Power'
- [2] The player probably has 'ultimate Hunter' and 'eyeball' runes (exact rune name for 'eyeball' unresolved). (resolution: unresolved)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [3] The current setup is not bad ('that doesn't mean this is bad'). (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"

#### conditions (4 items)
- [1] If the player goes Exhaust + Smite, the player has less fighting power. (resolution: context_resolved)
  - cite[4971:5027] (citations verified byte-exact at import): "you're going exhaust SM but you have less fighting Power"
- [2] The player's lessened fighting power is because the player is losing Last Stand, Dark Harvest, and Cheap Shot. (resolution: context_resolved)
  - cite[4999:5107] (citations verified byte-exact at import): "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [3] The player is losing Cheap Shot because the player probably has Ultimate Hunter and Eyeball. (resolution: context_resolved)
  - cite[5083:5157] (citations verified byte-exact at import): "you're losing cheap shot because you probably have ultimate Hunter eyeball"
- [4] The advice to play with Flash is given because the player is going Exhaust + Smite. (resolution: context_resolved)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"

#### recommended_advice (2 items)
- [1] Play with Flash. (resolution: literal_explicit)
  - cite[4942:4957] (citations verified byte-exact at import): 'play with flash'
- [2] The coach would play with Flash instead of Exhaust + Smite when using this setup. (resolution: context_resolved)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
  - cite[4934:4957] (citations verified byte-exact at import): 'I would play with flash'

#### consequences_outcomes (4 items)
- [1] The player has less fighting power. (resolution: literal_explicit)
  - cite[4999:5027] (citations verified byte-exact at import): 'you have less fighting Power'
- [2] The player loses Last Stand, Dark Harvest, and Cheap Shot. (resolution: context_resolved)
  - cite[5036:5107] (citations verified byte-exact at import): "you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [3] Having Ultimate Hunter and Eyeball causes the loss of Cheap Shot. (resolution: context_resolved)
  - cite[5083:5157] (citations verified byte-exact at import): "you're losing cheap shot because you probably have ultimate Hunter eyeball"
- [4] That [having Ultimate Hunter/Eyeball] does not mean the setup is bad. (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"

#### explicit_relationships (10 items)
- [1] 'carus' refers to Karthus. (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [2] The coached player uses Exhaust and Smite. (resolution: context_resolved; relation: USES)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
- [3] Going Exhaust + Smite causes less fighting power. (resolution: context_resolved; relation: CAUSES)
  - cite[4971:5027] (citations verified byte-exact at import): "you're going exhaust SM but you have less fighting Power"
- [4] Losing Last Stand, Dark Harvest, and Cheap Shot causes less fighting power. (resolution: context_resolved; relation: CAUSES)
  - cite[4999:5107] (citations verified byte-exact at import): "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [5] Having Ultimate Hunter and Eyeball causes losing Cheap Shot. (resolution: context_resolved; relation: CAUSES)
  - cite[5083:5157] (citations verified byte-exact at import): "you're losing cheap shot because you probably have ultimate Hunter eyeball"
- [6] Flash is recommended conditional on the player going Exhaust + Smite. (resolution: context_resolved; relation: CONDITION)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
- [7] 'SM' refers to Smite. (resolution: context_resolved; relation: REFERS_TO)
  - cite[4971:4994] (citations verified byte-exact at import): "you're going exhaust SM"
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [8] 'Harvest' refers to Dark Harvest. (resolution: context_resolved; relation: REFERS_TO)
  - cite[5061:5082] (citations verified byte-exact at import): "you're losing Harvest"
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'
- [9] 'ultimate Hunter eyeball' refers to Ultimate Hunter and Eyeball Collection runes (eyeball's full name unresolved). (resolution: unresolved; relation: REFERS_TO)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'
- [10] 'That doesn't mean this is bad' negates the conclusion that the setup is bad. (resolution: unresolved; relation: NEGATES)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"

#### uncertainty_unresolved (3 items)
- [1] The target ends with 'like I would' before stating what the coach would do; the completion appears in the transcript immediately after the target but is outside the selected span. (resolution: unresolved)
  - cite[5174:5222] (citations verified byte-exact at import): "right that doesn't mean this is bad like I would"
- [2] 'this' in 'that doesn't mean this is bad' is unresolved: it likely refers to the current Exhaust+Smite/rune setup but the target does not explicitly name it. (resolution: unresolved)
  - cite[5180:5209] (citations verified byte-exact at import): "that doesn't mean this is bad"
- [3] 'eyeball' in 'ultimate Hunter eyeball' is interpreted as the Eyeball Collection rune, but the source does not give the full rune name. (resolution: unresolved)
  - cite[5116:5157] (citations verified byte-exact at import): 'you probably have ultimate Hunter eyeball'

#### supporting_source_spans (7 items)
- [1] for carus you are never allowed to go Smite and exhaust again (resolution: literal_explicit)
  - cite[4492:4553] (citations verified byte-exact at import): 'for carus you are never allowed to go Smite and exhaust again'
- [2] play with flash because like you're going exhaust SM (resolution: literal_explicit)
  - cite[4942:4994] (citations verified byte-exact at import): "play with flash because like you're going exhaust SM"
- [3] you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot (resolution: literal_explicit)
  - cite[4999:5107] (citations verified byte-exact at import): "you have less fighting Power because you're losing Last Stand you're losing Harvest you're losing cheap shot"
- [4] you're losing cheap shot because you probably have ultimate Hunter eyeball (resolution: literal_explicit)
  - cite[5083:5157] (citations verified byte-exact at import): "you're losing cheap shot because you probably have ultimate Hunter eyeball"
- [5] right that doesn't mean this is bad like I would (resolution: literal_explicit)
  - cite[5174:5222] (citations verified byte-exact at import): "right that doesn't mean this is bad like I would"
- [6] you should at least go dark Harvest (resolution: literal_explicit)
  - cite[4676:4711] (citations verified byte-exact at import): 'you should at least go dark Harvest'
- [7] I would play with flash (resolution: literal_explicit)
  - cite[4934:4957] (citations verified byte-exact at import): 'I would play with flash'

### TARGET p2k:case:0002

Selected candidate: candidate_4 (presentation order: candidate_4, candidate_5, candidate_3, candidate_1, candidate_2)

Verifier rationale: Candidate_4 best preserves the source-supported semantics of the target passage: it faithfully records the 11 CS statement and the doubted Aatrox-beats-Darius outcome, while explicitly marking ambiguous or garbled surface forms such as 'freeb', 'clo armor', 'T', and ASR fragments as unresolved rather than over-resolving them. It also avoids unsupported contextual inferences and captures the relevant speaker/addressee and reference bindings without adding new semantic claims.

#### actors_entities (4 items)
- [1] Aatrox (surface: 'atrox') (resolution: vocabulary_supported)
  - cite[30531:30571] (citations verified byte-exact at import): 'look at atrox atrox has 11 CS after this'
- [2] Darius (surface: 'darus') (resolution: vocabulary_supported)
  - cite[30698:30718] (citations verified byte-exact at import): 'will beat this darus'
- [3] Coach (speaker; surface: 'I') (resolution: context_resolved)
  - cite[30719:30762] (citations verified byte-exact at import): "probably not right because I don't think so"
- [4] Player being coached (addressee; surface: 'you') (resolution: context_resolved)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'

#### reference_bindings (8 items)
- [1] 'he' in 'he's giving a freeb' binds to Aatrox (resolution: context_resolved)
  - cite[30572:30603] (citations verified byte-exact at import): "if at he's giving a freeb right"
- [2] 'you' in 'do you really think' binds to the player being coached (resolution: context_resolved)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [3] 'I' in 'I don't think so' binds to the coach (resolution: context_resolved)
  - cite[30719:30762] (citations verified byte-exact at import): "probably not right because I don't think so"
- [4] 'so' in 'I don't think so' binds to the proposition 'Aatrox with cloth armor and full HP will beat this Darius' (denied by 'don't think') (resolution: context_resolved)
  - cite[30719:30762] (citations verified byte-exact at import): "probably not right because I don't think so"
- [5] 'this' in 'after this' is unresolved in the target passage (resolution: unresolved)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'
- [6] 'this' in 'because this if yeah' is garbled and unresolved (resolution: unresolved)
  - cite[30604:30648] (citations verified byte-exact at import): 'because this if yeah if atrox T is back here'
- [7] 'at' in 'if at he's giving' is an unresolved ASR fragment; possibly truncated 'atrox' (resolution: unresolved)
  - cite[30572:30603] (citations verified byte-exact at import): "if at he's giving a freeb right"
- [8] 'this' in 'this darus' binds to the current Darius in the game (resolution: context_resolved)
  - cite[30698:30718] (citations verified byte-exact at import): 'will beat this darus'

#### abilities_resources (3 items)
- [1] Aatrox's Teleport representation (surface 'T'; candidate resolution, not explicit) (resolution: unresolved)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
- [2] Aatrox's Cloth Armor item candidate (surface 'clo armor'; not exact) (resolution: unresolved)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [3] Aatrox's creep score: 11 CS (resolution: literal_explicit)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'

#### events_actions (1 items)
- [1] Aatrox is giving a 'freeb' (unresolved term; possibly a free base/back) (resolution: unresolved)
  - cite[30572:30603] (citations verified byte-exact at import): "if at he's giving a freeb right"

#### states (4 items)
- [1] Aatrox has 11 CS after this (resolution: literal_explicit)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'
- [2] Aatrox is at full HP in the considered scenario (resolution: literal_explicit)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [3] Aatrox has cloth armor in the considered scenario (surface 'clo armor') (resolution: unresolved)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [4] Aatrox's T is back here in the considered scenario (likely Teleport availability) (resolution: unresolved)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'

#### conditions (3 items)
- [1] If Aatrox is giving a 'freeb' (unresolved term) (resolution: unresolved)
  - cite[30572:30603] (citations verified byte-exact at import): "if at he's giving a freeb right"
- [2] If Aatrox's T is back here (resolution: unresolved)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
- [3] In the considered scenario where Aatrox has cloth armor and full HP (resolution: literal_explicit)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'

#### recommended_advice (0 items)

(none)

#### consequences_outcomes (1 items)
- [1] Aatrox with cloth armor and full HP probably will not beat this Darius (resolution: literal_explicit)
  - cite[30649:30762] (citations verified byte-exact at import): "do you really think atrox with clo armor full HP will beat this darus probably not right because I don't th..."

#### explicit_relationships (4 items)
- [1] Surface 'atrox' refers to Aatrox (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[30531:30571] (citations verified byte-exact at import): 'look at atrox atrox has 11 CS after this'
- [2] Surface 'darus' refers to Darius (resolution: vocabulary_supported; relation: REFERS_TO)
  - cite[30698:30718] (citations verified byte-exact at import): 'will beat this darus'
- [3] Aatrox is the actor of the proposed 'will beat this darus' scenario (resolution: literal_explicit; relation: ACTOR)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [4] Darius is the target of the proposed 'will beat this darus' scenario (resolution: literal_explicit; relation: TARGET)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'

#### uncertainty_unresolved (6 items)
- [1] 'freeb' in 'giving a freeb' is unrecoverable from the source; possibly 'free base' or 'free back' (resolution: unresolved)
  - cite[30572:30603] (citations verified byte-exact at import): "if at he's giving a freeb right"
- [2] Phrase 'if at he's' is ASR-garbled; 'at' cannot be reliably bound (resolution: unresolved)
  - cite[30572:30603] (citations verified byte-exact at import): "if at he's giving a freeb right"
- [3] 'T' in 'atrox T is back here' is an unresolved abbreviation; likely Teleport but not expanded in the source (resolution: unresolved)
  - cite[30625:30648] (citations verified byte-exact at import): 'if atrox T is back here'
- [4] 'clo armor' in 'with clo armor full HP' is uncertain; likely 'cloth armor' but not spelled out (resolution: unresolved)
  - cite[30649:30718] (citations verified byte-exact at import): 'do you really think atrox with clo armor full HP will beat this darus'
- [5] Antecedent of 'after this' is unresolved in the target passage (resolution: unresolved)
  - cite[30545:30571] (citations verified byte-exact at import): 'atrox has 11 CS after this'
- [6] 'because this if yeah if' is garbled; the 'this' and clause structure cannot be reliably restored (resolution: unresolved)
  - cite[30604:30648] (citations verified byte-exact at import): 'because this if yeah if atrox T is back here'

#### supporting_source_spans (1 items)
- [1] Target passage source span (resolution: literal_explicit)
  - cite[30531:30762] (citations verified byte-exact at import): "look at atrox atrox has 11 CS after this if at he's giving a freeb right because this if yeah if atrox T is..."

### TARGET p2k:case:0003

Selected candidate: candidate_3 (presentation order: candidate_1, candidate_3, candidate_4, candidate_2, candidate_5)

Verifier rationale: Candidate 3 best preserves the source-supported semantics of the target passage. It accurately captures the Ignite-versus-Fiora recommendation, the Flash-broken-on-Camille basis for Flash TP in most cases, and the conditional use of Ignite when heal cut is needed. It also appropriately treats the bracketed ASR gap in 'more than [ __ ] window' as unresolved and explicitly avoids importing the continuation after the passage-final 'because', unlike candidates 2 and 5.

#### actors_entities (4 items)
- [1] Camille (the champion Flash is described as broken on; the recommended setup is Flash TP in most cases) (resolution: literal_explicit)
  - cite[13406:13477] (citations verified byte-exact at import): 'I think flash is broken on Camille So I would go flash TP in most cases'
- [2] Fiora (the opponent versus whom Ignite would always be taken) (resolution: literal_explicit)
  - cite[13322:13401] (citations verified byte-exact at import): 'I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window'
- [3] The coach (speaker who says 'I' and gives the recommendations) (resolution: context_resolved)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [4] The student/Camille player (addressed as 'you' in 'when you need heal cut') (resolution: context_resolved)
  - cite[13502:13524] (citations verified byte-exact at import): 'when you need heal cut'

#### reference_bindings (9 items)
- [1] 'I' in 'I mean' refers to the coach/speaker. (resolution: context_resolved)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [2] 'I' in 'I would always go ignite' refers to the coach/speaker. (resolution: context_resolved)
  - cite[13315:13359] (citations verified byte-exact at import): 'I mean I would always go ignite versus Fiora'
- [3] 'she' in 'she heals more than [ __ ] window' refers to Fiora. (resolution: literal_explicit)
  - cite[13347:13401] (citations verified byte-exact at import): 'versus Fiora because she heals more than [\xa0__\xa0] window'
- [4] 'I' in 'I think flash is broken on Camille' refers to the coach/speaker. (resolution: context_resolved)
  - cite[13406:13440] (citations verified byte-exact at import): 'I think flash is broken on Camille'
- [5] 'I' in 'I would go flash TP in most cases' refers to the coach/speaker (modeling the recommendation for the player). (resolution: context_resolved)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [6] 'I' in 'I think ignite when you need heal cut is fine' refers to the coach/speaker. (resolution: context_resolved)
  - cite[13487:13532] (citations verified byte-exact at import): 'I think ignite when you need heal cut is fine'
- [7] 'you' in 'when you need heal cut' refers to the student/Camille player being coached. (resolution: context_resolved)
  - cite[13502:13524] (citations verified byte-exact at import): 'when you need heal cut'
- [8] 'Camille' in 'flash is broken on Camille' refers to the champion Camille. (resolution: literal_explicit)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
- [9] 'Fiora' in 'versus Fiora' refers to the champion Fiora. (resolution: literal_explicit)
  - cite[13322:13359] (citations verified byte-exact at import): 'I would always go ignite versus Fiora'

#### abilities_resources (5 items)
- [1] Ignite (summoner spell) is recommended versus Fiora. (resolution: literal_explicit)
  - cite[13322:13359] (citations verified byte-exact at import): 'I would always go ignite versus Fiora'
- [2] Flash (summoner spell) is described as broken on Camille and is part of the most-cases Flash TP setup. (resolution: literal_explicit)
  - cite[13406:13477] (citations verified byte-exact at import): 'I think flash is broken on Camille So I would go flash TP in most cases'
- [3] TP (Teleport summoner spell) is paired with Flash in most cases. (resolution: vocabulary_supported)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [4] Heal cut (healing reduction) is what justifies choosing Ignite when needed. (resolution: literal_explicit)
  - cite[13502:13524] (citations verified byte-exact at import): 'when you need heal cut'
- [5] Fiora's healing is cited as a reason for Ignite ('she heals more than [ __ ] window'). (resolution: literal_explicit)
  - cite[13368:13401] (citations verified byte-exact at import): 'she heals more than [\xa0__\xa0] window'

#### events_actions (0 items)

(none)

#### states (0 items)

(none)

#### conditions (4 items)
- [1] Condition: the matchup is versus Fiora, in which case Ignite would always be taken. (resolution: literal_explicit)
  - cite[13322:13359] (citations verified byte-exact at import): 'I would always go ignite versus Fiora'
- [2] Condition: in most cases, Flash TP is taken. (resolution: literal_explicit)
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
- [3] Condition: when you need heal cut, Ignite is fine. (resolution: literal_explicit)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'
- [4] Reason for Ignite versus Fiora: Fiora heals more than [ __ ] window. (resolution: literal_explicit)
  - cite[13322:13401] (citations verified byte-exact at import): 'I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window'

#### recommended_advice (3 items)
- [1] Always go Ignite versus Fiora. (resolution: literal_explicit)
  - cite[13322:13359] (citations verified byte-exact at import): 'I would always go ignite versus Fiora'
- [2] Go Flash TP in most cases on Camille. (resolution: literal_explicit)
  - cite[13406:13477] (citations verified byte-exact at import): 'I think flash is broken on Camille So I would go flash TP in most cases'
- [3] If/when you need heal cut, Ignite is fine. (resolution: literal_explicit)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'

#### consequences_outcomes (3 items)
- [1] Because Fiora heals more than [ __ ] window, Ignite is the chosen option versus Fiora. (resolution: literal_explicit)
  - cite[13322:13401] (citations verified byte-exact at import): 'I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window'
- [2] Because Flash is broken on Camille, Flash TP is the default loadout in most cases. (resolution: literal_explicit)
  - cite[13406:13477] (citations verified byte-exact at import): 'I think flash is broken on Camille So I would go flash TP in most cases'
- [3] When heal cut is needed, Ignite is an acceptable summoner-spell choice. (resolution: literal_explicit)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'

#### explicit_relationships (5 items)
- [1] Fiora's healing (more than [ __ ] window) causes/justifies choosing Ignite versus Fiora. (resolution: literal_explicit; relation: CAUSES)
  - cite[13322:13401] (citations verified byte-exact at import): 'I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window'
- [2] The matchup versus Fiora is the condition for always going Ignite. (resolution: literal_explicit; relation: CONDITION)
  - cite[13322:13359] (citations verified byte-exact at import): 'I would always go ignite versus Fiora'
- [3] Flash being broken on Camille results in Flash TP in most cases. (resolution: literal_explicit; relation: RESULT)
  - cite[13406:13477] (citations verified byte-exact at import): 'I think flash is broken on Camille So I would go flash TP in most cases'
- [4] The need for heal cut is the condition for Ignite being fine. (resolution: literal_explicit; relation: CONDITION)
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'
- [5] Camille uses Flash and TP as the recommended summoner spell setup. (resolution: context_resolved; relation: USES)
  - cite[13414:13440] (citations verified byte-exact at import): 'flash is broken on Camille'
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'

#### uncertainty_unresolved (3 items)
- [1] The bracketed '[ __ ]' in 'more than [ __ ] window' is an unrecoverable ASR gap; its intended word cannot be determined from the source. (resolution: unresolved)
  - cite[13368:13401] (citations verified byte-exact at import): 'she heals more than [\xa0__\xa0] window'
- [2] The word 'window' after the gap is ambiguous/corrupted; no resolution is supported. (resolution: unresolved)
  - cite[13368:13401] (citations verified byte-exact at import): 'she heals more than [\xa0__\xa0] window'
- [3] The target passage ends with 'because', leaving the rationale incomplete within the passage; later text is not imported. (resolution: unresolved)
  - cite[13495:13540] (citations verified byte-exact at import): 'ignite when you need heal cut is fine because'

#### supporting_source_spans (1 items)
- [1] Minimal quotes supporting the extracted target-passage content. (resolution: literal_explicit)
  - cite[13322:13401] (citations verified byte-exact at import): 'I would always go ignite versus Fiora because she heals more than [\xa0__\xa0] window'
  - cite[13406:13440] (citations verified byte-exact at import): 'I think flash is broken on Camille'
  - cite[13444:13477] (citations verified byte-exact at import): 'I would go flash TP in most cases'
  - cite[13495:13532] (citations verified byte-exact at import): 'ignite when you need heal cut is fine'

### TARGET p2k:case:0004

Selected candidate: candidate_3 (presentation order: candidate_4, candidate_5, candidate_3, candidate_2, candidate_1)

Verifier rationale: candidate_3 best preserves the target's source-supported semantics. It captures both conditional 'go' triggers while explicitly marking the initial 'move' subject as unresolved IIA and the object of 'Brier loses to' as omitted. It also preserves the invade rationale ('could make it winning') and the hoped avoidance of Talia 2v1'ing you without over-resolving into unsupported advice such as 'you should move' or additional fight-causation claims.

#### actors_entities (5 items)
- [1] Brier (Briar) (resolution: context_resolved)
  - cite[54348:54373] (citations verified byte-exact at import): "Brier doesn't win one one"
- [2] Talia (Taliyah) (resolution: context_resolved)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
- [3] you (student/Aatrox player) (resolution: context_resolved)
  - cite[54321:54425] (citations verified byte-exact at import): "move then you should go if Brier doesn't win one one then you should go because you still want to invade"
- [4] I (coach/speaker) (resolution: context_resolved)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
- [5] IIA (unresolved subject of the initial 'move' clause) (resolution: unresolved)
  - cite[54313:54325] (citations verified byte-exact at import): 'IIA can move'
  - cite[54321:54344] (citations verified byte-exact at import): 'move then you should go'

#### reference_bindings (4 items)
- [1] 'you' -> the student/Aatrox player being coached (resolution: context_resolved)
  - cite[54321:54425] (citations verified byte-exact at import): "move then you should go if Brier doesn't win one one then you should go because you still want to invade"
- [2] 'I' -> the coach/speaker (resolution: context_resolved)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'
- [3] 'it' in 'make it winning' -> the invade/play under discussion (resolution: context_resolved)
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'
- [4] 'one one' -> a 1v1 / one-versus-one fight (resolution: context_resolved)
  - cite[54348:54373] (citations verified byte-exact at import): "Brier doesn't win one one"

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (2 items)
- [1] Brier doesn't win one one (stated as a conditional premise) (resolution: literal_explicit)
  - cite[54348:54373] (citations verified byte-exact at import): "Brier doesn't win one one"
- [2] Brier loses to an unstated opponent (stated as a premise) (resolution: unresolved)
  - cite[54439:54453] (citations verified byte-exact at import): 'Brier loses to'

#### conditions (3 items)
- [1] If Brier doesn't win one one, then you should go. (resolution: literal_explicit)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
- [2] If [IIA] can move, then you should go; the subject 'IIA' is unresolved. (resolution: unresolved)
  - cite[54313:54344] (citations verified byte-exact at import): 'IIA can move then you should go'
- [3] Brier losing to someone does not imply that you shouldn't invade. (resolution: unresolved)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"

#### recommended_advice (4 items)
- [1] You should go if Brier doesn't win one one. (resolution: literal_explicit)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
- [2] You should go if [IIA] can move (subject unresolved). (resolution: unresolved)
  - cite[54313:54344] (citations verified byte-exact at import): 'IIA can move then you should go'
- [3] You should still invade even if Brier loses to someone, because you could make it winning. (resolution: unresolved)
  - cite[54426:54517] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade cuz you could make it winning"
- [4] You should go because you still want to invade. (resolution: literal_explicit)
  - cite[54374:54425] (citations verified byte-exact at import): 'then you should go because you still want to invade'

#### consequences_outcomes (2 items)
- [1] You could make the invade winning. (resolution: context_resolved)
  - cite[54488:54517] (citations verified byte-exact at import): 'cuz you could make it winning'
- [2] Potential bad outcome the coach hopes to avoid: Talia 2v1s you. (resolution: context_resolved)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'

#### explicit_relationships (6 items)
- [1] Brier doesn't win one one -> you should go (resolution: literal_explicit; relation: CONDITION)
  - cite[54345:54392] (citations verified byte-exact at import): "if Brier doesn't win one one then you should go"
- [2] [IIA] can move -> you should go (subject unresolved) (resolution: unresolved; relation: CONDITION)
  - cite[54313:54344] (citations verified byte-exact at import): 'IIA can move then you should go'
- [3] you still want to invade -> you should go (resolution: literal_explicit; relation: CAUSES)
  - cite[54374:54425] (citations verified byte-exact at import): 'then you should go because you still want to invade'
- [4] you could make it winning -> you should invade (resolution: context_resolved; relation: CAUSES)
  - cite[54426:54517] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade cuz you could make it winning"
- [5] Brier losing does not mean you shouldn't invade (resolution: unresolved; relation: NEGATES)
  - cite[54426:54487] (citations verified byte-exact at import): "just because Brier loses to doesn't mean you shouldn't invade"
- [6] Talia would target you in a 2v1; the coach hopes this does not happen (resolution: context_resolved; relation: TARGET)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'

#### uncertainty_unresolved (5 items)
- [1] The subject of the initial 'move' clause is outside the target and is transcribed as 'IIA'; it could not be resolved. (resolution: unresolved)
  - cite[54313:54325] (citations verified byte-exact at import): 'IIA can move'
  - cite[54321:54344] (citations verified byte-exact at import): 'move then you should go'
- [2] The object of 'Brier loses to' is missing; the target does not say who Brier loses to. (resolution: unresolved)
  - cite[54439:54453] (citations verified byte-exact at import): 'Brier loses to'
- [3] 'one one' likely represents '1v1' but is not spelled out literally. (resolution: unresolved)
  - cite[54348:54373] (citations verified byte-exact at import): "Brier doesn't win one one"
- [4] The excerpt ends mid-sentence with 'but'; the continuation is outside the target passage. (resolution: unresolved)
  - cite[54321:54566] (citations verified byte-exact at import): "move then you should go if Brier doesn't win one one then you should go because you still want to invade ju..."
- [5] 'I hope Talia does not 2v one you' is expressed as a hope/uncertainty, not a confirmed outcome. (resolution: literal_explicit)
  - cite[54524:54556] (citations verified byte-exact at import): 'I hope Talia does not 2v one you'

#### supporting_source_spans (2 items)
- [1] Full target passage from the condition transcript (resolution: literal_explicit)
  - cite[54321:54566] (citations verified byte-exact at import): "move then you should go if Brier doesn't win one one then you should go because you still want to invade ju..."
- [2] Preceding context for the unresolved 'move' subject (resolution: unresolved)
  - cite[54313:54325] (citations verified byte-exact at import): 'IIA can move'

### TARGET p2k:case:0005

Selected candidate: candidate_1 (presentation order: candidate_1, candidate_3, candidate_2, candidate_4, candidate_5)

Verifier rationale: Candidate 1 best preserves the source-supported semantics of the target passage without introducing unsupported League-specific inferences. It correctly grounds Q, E, HP loss, wave loss, and the post-E 'no spell' state in the source/context, binds 'she'/'her' to Syndra and 'you' to the coached player, preserves the run-at-Syndra-after-Q advice, and honestly marks unresolved items such as the 'we' filler and unresolved player-champion identity.

#### actors_entities (3 items)
- [1] Syndra (the referent of 'she'/'her' in the target passage) (resolution: context_resolved)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] the player/coachee (the referent of 'you' in the target passage) (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [3] the whole wave (resolution: literal_explicit)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'

#### reference_bindings (8 items)
- [1] 'she' in 'she uses Q' -> Syndra (resolution: context_resolved)
  - cite[55445:55464] (citations verified byte-exact at import): 'now that she uses Q'
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [2] 'her' in 'run at her' -> Syndra (resolution: context_resolved)
  - cite[55423:55444] (citations verified byte-exact at import): 'you should run at her'
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [3] 'you' in 'you should run at her' -> the player/coachee (resolution: context_resolved)
  - cite[55423:55444] (citations verified byte-exact at import): 'you should run at her'
- [4] 'she' in 'she loses the whole wave' -> Syndra (resolution: context_resolved)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [5] 'she' in 'if she uses e' -> Syndra (resolution: context_resolved)
  - cite[51869:51882] (citations verified byte-exact at import): 'if she uses e'
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [6] 'she' in 'does she get to farm' -> Syndra (resolution: context_resolved)
  - cite[55344:55370] (citations verified byte-exact at import): 'does she get to farm no no'
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [7] 'she' in final 'she has no' -> Syndra (resolution: context_resolved)
  - cite[55526:55545] (citations verified byte-exact at import): 'but then she has no'
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [8] 'we' in 'right we then you lose' -> unresolved; appears to be a discourse filler (resolution: unresolved)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'

#### abilities_resources (4 items)
- [1] Q (the ability Syndra uses) (resolution: context_resolved)
  - cite[55445:55464] (citations verified byte-exact at import): 'now that she uses Q'
- [2] e (the ability Syndra uses) (resolution: context_resolved)
  - cite[51869:51882] (citations verified byte-exact at import): 'if she uses e'
- [3] spell (the generic ability Syndra no longer has after using E) (resolution: context_resolved)
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'
- [4] HP/health (about 100-150 HP the player loses) (resolution: literal_explicit)
  - cite[55515:55525] (citations verified byte-exact at import): '100 HP 150'

#### events_actions (2 items)
- [1] Syndra uses Q. (resolution: context_resolved)
  - cite[55445:55464] (citations verified byte-exact at import): 'now that she uses Q'
- [2] Syndra uses E. (resolution: context_resolved)
  - cite[51869:51882] (citations verified byte-exact at import): 'if she uses e'

#### states (4 items)
- [1] Syndra has no spell available after using E. (resolution: context_resolved)
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'
- [2] Syndra loses the whole wave. (resolution: context_resolved)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'
- [3] Syndra does not get to farm. (resolution: context_resolved)
  - cite[55344:55370] (citations verified byte-exact at import): 'does she get to farm no no'
- [4] The player loses about 100-150 HP if Syndra uses E. (resolution: context_resolved)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'

#### conditions (3 items)
- [1] The timing/condition for running at Syndra is now that she uses Q. (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [2] If Syndra uses E, the player loses about 100-150 HP. (resolution: context_resolved)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'
- [3] If Syndra uses E, she has no spell afterward. (resolution: context_resolved)
  - cite[55465:55551] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no spell'

#### recommended_advice (1 items)
- [1] You should run at Syndra now that she uses Q. (resolution: context_resolved)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"

#### consequences_outcomes (4 items)
- [1] Syndra does not get to farm. (resolution: context_resolved)
  - cite[55344:55370] (citations verified byte-exact at import): 'does she get to farm no no'
- [2] Syndra loses the whole wave. (resolution: context_resolved)
  - cite[55371:55395] (citations verified byte-exact at import): 'she loses the whole wave'
- [3] If Syndra uses E, the player loses about 100-150 HP. (resolution: context_resolved)
  - cite[55501:55525] (citations verified byte-exact at import): 'you lose like 100 HP 150'
- [4] After Syndra uses E, she has no spell. (resolution: context_resolved)
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'

#### explicit_relationships (10 items)
- [1] Syndra uses Q. (resolution: context_resolved; relation: USES)
  - cite[55445:55464] (citations verified byte-exact at import): 'now that she uses Q'
- [2] Syndra uses E. (resolution: context_resolved; relation: USES)
  - cite[51869:51882] (citations verified byte-exact at import): 'if she uses e'
- [3] Syndra using Q is the condition for the player to run at her. (resolution: context_resolved; relation: CONDITION)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [4] If Syndra uses E, the player loses about 100-150 HP. (resolution: context_resolved; relation: CONDITION)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'
- [5] Syndra's use of E results in the player losing about 100-150 HP. (resolution: context_resolved; relation: RESULT)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'
- [6] Syndra does not get to farm, resulting in her losing the whole wave. (resolution: context_resolved; relation: RESULT)
  - cite[55344:55395] (citations verified byte-exact at import): 'does she get to farm no no she loses the whole wave'
- [7] After Syndra uses E, she has no spell. (resolution: context_resolved; relation: AFTER)
  - cite[55465:55551] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150 but then she has no spell'
- [8] Syndra having used Q enables the player to run at her. (resolution: context_resolved; relation: ENABLES)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"
- [9] The player is the actor who should run at Syndra. (resolution: context_resolved; relation: ACTOR)
  - cite[55423:55444] (citations verified byte-exact at import): 'you should run at her'
- [10] Syndra is the target of the player's run-at action. (resolution: context_resolved; relation: TARGET)
  - cite[55423:55444] (citations verified byte-exact at import): 'you should run at her'

#### uncertainty_unresolved (2 items)
- [1] The 'we' in 'right we then you lose' has no clear referent; it appears to be a discourse filler. (resolution: unresolved)
  - cite[55465:55525] (citations verified byte-exact at import): 'because if she uses e right we then you lose like 100 HP 150'
- [2] The player's champion identity is not explicitly named inside the target passage. The surrounding transcript has 'way' in 'I'm way versus syndra', but the champion name remains unresolved. (resolution: unresolved)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"

#### supporting_source_spans (4 items)
- [1] The target passage itself. (resolution: literal_explicit)
  - cite[55339:55545] (citations verified byte-exact at import): "then does she get to farm no no she loses the whole wave then yes yes so that's why you should run at her n..."
- [2] Context establishing that the enemy mid is Syndra. (resolution: context_resolved)
  - cite[47585:47606] (citations verified byte-exact at import): "I'm way versus syndra"
- [3] Immediate continuation supplying 'spell' after the target's final 'has no'. (resolution: context_resolved)
  - cite[55526:55551] (citations verified byte-exact at import): 'but then she has no spell'
- [4] The 'run at her after Q' advice span. (resolution: literal_explicit)
  - cite[55409:55464] (citations verified byte-exact at import): "so that's why you should run at her now that she uses Q"

### TARGET p2k:case:0006

Selected candidate: candidate_2 (presentation order: candidate_5, candidate_3, candidate_1, candidate_2, candidate_4)

Verifier rationale: Candidate 2 best preserves the source-supported semantics of the target passage. It correctly identifies Mel's Q/queue as the ability whose point is to proc Scorch and Comet, treats the unwanted habit as the habit of only getting a little Q / stepping back, and preserves the better alternative of staying to land two more ticks. It also handles the truncated 'because if this was' conditional honestly, marking the referent as unresolved rather than overcommitting to an unsupported resolution. It avoids the extra unsupported League-specific additions and misbindings present in other candidates.

#### actors_entities (3 items)
- [1] The coached player ('you'), who is playing Mel (resolution: context_resolved)
  - cite[4272:4307] (citations verified byte-exact at import): "it's a habit you don't want to have"
  - cite[4179:4200] (citations verified byte-exact at import): 'of course you are Mel'
- [2] Mel, the champion being played by 'you' (resolution: context_resolved)
  - cite[4179:4200] (citations verified byte-exact at import): 'of course you are Mel'
- [3] Other champions (champions other than Mel that the player might pick up) (resolution: literal_explicit)
  - cite[4382:4414] (citations verified byte-exact at import): 'when you pick up other champions'

#### reference_bindings (5 items)
- [1] 'you' in the target passage binds to the coached Mel player (resolution: context_resolved)
  - cite[4272:4307] (citations verified byte-exact at import): "it's a habit you don't want to have"
  - cite[4179:4200] (citations verified byte-exact at import): 'of course you are Mel'
- [2] 'it' in 'it's a habit' binds to the habit of only getting a little bit of your queue / stepping back after farming (resolution: context_resolved)
  - cite[4109:4148] (citations verified byte-exact at import): 'you only get a little bit of your queue'
  - cite[4593:4635] (citations verified byte-exact at import): 'you do this random step back after farming'
- [3] 'here' in 'you could just be here' binds to the desired position/freeze frame (resolution: context_resolved)
  - cite[4316:4368] (citations verified byte-exact at import): 'you could just be here and land maybe two more ticks'
- [4] 'this' in 'this is a extremely bad freeze frame' binds to the current freeze frame/position (resolution: context_resolved)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'
- [5] 'this' in 'if this was' is not fully resolvable from the isolated target; adjacent source completes it as 'Synindra Q or something else' (resolution: unresolved)
  - cite[4452:4471] (citations verified byte-exact at import): 'because if this was'
  - cite[4460:4526] (citations verified byte-exact at import): 'if this was Synindra Q or something else maybe you miss everything'

#### abilities_resources (3 items)
- [1] Mel's Q ability ('your queue'), whose main point is to proc Scorch and Comet (resolution: context_resolved)
  - cite[4205:4261] (citations verified byte-exact at import): 'the main point of your queue is to proc scorch and comet'
  - cite[4179:4200] (citations verified byte-exact at import): 'of course you are Mel'
- [2] Scorch and Comet, procced by Mel's Q (resolution: literal_explicit)
  - cite[4240:4261] (citations verified byte-exact at import): 'proc scorch and comet'
- [3] Damage ticks ('two more ticks') that could be landed by staying in position (resolution: context_resolved)
  - cite[4343:4368] (citations verified byte-exact at import): 'land maybe two more ticks'

#### events_actions (1 items)
- [1] The stated purpose/action of the Q is to proc Scorch and Comet (resolution: literal_explicit)
  - cite[4234:4261] (citations verified byte-exact at import): 'is to proc scorch and comet'

#### states (3 items)
- [1] The player is at a position ('here') from which two more ticks could be landed (resolution: context_resolved)
  - cite[4316:4368] (citations verified byte-exact at import): 'you could just be here and land maybe two more ticks'
- [2] The current freeze frame is extremely bad when picking up other champions (resolution: literal_explicit)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'
- [3] The player has the habit of only getting a little bit of Q / stepping back after farming (resolution: context_resolved)
  - cite[4109:4148] (citations verified byte-exact at import): 'you only get a little bit of your queue'
  - cite[4593:4635] (citations verified byte-exact at import): 'you do this random step back after farming'

#### conditions (2 items)
- [1] When the player picks up other champions, the freeze frame is extremely bad (resolution: literal_explicit)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'
- [2] If this was another champion's Q (e.g. Synindra Q), missing everything is the stated possibility (resolution: context_resolved)
  - cite[4460:4526] (citations verified byte-exact at import): 'if this was Synindra Q or something else maybe you miss everything'

#### recommended_advice (2 items)
- [1] Do not want/have the habit; instead be here and land maybe two more ticks (resolution: context_resolved)
  - cite[4272:4368] (citations verified byte-exact at import): "it's a habit you don't want to have because you could just be here and land maybe two more ticks"
- [2] When playing other champions, recognize this freeze frame as extremely bad (resolution: literal_explicit)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'

#### consequences_outcomes (3 items)
- [1] Keeping the habit means missing maybe two more ticks that could be landed from here (resolution: context_resolved)
  - cite[4272:4368] (citations verified byte-exact at import): "it's a habit you don't want to have because you could just be here and land maybe two more ticks"
- [2] If this was another champion's Q, the player might miss everything (resolution: context_resolved)
  - cite[4460:4526] (citations verified byte-exact at import): 'if this was Synindra Q or something else maybe you miss everything'
- [3] Picking up other champions makes this freeze frame extremely bad (resolution: literal_explicit)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'

#### explicit_relationships (4 items)
- [1] Mel's Q procs Scorch and Comet (resolution: context_resolved; relation: CAUSES)
  - cite[4205:4261] (citations verified byte-exact at import): 'the main point of your queue is to proc scorch and comet'
  - cite[4179:4200] (citations verified byte-exact at import): 'of course you are Mel'
- [2] Picking up other champions is a condition under which the freeze frame is extremely bad (resolution: literal_explicit; relation: CONDITION)
  - cite[4382:4451] (citations verified byte-exact at import): 'when you pick up other champions this is a extremely bad freeze frame'
- [3] Missing everything is the possible result if this was another champion's Q (resolution: context_resolved; relation: RESULT)
  - cite[4460:4526] (citations verified byte-exact at import): 'if this was Synindra Q or something else maybe you miss everything'
- [4] 'it' in 'it's a habit' refers to the habit of stepping back / only getting a little Q (resolution: context_resolved; relation: REFERS_TO)
  - cite[4109:4148] (citations verified byte-exact at import): 'you only get a little bit of your queue'
  - cite[4593:4635] (citations verified byte-exact at import): 'you do this random step back after farming'

#### uncertainty_unresolved (3 items)
- [1] The connective 'and on' in 'right and on when you pick up other champions' is unclear and not confidently resolvable (resolution: unresolved)
  - cite[4369:4414] (citations verified byte-exact at import): 'right and on when you pick up other champions'
- [2] The target passage truncates at 'because if this was'; the adjacent source continues with 'Synindra Q or something else maybe you miss everything,' but the conditional is not fully contained in the marked target (resolution: unresolved)
  - cite[4452:4471] (citations verified byte-exact at import): 'because if this was'
  - cite[4460:4526] (citations verified byte-exact at import): 'if this was Synindra Q or something else maybe you miss everything'
- [3] The exact referent of 'this' in 'if this was' remains only inferable from adjacent source, not explicit in the isolated target (resolution: unresolved)
  - cite[4452:4471] (citations verified byte-exact at import): 'because if this was'

#### supporting_source_spans (1 items)
- [1] Minimal source spans supporting the target passage extraction (resolution: context_resolved)
  - cite[4234:4471] (citations verified byte-exact at import): "is to proc scorch and comet right but it's a habit you don't want to have because you could just be here an..."
  - cite[4179:4200] (citations verified byte-exact at import): 'of course you are Mel'
  - cite[4205:4261] (citations verified byte-exact at import): 'the main point of your queue is to proc scorch and comet'
  - cite[4109:4148] (citations verified byte-exact at import): 'you only get a little bit of your queue'
  - cite[4593:4635] (citations verified byte-exact at import): 'you do this random step back after farming'
  - cite[4460:4526] (citations verified byte-exact at import): 'if this was Synindra Q or something else maybe you miss everything'

### TARGET p2k:case:0007

Selected candidate: candidate_1 (presentation order: candidate_4, candidate_3, candidate_2, candidate_5, candidate_1)

Verifier rationale: Candidate 1 best preserves the target passage semantics: it keeps the next-freeze conditional ignite, the advice to act now, the probable potion use, the possible Nami heal of 'sa', and the unresolved W/twoo/sa uncertainties. It avoids importing later-context claims and does not overstate uncertain events as definite.

#### actors_entities (5 items)
- [1] coached player ('you'), the Lucian ADC in this coaching session (resolution: context_resolved)
  - cite[9600:9627] (citations verified byte-exact at import): 'you would ignite twoo right'
  - cite[8100:8118] (citations verified byte-exact at import): "you're Lucian Milo"
- [2] Nami, the enemy support who may heal the sa (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [3] sa, the enemy ADC referred to by the ASR span 'sa' (unresolved exactly) (resolution: unresolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [4] they, the enemy bot-lane pair (Nami and the sa) (resolution: context_resolved)
  - cite[9550:9580] (citations verified byte-exact at import): 'they give you this all in here'
- [5] coach, the speaker of 'I don't know' (resolution: context_resolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### reference_bindings (7 items)
- [1] you → coached player (Lucian ADC) (resolution: context_resolved)
  - cite[9600:9627] (citations verified byte-exact at import): 'you would ignite twoo right'
- [2] they → enemy bot-lane pair (Nami and the sa) (resolution: context_resolved)
  - cite[9550:9580] (citations verified byte-exact at import): 'they give you this all in here'
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"
- [3] this all in → the all-in trade on the freeze under discussion (resolution: context_resolved)
  - cite[9564:9594] (citations verified byte-exact at import): 'this all in here on the freeze'
- [4] it → the advised ignite/action to do now (resolution: context_resolved)
  - cite[9631:9648] (citations verified byte-exact at import): 'why not do it now'
- [5] I → the coach (resolution: context_resolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [6] she → Nami (resolution: context_resolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [7] the sa → enemy ADC; exact identity unresolved (resolution: unresolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'

#### abilities_resources (3 items)
- [1] Ignite (summoner spell) (resolution: literal_explicit)
  - cite[9600:9627] (citations verified byte-exact at import): 'you would ignite twoo right'
- [2] Nami's W (Ebb and Flow), the heal ability whose availability is uncertain (resolution: vocabulary_supported)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [3] Potion (enemy consumable/sustain resource) (resolution: literal_explicit)
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"

#### events_actions (5 items)
- [1] They give you this all-in on the freeze (hypothetical next time) (resolution: literal_explicit)
  - cite[9540:9594] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze'
- [2] You would ignite too (ASR 'twoo') on that next all-in (resolution: literal_explicit)
  - cite[9595:9627] (citations verified byte-exact at import): 'then you would ignite twoo right'
- [3] Do it now — perform the ignite/action now instead of next time (resolution: literal_explicit)
  - cite[9631:9648] (citations verified byte-exact at import): 'why not do it now'
- [4] They will probably use potion (resolution: literal_explicit)
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"
- [5] Nami may heal the sa (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'

#### states (4 items)
- [1] The wave is in a freeze; the all-in is offered on the freeze (resolution: literal_explicit)
  - cite[9564:9594] (citations verified byte-exact at import): 'this all in here on the freeze'
- [2] Enemy potion use is probable, not certain (resolution: literal_explicit)
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"
- [3] Nami's heal of the sa is a possibility ('maybe') (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [4] Nami's W availability is unknown to the coach (resolution: literal_explicit)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### conditions (3 items)
- [1] If/next time they give you this all-in on the freeze, then you would ignite (resolution: literal_explicit)
  - cite[9540:9627] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] When they are going to probably potion and maybe Nami will heal the sa is the timing for doing it now (resolution: literal_explicit)
  - cite[9649:9718] (citations verified byte-exact at import): "when they're going to probably potion and maybe Nami will heal the sa"
- [3] The embedded condition is 'if she has W'; whether it holds is unresolved (resolution: unresolved)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### recommended_advice (2 items)
- [1] Ignite too on the next freeze all-in ('twoo' ASR for too) (resolution: literal_explicit)
  - cite[9540:9627] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] Do it now rather than waiting for next time, because they will probably potion and Nami may heal the sa (resolution: literal_explicit)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"

#### consequences_outcomes (2 items)
- [1] If the ignite/action is delayed to next time, the enemies will likely potion and Nami may heal the sa, reducing the value of waiting (resolution: context_resolved)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"
- [2] Nami may heal the sa if she has W; W availability is unknown (resolution: unresolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

#### explicit_relationships (7 items)
- [1] The freeze all-in ('next time they give you this all in here on the freeze') is the condition for the proposed ignite. (resolution: literal_explicit; relation: CONDITION)
  - cite[9540:9627] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] Nami is the actor who will/may heal the sa. (resolution: literal_explicit; relation: ACTOR)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [3] The sa is the target of Nami's heal. (resolution: unresolved; relation: TARGET)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [4] Nami's W is the ability that could heal the sa, if it is up. (resolution: vocabulary_supported; relation: USES)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [5] 'she' refers to Nami. (resolution: context_resolved; relation: REFERS_TO)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [6] The 'next time' all-in is after the current moment; the coach asks why not ignite now instead. (resolution: context_resolved; relation: AFTER)
  - cite[9540:9648] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right so why not do it now'
- [7] A likely result of waiting is that they potion and Nami heals the sa, so doing it now is better. (resolution: context_resolved; relation: RESULT)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"

#### uncertainty_unresolved (5 items)
- [1] 'twoo' is an ASR span for 'too' in 'ignite twoo right' (resolution: unresolved)
  - cite[9617:9621] (citations verified byte-exact at import): 'twoo'
- [2] 'sa' is an unresolved ASR name for the enemy ADC; likely Samira but not supplied exactly in vocabulary (resolution: unresolved)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'
- [3] Coach uncertainty: whether Nami has W up (resolution: literal_explicit)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"
- [4] Enemy potion is probable, not certain ('probably') (resolution: literal_explicit)
  - cite[9654:9686] (citations verified byte-exact at import): "they're going to probably potion"
- [5] Nami's heal is possible, not certain ('maybe') (resolution: literal_explicit)
  - cite[9691:9718] (citations verified byte-exact at import): 'maybe Nami will heal the sa'

#### supporting_source_spans (3 items)
- [1] Condition + proposed action span (resolution: literal_explicit)
  - cite[9540:9627] (citations verified byte-exact at import): 'next time they give you this all in here on the freeze then you would ignite twoo right'
- [2] Advice now + timing/sustain reason span (resolution: literal_explicit)
  - cite[9631:9718] (citations verified byte-exact at import): "why not do it now when they're going to probably potion and maybe Nami will heal the sa"
- [3] Uncertainty span about Nami's W (resolution: literal_explicit)
  - cite[9733:9758] (citations verified byte-exact at import): "I don't know if she has W"

### TARGET p2k:case:0008

Selected candidate: candidate_3 (presentation order: candidate_3, candidate_5, candidate_4, candidate_2, candidate_1)

Verifier rationale: Candidate 3 best preserves source-supported semantics: it correctly resolves 'it' in the ban advice to Ambessa from the preceding comparison, preserves the conditional ban advice without treating the conditional feeling as an asserted state, accurately records the student's 'might have to do that' response, and handles the target's truncated 'because against Riven like if' by citing the full-transcript continuation rather than guessing. Its references, relationship bindings, and uncertainty markers are precise and grounded in the source.

#### actors_entities (3 items)
- [1] Ambessa (the champion the target passage discusses banning; the 'it' in 'ban it' resolves to her from the preceding source clause 'Ambessa is by far better champion than Riven') (resolution: context_resolved)
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [2] Riven (the champion explicitly named in 'than Riven' and 'against Riven' as the comparison/opponent) (resolution: literal_explicit)
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'
- [3] The player being coached (the 'you' addressed by the coach and the 'I' in 'I might have to do that') (resolution: context_resolved)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

#### reference_bindings (8 items)
- [1] 'it' in 'a good reason to ban it' refers to Ambessa. (resolution: context_resolved)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
- [2] 'it' in 'you should ban it for sure' refers to Ambessa. (resolution: context_resolved)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
- [3] 'that' in 'So that's already a good reason to ban it' refers to the proposition that Ambessa is by far better champion than Riven. (resolution: context_resolved)
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [4] 'that' in 'on top of that' refers to the already-good reason, i.e. Ambessa being by far better champion than Riven. (resolution: context_resolved)
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
- [5] 'you' in 'you feel' and 'you should ban it' refers to the player being coached. (resolution: context_resolved)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
- [6] 'the lane' in 'progress the lane' refers to the top lane, the player's lane in this game. (resolution: context_resolved)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
  - cite[3756:3768] (citations verified byte-exact at import): 'Top >> lane.'
- [7] 'I' in 'I might have to do that' refers to the player being coached. (resolution: context_resolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
- [8] 'that' in 'do that' refers to the advised action of banning Ambessa. (resolution: context_resolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."

#### abilities_resources (0 items)

(none)

#### events_actions (0 items)

(none)

#### states (2 items)
- [1] Ambessa is by far better champion than Riven. (resolution: literal_explicit)
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
- [2] There is already a good reason to ban Ambessa. (resolution: literal_explicit)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."

#### conditions (3 items)
- [1] If the player feels, on top of the champion-strength reason, that he is not able to progress the lane, then he should ban Ambessa for sure. (resolution: literal_explicit)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
- [2] Given that Ambessa is by far better champion than Riven, that is already a good reason to ban Ambessa. (resolution: context_resolved)
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [3] Against Riven, if they all-in the player, the player can usually outplay most Rivens (the full-transcript continuation of the target's trailing 'because against Riven like if'). (resolution: context_resolved)
  - cite[76148:76227] (citations verified byte-exact at import): 'against Riven like if they all in me and like I can usually outplay most Rivens'

#### recommended_advice (2 items)
- [1] Ambessa being by far better champion than Riven is already a good reason to ban Ambessa. (resolution: literal_explicit)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
- [2] If you also feel unable to progress the lane, you should ban Ambessa for sure. (resolution: literal_explicit)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."

#### consequences_outcomes (1 items)
- [1] The player responds that he might have to ban Ambessa ('Yeah, I might have to do that'), indicating consideration/acceptance of the advice to ban Ambessa. (resolution: context_resolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."

#### explicit_relationships (5 items)
- [1] 'it' in the ban advice refers to Ambessa. (resolution: context_resolved; relation: REFERS_TO)
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
- [2] Ambessa being by far better champion than Riven causes/justifies the statement that there is already a good reason to ban Ambessa. (resolution: context_resolved; relation: CAUSES)
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
- [3] The player feeling unable to progress the lane is the condition under which the coach says he should ban Ambessa for sure. (resolution: literal_explicit; relation: CONDITION)
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
- [4] 'that' in 'I might have to do that' refers to the advised action of banning Ambessa. (resolution: context_resolved; relation: REFERS_TO)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
- [5] Against Riven, 'they all in me' is the condition under which the player says he can usually outplay most Rivens. (resolution: context_resolved; relation: CONDITION)
  - cite[76148:76227] (citations verified byte-exact at import): 'against Riven like if they all in me and like I can usually outplay most Rivens'

#### uncertainty_unresolved (2 items)
- [1] Target-boundary caveat: the target passage stops at 'because against Riven like if'; the full transcript resolves it with 'they all in me and like I can usually outplay most Rivens', so no guess was used. (resolution: context_resolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'
  - cite[76148:76227] (citations verified byte-exact at import): 'against Riven like if they all in me and like I can usually outplay most Rivens'
- [2] Whether the player will actually act on the advice to ban Ambessa is not certain; he only says 'I might have to do that'. (resolution: unresolved)
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'

#### supporting_source_spans (1 items)
- [1] Minimal source spans supporting the extraction: the Ambessa/Riven comparison, the reason to ban, the conditional 'not able to progress the lane' advice, the player's reply, and the full-transcript continuation about Riven all-ins. (resolution: literal_explicit)
  - cite[75887:75949] (citations verified byte-exact at import): "To me, it's just Ambessa is by far better champion than Riven."
  - cite[75953:75995] (citations verified byte-exact at import): "So that's already a good reason to ban it."
  - cite[75996:76109] (citations verified byte-exact at import): "But if on top of that, you feel like you're not able to progress the lane, then yeah, you should ban it for..."
  - cite[76110:76169] (citations verified byte-exact at import): 'Yeah, I might have to do that because against Riven like if'
  - cite[76148:76227] (citations verified byte-exact at import): 'against Riven like if they all in me and like I can usually outplay most Rivens'
  - cite[3756:3768] (citations verified byte-exact at import): 'Top >> lane.'

### TARGET p2k:case:0009

Selected candidate: candidate_1 (presentation order: candidate_2, candidate_3, candidate_1, candidate_5, candidate_4)

Verifier rationale: candidate_1 best preserves the source-supported semantics of the target passage. It correctly captures the wave-as-ward reasoning, Mel not being on mid, the player not playing around waves and therefore chasing, and the conditional flip if Mel shows mid. It is also the most careful about the ambiguous 'She's not a bot' wording, marking it unresolved rather than asserting an unsupported bot-lane conclusion. It avoids importing extra advice or outside-passage claims, and its pronoun/reference bindings stay grounded in the passage.

#### actors_entities (6 items)
- [1] Mel (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [2] Poppy (resolution: literal_explicit)
  - cite[56606:56629] (citations verified byte-exact at import): "she's not killing Poppy"
- [3] The addressed player ('you') (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [4] Baron (one of Mel's possible locations in the surrounding reasoning) (resolution: context_resolved)
  - cite[56456:56552] (citations verified byte-exact at import): "Mel only has three options, right? Yeah. So, one is mid, two is Baron, right? And then it's bot."
- [5] mid lane (Mel is not there according to the source) (resolution: context_resolved)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [6] bot/bottom lane (surface wording 'bot'/'a bot'; ambiguous) (resolution: unresolved)
  - cite[56538:56630] (citations verified byte-exact at import): "then it's bot. Yes? Yeah. She's not a bot, because well, obviously, she's not killing Poppy."

#### reference_bindings (3 items)
- [1] 'She' refers to Mel. (resolution: context_resolved)
  - cite[56456:56630] (citations verified byte-exact at import): "Mel only has three options, right? Yeah. So, one is mid, two is Baron, right? And then it's bot. Yes? Yeah...."
- [2] 'you' and 'your' refer to the addressed player. (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [3] 'it' in 'then it's bot' refers to the third listed option among Mel's three options. (resolution: context_resolved)
  - cite[56456:56552] (citations verified byte-exact at import): "Mel only has three options, right? Yeah. So, one is mid, two is Baron, right? And then it's bot."

#### abilities_resources (1 items)
- [1] Ward, used as a vision resource in the metaphor 'waves being your ward'. (resolution: literal_explicit)
  - cite[56701:56722] (citations verified byte-exact at import): 'waves being your ward'

#### events_actions (1 items)
- [1] The addressed player chases. (resolution: context_resolved)
  - cite[56724:56738] (citations verified byte-exact at import): 'So, you chase.'

#### states (5 items)
- [1] Wave equals ward (the wave acts as a ward/vision). (resolution: literal_explicit)
  - cite[56657:56673] (citations verified byte-exact at import): 'wave equals ward'
- [2] Mel is not on mid. (resolution: context_resolved)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [3] Mel is not killing Poppy. (resolution: context_resolved)
  - cite[56606:56629] (citations verified byte-exact at import): "she's not killing Poppy"
- [4] The addressed player does not play around waves being their ward. (resolution: context_resolved)
  - cite[56675:56723] (citations verified byte-exact at import): "But you don't play around waves being your ward."
- [5] Surface wording: 'She's not a bot'; intended location is unresolved. (resolution: unresolved)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."

#### conditions (1 items)
- [1] If Mel is showing on mid. (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'

#### recommended_advice (0 items)

(none)

#### consequences_outcomes (2 items)
- [1] This could be a good flip (if Mel is showing on mid). (resolution: literal_explicit)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [2] The addressed player chases as a result of not playing around waves being their ward. (resolution: context_resolved)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."

#### explicit_relationships (8 items)
- [1] 'She' refers to Mel. (resolution: context_resolved; relation: REFERS_TO)
  - cite[56456:56630] (citations verified byte-exact at import): "Mel only has three options, right? Yeah. So, one is mid, two is Baron, right? And then it's bot. Yes? Yeah...."
- [2] 'you'/'your' refer to the addressed player. (resolution: context_resolved; relation: REFERS_TO)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [3] 'it' in 'then it's bot' refers to the third listed option among Mel's three options. (resolution: context_resolved; relation: REFERS_TO)
  - cite[56456:56552] (citations verified byte-exact at import): "Mel only has three options, right? Yeah. So, one is mid, two is Baron, right? And then it's bot."
- [4] Not playing around waves being your ward leads to chasing. (resolution: context_resolved; relation: CAUSES)
  - cite[56675:56738] (citations verified byte-exact at import): "But you don't play around waves being your ward. So, you chase."
- [5] Mel showing on mid is a condition for 'this could be a good flip'. (resolution: literal_explicit; relation: CONDITION)
  - cite[56739:56791] (citations verified byte-exact at import): 'If Mel is showing on mid, this could be a good flip,'
- [6] The source gives 'because wave equals ward' as the reason Mel is not on mid. (resolution: literal_explicit; relation: CAUSES)
  - cite[56631:56674] (citations verified byte-exact at import): "She's not on mid, because wave equals ward."
- [7] The source gives 'she's not killing Poppy' as the reason for 'She's not a bot'; the intended meaning of 'bot' remains unresolved. (resolution: unresolved; relation: CAUSES)
  - cite[56564:56630] (citations verified byte-exact at import): "She's not a bot, because well, obviously, she's not killing Poppy."
- [8] Poppy is the target of Mel's not-killing clause. (resolution: context_resolved; relation: TARGET)
  - cite[56606:56629] (citations verified byte-exact at import): "she's not killing Poppy"

#### uncertainty_unresolved (1 items)
- [1] The 'bot'/'a bot' wording is ambiguous; it may be a surface mishearing for 'at bot' or another location, and the clause 'she's not killing Poppy' does not cleanly disambiguate it. (resolution: unresolved)
  - cite[56538:56630] (citations verified byte-exact at import): "then it's bot. Yes? Yeah. She's not a bot, because well, obviously, she's not killing Poppy."

#### supporting_source_spans (1 items)
- [1] Target passage source span. (resolution: literal_explicit)
  - cite[56538:56791] (citations verified byte-exact at import): "then it's bot. Yes? Yeah. She's not a bot, because well, obviously, she's not killing Poppy. She's not on m..."

### TARGET p2k:case:0010

Selected candidate: candidate_2 (presentation order: candidate_4, candidate_1, candidate_5, candidate_2, candidate_3)

Verifier rationale: Candidate 2 best preserves the target passage's source-supported semantics. It accurately captures the coached Varus player, the Blitzcrank/Kalista matchup context, the push-as-win-condition, the angle-to-push conditional, and the 'not meant to win this' outcome. It also handles reference binding carefully, explicitly distinguishes the generic 'you know' from game-world 'you', and flags the unresolved MH token, the non-canonical 'bliss crank' surface, and the dangling trailing 'but' without overclaiming. It is the most complete and faithful candidate without adding unsupported semantic claims.

#### actors_entities (4 items)
- [1] The coached ADC player ('you') who is playing Varus. (resolution: context_resolved)
  - cite[22939:22974] (citations verified byte-exact at import): "sometimes you're playing your varus"
  - cite[23119:23140] (citations verified byte-exact at import): "and now you don't win"
- [2] Varus, the champion the coached player is playing. (resolution: literal_explicit)
  - cite[22939:22974] (citations verified byte-exact at import): "sometimes you're playing your varus"
- [3] Blitzcrank (surface 'bliss crank'), the enemy support in the matchup. (resolution: vocabulary_supported)
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'
- [4] Kalista, the enemy ADC in the matchup. (resolution: literal_explicit)
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'

#### reference_bindings (4 items)
- [1] 'you' in 'you don't win' / 'you can win' / 'you do get push' / 'you could get it' / 'you could win' / 'you're not meant' refers to the coached Varus player. (resolution: context_resolved)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
  - cite[22939:22974] (citations verified byte-exact at import): "sometimes you're playing your varus"
- [2] 'it' in 'you could get it' refers to push. (resolution: context_resolved)
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"
- [3] 'this' in 'you're not meant to win this' refers to the current Blitzcrank/Kalista matchup. (resolution: context_resolved)
  - cite[23279:23307] (citations verified byte-exact at import): "you're not meant to win this"
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'
- [4] 'you' in 'you know' is a generic discourse second person rather than a game-world entity. (resolution: unresolved)
  - cite[23279:23320] (citations verified byte-exact at import): "you're not meant to win this you know but"

#### abilities_resources (0 items)

(none)

#### events_actions (2 items)
- [1] The player could get push if there is an angle to get push. (resolution: literal_explicit)
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"
- [2] The player could win after getting push. (resolution: literal_explicit)
  - cite[23253:23271] (citations verified byte-exact at import): 'then you could win'

#### states (3 items)
- [1] The player is currently in a non-winning situation in this matchup. (resolution: literal_explicit)
  - cite[23119:23140] (citations verified byte-exact at import): "and now you don't win"
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'
- [2] The player is not meant to win this matchup. (resolution: literal_explicit)
  - cite[23279:23307] (citations verified byte-exact at import): "you're not meant to win this"
- [3] Push is the wave state identified as the one way to win. (resolution: vocabulary_supported)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'

#### conditions (3 items)
- [1] If the player gets push, then the player could win. (resolution: literal_explicit)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
  - cite[23253:23271] (citations verified byte-exact at import): 'then you could win'
- [2] If there is an angle to get push, then the player could get push. (resolution: literal_explicit)
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"
- [3] In the Blitzcrank/Kalista matchup, the player does not win by default. (resolution: context_resolved)
  - cite[23119:23140] (citations verified byte-exact at import): "and now you don't win"
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'

#### recommended_advice (1 items)
- [1] If there is an angle to get push, get push, because that is the one way to win. (resolution: context_resolved)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"

#### consequences_outcomes (3 items)
- [1] Getting push would lead to the outcome that the player could win. (resolution: literal_explicit)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
  - cite[23253:23271] (citations verified byte-exact at import): 'then you could win'
- [2] The player does not win in the current situation. (resolution: literal_explicit)
  - cite[23119:23140] (citations verified byte-exact at import): "and now you don't win"
- [3] The player is not meant to win this matchup. (resolution: literal_explicit)
  - cite[23279:23307] (citations verified byte-exact at import): "you're not meant to win this"

#### explicit_relationships (7 items)
- [1] The coached Varus player ('you') is the actor who does not win and who could get push and win. (resolution: context_resolved; relation: ACTOR)
  - cite[23119:23140] (citations verified byte-exact at import): "and now you don't win"
  - cite[22939:22974] (citations verified byte-exact at import): "sometimes you're playing your varus"
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
- [2] Blitzcrank ('bliss crank') and Kalista are the opponents the Varus player would face in this matchup. (resolution: vocabulary_supported; relation: TARGET)
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'
- [3] Winning requires getting push; the one way to win is if the player gets push. (resolution: literal_explicit; relation: REQUIRES)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
- [4] An angle to get push is a condition for getting push. (resolution: literal_explicit; relation: CONDITION)
  - cite[23199:23252] (citations verified byte-exact at import): "if there's an angle to get push then you could get it"
- [5] Getting push enables winning. (resolution: literal_explicit; relation: ENABLES)
  - cite[23145:23195] (citations verified byte-exact at import): 'the one way that you can win is if you do get push'
  - cite[23253:23271] (citations verified byte-exact at import): 'then you could win'
- [6] The current situation negates winning ('you don't win'). (resolution: literal_explicit; relation: NEGATES)
  - cite[23119:23140] (citations verified byte-exact at import): "and now you don't win"
- [7] 'this' in 'win this' refers to the Blitzcrank/Kalista matchup. (resolution: context_resolved; relation: REFERS_TO)
  - cite[23279:23307] (citations verified byte-exact at import): "you're not meant to win this"
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'

#### uncertainty_unresolved (3 items)
- [1] 'MH' in 'you could win MH' is an unresolved token; it is likely a backchannel ('mhm') rather than a game term. (resolution: unresolved)
  - cite[23253:23307] (citations verified byte-exact at import): "then you could win MH but you're not meant to win this"
- [2] 'bliss crank' is a non-canonical ASR surface; the context and vocabulary support Blitzcrank, but the literal surface is not the champion name. (resolution: vocabulary_supported)
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'
- [3] The target passage ends with a dangling 'but', leaving the following clause outside the marked target span. (resolution: unresolved)
  - cite[23279:23320] (citations verified byte-exact at import): "you're not meant to win this you know but"

#### supporting_source_spans (1 items)
- [1] Minimal source spans supporting the target extraction: the target passage itself, the Varus-player context, and the Blitzcrank/Kalista matchup context. (resolution: literal_explicit)
  - cite[23114:23320] (citations verified byte-exact at import): "yeah and now you don't win but the one way that you can win is if you do get push so if there's an angle to..."
  - cite[22939:22974] (citations verified byte-exact at import): "sometimes you're playing your varus"
  - cite[23061:23113] (citations verified byte-exact at import): 'you will be against a bliss crank now with a kalista'
