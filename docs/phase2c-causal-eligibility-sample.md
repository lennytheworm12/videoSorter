# Phase 2C Causal Eligibility Sample

Source: `videos.db` insight IDs `4754` through `4803`, selected as one
deterministic contiguous ID range on 2026-08-15. All 50 records exist and have
nonempty text. The range covers three videos, so this is a **small
three-video audit**, not a representative corpus sample:

- `3nKrtwpZ6sQ`: IDs 4754-4775 (22 records)
- `uAdWuLPYn-0`: IDs 4776-4795 (20 records)
- `z5IXabhMLzQ`: IDs 4796-4803 (8 records)

Labels were manually reviewed from the stored insight text. A label measures
whether the text contains a causal mechanism recoverable as a source-grounded
relation; it does **not** assert that the mechanism maps to ontology v0 or
would pass the current compiler.

- A: explicit causal mechanism suitable for direct relation extraction.
- B: implicit but recoverable strategic mechanism.
- C: useful advice/conclusion without sufficient causal mechanism.
- D: unsuitable/noise.

| ID | Label | Causal-eligibility rationale |
| ---: | :---: | --- |
| 4754 | A | Powerful mobility/healing/engage is stated to make unbalanced aggression lead to over-commitment and death. |
| 4755 | C | Mental-state recommendation gives no reusable strategic game mechanism. |
| 4756 | A | Spent resources/cooldowns make the next play losing, so reset/vision is preferred. |
| 4757 | C | Leadership instruction has no stated strategic consequence. |
| 4758 | A | Tower death resets a bad wave and prevents an enemy freeze. |
| 4759 | A | Keeping the cannon alive pulls a freeze toward the tower. |
| 4760 | A | Bush presence creates pressure that forces respect and prevents free trades. |
| 4761 | A | Two-wave crash creates a roam window; staying cannot take the tower. |
| 4762 | B | Rakan E is said to permit tanking a hook; the defensive mechanism is recoverable but incomplete. |
| 4763 | B | Short W reduces over-commitment and exposure to crowd control, but does not name the exact causal state. |
| 4764 | A | Flash is unavailable during R, requiring flash before R for the instant combo. |
| 4765 | B | Saving Flash is conditioned on future survival being able to turn a later fight; utility is implicit. |
| 4766 | B | Available Guardian/E can make hook-tanking for the ADC viable; the protection mechanism is implied. |
| 4767 | B | Reliable CC makes aggressive W punishable unless it lands or the defensive cooldown is baited. |
| 4768 | A | Dive/plates provide more gold and tempo than the low-value dragon. |
| 4769 | C | Role assignment around Baron gives no stated causal effect of warding/scouting. |
| 4770 | C | Information-check instruction does not say how a specific spell/item changes the choice. |
| 4771 | B | Sweeper is reserved for places/times where enemy vision is plausibly present; value mechanism is implicit. |
| 4772 | A | Mid-lane wards track movement/rotations and are more valuable than deep wards before recall. |
| 4773 | B | Tankiness is selected to survive burst/lockdown; item-to-survival mechanism is recoverable but broad. |
| 4774 | B | Heavy CC/magic damage motivates Mercury's Treads, but the specific defensive effect is unstated. |
| 4775 | C | Practice-volume advice does not state a reusable in-game causal mechanism. |
| 4776 | B | Bard's off-map presence/unkillability is said to create space, but the relation is broad and champion-scoped. |
| 4777 | B | Tank items support standing in enemy space/being unkillable; the state transition is implied. |
| 4778 | A | Excessive game quantity causes mental fatigue and teammate-dependent flipping. |
| 4779 | A | Unmatched push permits a slow push that enables dive/zone. |
| 4780 | A | Inability to interact makes Relic Shield preferable to losing Spellthief trades. |
| 4781 | A | Unwarded bush entry lets Pyke charge hook and chunk HP before reaction. |
| 4782 | A | Hitting minions ensures crash; otherwise the lane cannot be contested. |
| 4783 | A | Recall after crash yields items/vision and avoids death or a bad trade. |
| 4784 | B | Pre-casting portal before engagement makes the escape available; timing-to-availability is implied. |
| 4785 | B | Q is recommended for stun when E is unreliable away from a wall; ability-function link is implicit. |
| 4786 | A | Baiting a high-impact ultimate onto self creates a team follow-up window. |
| 4787 | A | Warding/drawing attention creates engage space for the fed teammate. |
| 4788 | B | Objective setup vision plus reset is recommended, but why that sequence is superior is only implicit. |
| 4789 | C | Presence/vision setup direction contains no stated causal result. |
| 4790 | A | Front positioning by a tank creates space and draws cooldowns for carries. |
| 4791 | C | Communication instruction does not establish a tactical causal effect. |
| 4792 | A | Sweeping mid removes vision and limits enemy rotations. |
| 4793 | A | Swiftness enables positions the enemy support cannot reach; CDR is less relevant to the named playstyle. |
| 4794 | A | Health makes Randuin's more versatile against mixed damage. |
| 4795 | C | Information-check instruction does not give a source-grounded decision mechanism. |
| 4796 | C | Matchup conclusion lacks the mechanism that makes soft engage favorable or primary engage unfavorable. |
| 4797 | B | Thresh's kit is said to peel for immobile front-to-back carries, but the specific kit mechanism is unstated. |
| 4798 | B | Lantern-saving position versus frontlining implies a positional access/peel mechanism. |
| 4799 | C | Win-condition instruction does not state why the named macro line follows. |
| 4800 | C | Support communication responsibility contains no causal strategic mechanism. |
| 4801 | A | Pushing mid first forces a response and prevents collapse on split pushers. |
| 4802 | B | Base/farm time creates a window for vision denial/deep wards, with the value of that setup implicit. |
| 4803 | A | Challenging creeps forces enemy mispositioning. |

| Label | Count |
| --- | ---: |
| A | 23 |
| B | 16 |
| C | 11 |
| D | 0 |

Result for this small slice: 39/50 (78%) records are A/B causal-eligible;
11/50 (22%) are safe-zero cases. This supports a **narrow hypothesis** that
some existing insights preserve causal material beyond the 18-case benchmark.
It does not establish corpus-wide evidence sufficiency, ontology-v0 coverage,
or that the compiler alone explains Phase 2C's zero-recall result. A future
decision about corpus re-extraction needs a stratified, independently labeled
sample across videos, champions, creators, insight types, and confidence
bands. C/D zero output remains correct behavior for this compiler.
