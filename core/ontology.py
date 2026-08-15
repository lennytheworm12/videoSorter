"""Strategic ontology v0 for League reasoning experiments.

The ontology is intentionally small. New phrases should first be represented as
relations among these primitives before adding new concepts.
"""

from __future__ import annotations

from dataclasses import dataclass


ONTOLOGY_VERSION = "strategic-ontology-v0"


@dataclass(frozen=True)
class StrategicConcept:
    canonical_name: str
    concept_type: str
    description: str
    scope: str = "global"
    patch_sensitivity: str = "very_low"


STRATEGIC_CONCEPTS: dict[str, StrategicConcept] = {
    "access": StrategicConcept(
        "access",
        "interaction_control",
        "Ability to begin meaningful interaction with a target.",
    ),
    "continuity": StrategicConcept(
        "continuity",
        "interaction_control",
        "Ability to remain connected long enough to convert the interaction.",
    ),
    "range_asymmetry": StrategicConcept(
        "range_asymmetry",
        "space_control",
        "One side can interact while the other cannot.",
    ),
    "territory": StrategicConcept(
        "territory",
        "space_control",
        "Space a player can occupy without accepting an unfavorable interaction.",
    ),
    "persistent_pressure": StrategicConcept(
        "persistent_pressure",
        "resource_pattern",
        "Threat that can be applied repeatedly at low marginal cost.",
    ),
    "intermittent_pressure": StrategicConcept(
        "intermittent_pressure",
        "resource_pattern",
        "Threat gated by cooldown, mana, charges, or another discrete resource.",
    ),
    "threat_preservation": StrategicConcept(
        "threat_preservation",
        "resource_pattern",
        "Value created by holding a spell or resource rather than spending it.",
    ),
    "resource_exchange": StrategicConcept(
        "resource_exchange",
        "resource_pattern",
        "Spending one resource to attack another strategically important resource.",
    ),
    "combat_compression": StrategicConcept(
        "combat_compression",
        "fight_shape",
        "Reducing space or time until combat becomes a closed continuous fight.",
    ),
    "combat_expansion": StrategicConcept(
        "combat_expansion",
        "fight_shape",
        "Increasing space or time to preserve kiting, disengage, or delay.",
    ),
    "isolation": StrategicConcept(
        "isolation",
        "targeting",
        "Reducing the number of enemies who can interact with a target.",
    ),
    "wave_obligation": StrategicConcept(
        "wave_obligation",
        "lane_state",
        "Wave state forcing a player to expose themselves or take action.",
    ),
    "initiative": StrategicConcept(
        "initiative",
        "state_evaluation",
        "Which side can dictate or must answer state change.",
    ),
    "role_transfer": StrategicConcept(
        "role_transfer",
        "state_transition",
        "Event that changes who must create or preserve the state.",
    ),
    "conversion": StrategicConcept(
        "conversion",
        "state_transition",
        "Turning temporary advantage into durable value.",
    ),
    "reset": StrategicConcept(
        "reset",
        "state_transition",
        "Using recall, Teleport, or reset mechanics to renew pressure.",
    ),
    "tempo": StrategicConcept(
        "tempo",
        "state_evaluation",
        "Time advantage that changes who can act first or maintain pressure.",
    ),
    "local_numbers": StrategicConcept(
        "local_numbers",
        "targeting",
        "How many units or champions can interact with the relevant target.",
    ),
    "default_trajectory": StrategicConcept(
        "default_trajectory",
        "state_evaluation",
        "What happens if neither side successfully changes the state.",
    ),
    "winning_line": StrategicConcept(
        "winning_line",
        "state_transition",
        "Best reachable state transition sequence for a side from the current position.",
    ),
}


RELATION_TYPES = frozenset(
    {
        "creates",
        "requires",
        "denies",
        "enables",
        "amplifies",
        "reduces",
        "forces",
        "converts_into",
        "is_countered_by",
        "preserves",
        "consumes",
        "exchanges_for",
        "transfers_initiative_to",
        "expands",
        "compresses",
        "increases_cost_of",
        "reduces_cost_of",
    }
)

ENTITY_TYPES = frozenset(
    {
        "champion",
        "ability",
        "concept",
        "archetype",
        "lane_pair",
        "principle",
        "state",
        "event",
    }
)

PROVENANCE_TYPES = frozenset(
    {
        "source_claim",
        "coach_supported_inference",
        "system_derived_principle",
        "speculative_hypothesis",
        "manual_fixture",
    }
)

PATCH_SENSITIVITY = frozenset({"very_low", "low", "medium", "high"})
