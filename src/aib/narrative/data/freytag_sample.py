"""Freytag-inspired sample narratives for rhetorical classification."""

from __future__ import annotations

from ..models import Category, NarrativeExample

FREYTAG_CATEGORIES: tuple[Category, ...] = (
    Category("exposition", "Introduces setting, characters, or background."),
    Category("rising_action", "Conflict or tension builds."),
    Category("climax", "Peak tension or turning point."),
    Category("resolution", "Conflict resolved, outcome revealed."),
)

TRAIN_EXAMPLES: tuple[NarrativeExample, ...] = (
    NarrativeExample(
        "In a quiet village at the edge of the forest, a young baker named Mara lived alone.",
        labels=("exposition",),
    ),
    NarrativeExample(
        "Each night the wolves drew closer, and the villagers began to bar their doors.",
        labels=("rising_action",),
    ),
    NarrativeExample(
        "Mara stood between the pack and the children, torch raised, as the alpha lunged.",
        labels=("climax",),
    ),
    NarrativeExample(
        "By spring the wolves had retreated, and Mara was elected mayor.",
        labels=("resolution",),
    ),
    NarrativeExample(
        "The old lighthouse had stood empty for decades before Eli moved in.",
        labels=("exposition",),
    ),
)

VAL_EXAMPLES: tuple[NarrativeExample, ...] = (
    NarrativeExample(
        "Storm clouds gathered as the fishing boats failed to return for the third day.",
        labels=("rising_action",),
    ),
    NarrativeExample(
        "Eli climbed the tower alone and lit the beacon as waves crashed below.",
        labels=("climax",),
    ),
    NarrativeExample(
        "The fleet found its way home, and the harbor celebrated until dawn.",
        labels=("resolution",),
    ),
)
