# Suggestion: Improve Introduction with Explicit Definitions

## Problem

The current intro (Section 1) jumps from "orientation selectivity" to tuning curves and sharpness without defining these terms for someone encountering them for the first time. A reader who doesn't already know what orientation selectivity or reliability mean will struggle.

Three specific gaps:
1. **Orientation selectivity** is never defined in plain language
2. **Tuning sharpness** is introduced in the questions but not explained until the methods
3. **Reliability** appears in Question 2 without definition

## Suggested Revisions

### Current paragraph 1 (lines 138–144 of `31_generate_report_apr5.py`):

> Neurons in the primary visual cortex (V1) respond selectively to the orientation of visual stimuli, a property first described by Hubel and Wiesel (1962). This orientation selectivity is a fundamental building block of visual processing. Understanding how orientation is encoded across large neural populations remains a central question in systems neuroscience, with implications for sensory prosthetics and brain-computer interfaces.

### Suggested replacement:

> Neurons in the primary visual cortex (V1) are orientation selective: they respond strongly to some angles of visual stimuli but weakly or not at all to others. This property, first described by Hubel and Wiesel (1962), is a fundamental building block of visual processing. Each orientation-selective neuron has a preferred orientation to which it responds most vigorously, and a characteristic tuning sharpness that describes how narrowly focused that preference is — a sharply tuned neuron responds to a narrow range of angles around its preferred orientation, while a broadly tuned neuron responds to many orientations with less discrimination. Understanding how these tuning properties vary across large neural populations, and how they collectively encode orientation, remains a central question in systems neuroscience.

### What changed:
- **Defined orientation selectivity** in the first sentence: "respond strongly to some angles... but weakly or not at all to others"
- **Defined preferred orientation and tuning sharpness** in context, before the research questions reference them
- **Used concrete, intuitive language** — "narrowly focused," "responds to a narrow range of angles"
- Removed the prosthetics/BCI line (not connected to anything in the report)

### Add reliability definition before the questions:

After the questions list and before "These questions bridge...", add:

> By "reliability" (Question 2), we mean how consistently a neuron responds to the same stimulus across repeated presentations — a reliable neuron produces similar activity each time the same orientation is shown, while an unreliable neuron's responses vary widely from trial to trial.

## Code Change

In `scripts/31_generate_report_apr5.py`, replace the first `body(doc, ...)` call (lines 138–144) with the revised text above. Add the reliability definition as a new `body()` call between the questions list and the bridging sentence.

## Impact

Purely textual. No data or analysis changes needed. Makes the report accessible to readers without a neuroscience background — which includes most BENG 2800 classmates and the grading TA.
