# Unified Emotional Processing Framework

This repository contains the reference implementation of the framework proposed in the research article: **"Unified Emotional Processing: Towards a Common Framework for Computational Models of Emotion"**, submitted to *IEEE Transactions on Affective Computing*.

## Overview
Current Computational Models of Emotion (CMEs) often operate as isolated systems characterized by architectural rigidity and fixed execution cycles. This framework serves as a mediating core designed to integrate heterogeneous affective components into unified, theoretically coherent processes. By decoupling domain-specific implementations, the system employs an ontology-based semantic controller to align disparate terminologies and a dependency coordinator to enable configurable execution sequences.

## Key Features
* **Ontology-Based Semantic Alignment**: Resolves terminological inconsistencies using a formal affective ontology comprising 139 elements.
* **Dynamic Execution Orchestration**: Employs a dependency-aware execution planner that identifies independent component groups for parallel execution while managing sequential data flow.
* **Machine-Readable Traceability**: Enriches every affective output with formal metadata and ontological provenance, yielding a semantic trace suitable for explainability analysis.
* **Architectural Extensibility**: Supports the modular integration of diverse third-party components (e.g., appraisal evaluation, mood dynamics, behavioral generators) without requiring structural redesigns.

## Repository Contents

* **[FRAMEWORKV7_(English).py](FRAMEWORKV7_(English).py)**: The main Python implementation containing the core modules (Central Executive, Semantic Controller, and Execution Coordinator).
* **[CMEs_ontology.owl](CMEs_ontology.owl)**: The formal OWL ontology file containing the 139 affective elements used for semantic alignment.
* **[Supplementary_Material_Tables.pdf](Supplementary_Material_Tables.pdf)**: Detailed reference data excluded from the main manuscript due to length constraints. This document includes:
    * **Table VII**: Comprehensive listings of affective labels and theories extracted per model (ALMA, EEGS, EMIA, FLAME, EMA).
    * **Table VIII**: Unified list of variability scenarios used for the semantic matching validation in Case Study 2.


