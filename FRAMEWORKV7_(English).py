# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 09:48:47 2025

@author: Enrique Osuna
"""


import json
import threading
import time
import networkx as nx
import xml.etree.ElementTree as ET
from owlready2 import get_ontology, sync_reasoner
from fuzzywuzzy import process, fuzz

# ============================================================
# Function to print formatted phase headers
# ============================================================
def print_phase_header(phase_number, phase_description):
    header = f"PHASE {phase_number}: {phase_description}"
    border = "=" * len(header)
    print("\n" + border.center(80))
    print(header.center(80))
    print(border.center(80) + "\n")

# ============================================================
# Auxiliary functions for dynamic dependency resolution with reasoning
# ============================================================
def is_compatible(required_mapping, output_mapping, ontology):
    req_term = required_mapping["matched_term"].lower()
    out_term = output_mapping["matched_term"].lower()
    
    if req_term == "appraisal_variable":
        appraisal_prop = getattr(ontology, "appraisal_variable", None)
        if appraisal_prop is None:
            return req_term == out_term
        for dom in appraisal_prop.domain:
            out_class = getattr(ontology, out_term, None)
            if out_class is not None:
                if out_class == dom or out_class in dom.subclasses():
                    return True
        return False
    return req_term == out_term

def resolve_dependency(required_term, required_mapping, component_buffer, affective_buffer, ontology):
    for comp_id, output in affective_buffer.buffer.items():
        out_mappings = component_buffer[comp_id]["mappings"]
        for out_key, mapping in out_mappings.items():
            if is_compatible(required_mapping, mapping, ontology):
                print(f"   [Resolver] '{required_term}' (mapping: {required_mapping}) is compatible with the output '{out_key}' from {comp_id} (mapping: {mapping}).")
                return output.get(out_key)
    print(f"   [Resolver] No compatible output found for '{required_term}'.")
    return None

# ============================================================
# Function to enrich output with semantic information
# ============================================================
def enrich_output(comp_id, output, component_buffer, ontology, inputs_used, original_output):
    enriched = {}
    mappings = component_buffer[comp_id]["mappings"]
    for key, value in output.items():
        if key in mappings:
            mapping = mappings[key]
            matched_term = mapping["matched_term"]
            semantic_type = mapping["type"]
            entity = getattr(ontology, matched_term, None)
            if entity is not None:
                ancestors = [a.name for a in entity.ancestors() if a.name not in ['Thing', 'owl:Thing']]
                ontology_iri = getattr(entity, "iri", None)
            else:
                ancestors = []
                ontology_iri = None
            enriched[key] = {
                "value": value,
                "matched_term": matched_term,
                "semantic_type": semantic_type,
                "ontology_ancestors": ancestors,
                "ontology_iri": ontology_iri,
                "processing_trace": {
                    "component_inputs": inputs_used,
                    "original_output": original_output
                }
            }
        else:
            enriched[key] = {"value": value}
    return enriched

# ============================================================
# Definition of base classes and affective components
# ============================================================
class AffectiveComponent:
    def __init__(self, component_id, theory, input_requirements, output_specification):
        self.component_id = component_id
        self.theory = theory
        self.input_requirements = input_requirements
        self.output_specification = output_specification

    def execute(self, inputs):
        raise NotImplementedError("Must be implemented in the subclass.")

class Infra_Physiological_Evaluation(AffectiveComponent):
    def __init__(self):
        super().__init__(
            component_id="Infra_Physiological_Evaluation",
            theory="occ",
            input_requirements=["stimulus"],
            output_specification={"desirability": 0.1, "unexpectedness": 0.9, "familiarity": 0.1}
        )
    def execute(self, inputs):
        print(f"[{self.component_id}] Executing with inputs: {inputs}")
        stimulus = inputs.get("stimulus")
        if stimulus is None:
            raise ValueError("Missing 'stimulus' in Infra_Physiological_Evaluation.")
        output = {"desirability": 0.1, "unexpectedness": 0.9, "familiarity": 0.1}
        print(f"[{self.component_id}] Output: {output}")
        return output

class EEGS_Emotional_State(AffectiveComponent):
    def __init__(self):
        super().__init__(
            component_id="EEGS_Emotional_State",
            theory="occ",
            input_requirements=["appraisal_variable"],
            output_specification={"emotion": "fear", "intensity": 0.8}
        )
    def execute(self, inputs):
        print(f"[{self.component_id}] Executing with inputs: {inputs}")
        appraisal = inputs.get("appraisal_variable")
        if appraisal is None:
            raise ValueError("Missing 'appraisal_variable' in EEGS_Emotional_State.")
        output = {"emotion": "fear", "intensity": 0.8}
        print(f"[{self.component_id}] Output: {output}")
        return output

class Infra_Emotional_Response(AffectiveComponent):
    def __init__(self):
        super().__init__(
            component_id="Infra_Emotional_Response",
            theory="frijda",
            input_requirements=["emotion", "appraisal_variable"],
            output_specification={"action_tendency": "withdrawal"}
        )
    def execute(self, inputs):
        print(f"[{self.component_id}] Executing with inputs: {inputs}")
        emotion = inputs.get("emotion")
        appraisal = inputs.get("appraisal_variable")
        if emotion is None and appraisal is None:
            raise ValueError("Neither 'emotion' nor 'appraisal_variable' received in Infra_Emotional_Response.")
        if emotion is None:
            print(f"[{self.component_id}] Only 'appraisal_variable' received.")
        if appraisal is None:
            print(f"[{self.component_id}] Only 'emotion' received.")
        output = {"action_tendency": "withdrawal"}
        print(f"[{self.component_id}] Output: {output}")
        return output

class FLAME_Cognitive_Evaluation(AffectiveComponent):
    def __init__(self):
        super().__init__(
            component_id="FLAME_Cognitive_Evaluation",
            theory="occ",
            input_requirements=["stimulus"],
            output_specification={"desirability": 0.3}
        )
    def execute(self, inputs):
        print(f"[{self.component_id}] Executing with inputs: {inputs}")
        stimulus = inputs.get("stimulus")
        if stimulus is None:
            raise ValueError("Missing 'stimulus' in FLAME_Cognitive_Evaluation.")
        output = {"desirability": 0.3}
        print(f"[{self.component_id}] Output: {output}")
        return output

class EEGS_Physiological_Evaluation(AffectiveComponent):
    def __init__(self):
        super().__init__(
            component_id="EEGS_Physiological_Evaluation",
            theory="lambie",
            input_requirements=["stimulus"],
            output_specification={"phenomenological_appraisal": 0.8}
        )
    def execute(self, inputs):
        print(f"[{self.component_id}] Executing with inputs: {inputs}")
        stimulus = inputs.get("stimulus")
        if stimulus is None:
            raise ValueError("Missing 'stimulus' in EEGS_Physiological_Evaluation.")
        output = {"phenomenological_appraisal": 0.8}
        print(f"[{self.component_id}] Output: {output}")
        return output

class Infra_Emotional_State_Ekman(AffectiveComponent):
    def __init__(self):
        super().__init__(
            component_id="Infra_Emotional_State_Ekman",
            theory="ekman",
            input_requirements=["appraisal_variable", "phenomenological_appraisal"],
            output_specification={"emotion": "fear", "intensity": 0.9}
        )
    def execute(self, inputs):
        print(f"[{self.component_id}] Executing with inputs: {inputs}")
        appraisal = inputs.get("appraisal_variable")
        phen_app = inputs.get("phenomenological_appraisal")
        if appraisal is None or phen_app is None:
            raise ValueError("Missing 'appraisal_variable' or 'phenomenological_appraisal' in Infra_Emotional_State_Ekman.")
        output = {"emotion": "fear", "intensity": 0.9}
        print(f"[{self.component_id}] Output: {output}")
        return output

class ALMA_mood_state(AffectiveComponent):
    def __init__(self):
        super().__init__(
            component_id="ALMA_mood_state",
            theory="pad",
            input_requirements=["emotion"],
            output_specification={"anxious": 0.6}
        )
    def execute(self, inputs):
        print(f"[{self.component_id}] Executing with inputs: {inputs}")
        emotion = inputs.get("emotion")
        if emotion is None:
            raise ValueError("Missing 'emotion' in ALMA_mood_state.")
        output = {"anxious": 0.6}
        print(f"[{self.component_id}] Output: {output}")
        return output

# ============================================================
# Management and coordination of the framework
# ============================================================
class CentralExecutive:
    def __init__(self):
        self.temporary_memory = {}
    def register_component(self, component):
        metadata = {
            "component_id": component.component_id,
            "theory": component.theory,
            "input_requirements": component.input_requirements,
            "output_specification": component.output_specification
        }
        self.temporary_memory[component.component_id] = metadata
        print(f"[CentralExecutive] Registered: {component.component_id}")
        print(f"   Metadata: {metadata}")
    def notify_component(self, component, inputs):
        print(f"[CentralExecutive] Notifying '{component.component_id}' for execution.")
        return component.execute(inputs)

class SemanticController:
    def __init__(self, ontology_path):
        try:
            self.ontology = get_ontology(ontology_path).load()
            sync_reasoner(self.ontology)
            print(f"[SemanticController] Ontology loaded and reasoned from: {ontology_path}")
        except Exception as e:
            print(f"[SemanticController] Error: {e}")
            self.ontology = None
    def perform_semantic_matching(self, temporary_memory):
        print("[SemanticController] Starting semantic matching of metadata...")
        candidate_terms = []
        candidate_types = {}
        for cls in self.ontology.classes():
            candidate_terms.append(cls.name)
            candidate_types[cls.name] = "class"
        for dp in self.ontology.data_properties():
            candidate_terms.append(dp.name)
            candidate_types[dp.name] = "data_property"
        for op in self.ontology.object_properties():
            candidate_terms.append(op.name)
            candidate_types[op.name] = "object_property"
        print(f"[SemanticController] Candidates: {candidate_terms}\n")
        component_buffer = {}
        threshold = 70
        for comp_id, metadata in temporary_memory.items():
            mappings = {}
            print(f"[SemanticController] Processing: {comp_id}")
            for req in metadata["input_requirements"]:
                best_match, score = process.extractOne(req, candidate_terms, scorer=fuzz.ratio)
                if score >= threshold:
                    mappings[req] = {"matched_term": best_match, "score": score, "type": candidate_types[best_match]}
                    print(f"   [Input] '{req}' -> '{best_match}' (score: {score}, type: {candidate_types[best_match]})")
                else:
                    raise ValueError(f"[SemanticController] '{req}' has no valid mapping (score: {score})")
            for key in metadata["output_specification"]:
                best_match, score = process.extractOne(key, candidate_terms, scorer=fuzz.ratio)
                if score >= threshold:
                    mappings[key] = {"matched_term": best_match, "score": score, "type": candidate_types[best_match]}
                    print(f"   [Output] '{key}' -> '{best_match}' (score: {score}, type: {candidate_types[best_match]})")
                else:
                    raise ValueError(f"[SemanticController] '{key}' has no valid mapping (score: {score})")
            component_buffer[comp_id] = {"metadata": metadata, "mappings": mappings}
            print("")
        print("[SemanticController] Semantic matching completed.\n")
        return component_buffer

class ExecutionCoordinator:
    def __init__(self):
        self.execution_plan = []
    def plan_execution(self, sequence_structure):
        # sequence_structure is the preconfigured sequence.
        # It can be a string (sequential) or a nested list (parallel execution).
        self.execution_plan = sequence_structure
        print(f"[ExecutionCoordinator] Planned sequence: {self.execution_plan}")
        return self.execution_plan

class AffectiveBuffer:
    def __init__(self):
        self.buffer = {}
    def update(self, component_id, output):
        self.buffer[component_id] = output
        print(f"[AffectiveBuffer] Updated '{component_id}': {output}")
    def get(self, key, default=None):
        return self.buffer.get(key, default)
    def consolidate(self):
        print("[AffectiveBuffer] Consolidating outputs...")
        return self.buffer

# ============================================================
# Auxiliary functions for execution and XML generation
# ============================================================
def enrich_output(comp_id, output, component_buffer, ontology, inputs_used, original_output):
    enriched = {}
    mappings = component_buffer[comp_id]["mappings"]
    for key, value in output.items():
        if key in mappings:
            mapping = mappings[key]
            matched_term = mapping["matched_term"]
            semantic_type = mapping["type"]
            entity = getattr(ontology, matched_term, None)
            if entity is not None:
                ancestors = [a.name for a in entity.ancestors() if a.name not in ['Thing', 'owl:Thing']]
                ontology_iri = getattr(entity, "iri", None)
            else:
                ancestors = []
                ontology_iri = None
            enriched[key] = {
                "value": value,
                "matched_term": matched_term,
                "semantic_type": semantic_type,
                "ontology_ancestors": ancestors,
                "ontology_iri": ontology_iri,
                "processing_trace": {
                    "component_inputs": inputs_used,
                    "original_output": original_output
                }
            }
        else:
            enriched[key] = {"value": value}
    return enriched

def execute_component(comp_id, components_dict, component_buffer, inputs, central_executive, affective_buffer, ontology):
    comp_instance = components_dict[comp_id]
    req_inputs = {}
    print(f"\n[Execute] Preparing inputs for '{comp_id}'...")
    for req in comp_instance.input_requirements:
        if req in inputs:
            req_inputs[req] = inputs[req]
            print(f"   Input '{req}' from general inputs: {inputs[req]}")
        else:
            mapping = component_buffer[comp_id]["mappings"].get(req)
            resolved = resolve_dependency(req, mapping, component_buffer, affective_buffer, ontology)
            req_inputs[req] = resolved
            print(f"   Resolution: '{req}' -> {resolved}")
    start_time = time.time()
    output = central_executive.notify_component(comp_instance, req_inputs)
    original_output = output.copy()
    end_time = time.time()
    duration = end_time - start_time
    output.update({
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "thread_id": threading.get_ident()
    })
    enriched_output = enrich_output(comp_id, output, component_buffer, ontology, req_inputs, original_output)
    affective_buffer.update(comp_id, enriched_output)
    inputs.update(enriched_output)
    print(f"[Execute] Updated inputs: {inputs}")

def execute_sequence(sequence, components_dict, component_buffer, inputs, central_executive, affective_buffer, ontology):
    for step in sequence:
        if isinstance(step, str):
            execute_component(step, components_dict, component_buffer, inputs, central_executive, affective_buffer, ontology)
        elif isinstance(step, list):
            threads = []
            for comp_id in step:
                t = threading.Thread(target=execute_component, args=(comp_id, components_dict, component_buffer, inputs, central_executive, affective_buffer, ontology))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        else:
            raise ValueError("The sequence must contain strings or lists.")
    return inputs


def consolidate_to_xml(affective_buffer):
    # Create the root XML element
    root = ET.Element("AffectiveState")
    
    for comp_id, enriched_output in affective_buffer.items():
        # Create an element for each component
        comp_elem = ET.SubElement(root, "Component", name=comp_id)
        # Separate timing information and variables
        timing_keys = {"start_time", "end_time", "duration", "thread_id"}
        timing_elem = ET.SubElement(comp_elem, "Timing")
        variables_elem = ET.SubElement(comp_elem, "Variables")
        
        for key, content in enriched_output.items():
            if key in timing_keys:
                sub = ET.SubElement(timing_elem, key)
                # content can be a dictionary with "value" or a direct value
                if isinstance(content, dict) and "value" in content:
                    sub.text = str(content["value"])
                else:
                    sub.text = str(content)
            else:
                var_elem = ET.SubElement(variables_elem, "Variable", name=key)
                # Add enriched fields
                if isinstance(content, dict):
                    for field in ["value", "matched_term", "semantic_type", "ontology_iri"]:
                        if field in content:
                            sub = ET.SubElement(var_elem, field.capitalize())
                            sub.text = str(content[field])
                    if "ontology_ancestors" in content:
                        anc_elem = ET.SubElement(var_elem, "OntologyAncestors")
                        for anc in content["ontology_ancestors"]:
                            a = ET.SubElement(anc_elem, "Ancestor")
                            a.text = anc
                    if "processing_trace" in content:
                        trace_elem = ET.SubElement(var_elem, "ProcessingTrace")
                        inputs_elem = ET.SubElement(trace_elem, "ComponentInputs")
                        for input_name, input_value in content["processing_trace"]["component_inputs"].items():
                            in_elem = ET.SubElement(inputs_elem, "Input", name=input_name)
                            in_elem.text = str(input_value)
                        orig_elem = ET.SubElement(trace_elem, "OriginalOutput")
                        for orig_key, orig_value in content["processing_trace"]["original_output"].items():
                            o_elem = ET.SubElement(orig_elem, "Output", name=orig_key)
                            o_elem.text = str(orig_value)
                else:
                    var_elem.text = str(content)
    
    # Convert the XML tree to a string
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    return xml_str

# ============================================================
# Main function of the framework
# ============================================================
def main():
    print_phase_header(1, "Component Registration and Initial Configuration")
    ontology_path = r"C:/Users/ocmas/OneDrive/Escritorio/Doctorado/FRAMEWORK (implementación)/CMEs_ontology.owl"
    central_executive = CentralExecutive()
    semantic_controller = SemanticController(ontology_path)
    execution_coordinator = ExecutionCoordinator()
    affective_buffer = AffectiveBuffer()
    
    # Instantiate all available components
    comp1 = Infra_Physiological_Evaluation()
    comp2 = EEGS_Emotional_State()
    comp3 = Infra_Emotional_Response()
    comp4 = FLAME_Cognitive_Evaluation()
    comp5 = EEGS_Physiological_Evaluation()
    comp6 = Infra_Emotional_State_Ekman()
    comp7 = ALMA_mood_state()
    
    components_dict = {
        comp1.component_id: comp1,
        comp2.component_id: comp2,
        comp3.component_id: comp3,
        comp4.component_id: comp4,
        comp5.component_id: comp5,
        comp6.component_id: comp6,
        comp7.component_id: comp7
    }
    
    print("[PHASE 1] Component registration:")
    for comp in components_dict.values():
        central_executive.register_component(comp)
    temporary_memory = central_executive.temporary_memory
    print(f"[PHASE 1] TemporaryMemory:\n{json.dumps(temporary_memory, indent=2)}\n")
    
    print_phase_header(2, "Semantic Matching with Ontology")
    component_buffer = semantic_controller.perform_semantic_matching(temporary_memory)
    print(f"[PHASE 2] ComponentBuffer:\n{json.dumps(component_buffer, indent=2)}\n")

    print_phase_header(3, "Definition of Preconfigured Sequences")
    preconfigured_sequences = {
        "A1": ["Infra_Physiological_Evaluation", "EEGS_Emotional_State", "Infra_Emotional_Response"],
        "A2": ["Infra_Physiological_Evaluation", ["EEGS_Emotional_State", "Infra_Emotional_Response"]],
        "A3": ["FLAME_Cognitive_Evaluation", "EEGS_Emotional_State", "Infra_Emotional_Response"],
        "A4": [["FLAME_Cognitive_Evaluation", "EEGS_Physiological_Evaluation"], "Infra_Emotional_State_Ekman", "Infra_Emotional_Response"],
        "A5": [["FLAME_Cognitive_Evaluation", "EEGS_Physiological_Evaluation"], "Infra_Emotional_State_Ekman", ["Infra_Emotional_Response", "ALMA_mood_state"]]
    }
    print("[PHASE 3] Available sequences:")
    for agent, seq in preconfigured_sequences.items():
        print(f"   {agent}: {seq}")
    print("")
    chosen_sequence = "A5"  # Change to "A2", "A3", "A4", or "A5" as desired
    sequence_to_run = preconfigured_sequences[chosen_sequence]
    execution_plan = execution_coordinator.plan_execution(sequence_to_run)
    
    print_phase_header(4, "Execution of the Component Sequence")
    inputs = {"stimulus": "test_stimulus"}
    final_inputs = execute_sequence(execution_plan, components_dict, component_buffer, inputs, central_executive, affective_buffer, semantic_controller.ontology)
    
    print_phase_header(5, "Generation of the Final Semantic Representation")
    # Consolidate and display in JSON format
    final_representation = affective_buffer.consolidate()
    print("[PHASE 5] Final representation in JSON:")
    print(json.dumps(final_representation, indent=2, ensure_ascii=False))
    
    # Additionally, generate semantic XML
    xml_output = consolidate_to_xml(final_representation)
    with open("final_affective_representation.xml", "w", encoding="utf-8") as f:
        f.write(xml_output)
    print("[PHASE 5] Final representation stored in 'final_affective_representation.xml'.")
    
if __name__ == "__main__":
    main()

