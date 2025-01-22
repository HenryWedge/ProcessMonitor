import json
import string

from process_mining_core.datastructure.core.model.petri_net import SerializablePetriNet

class PetriNetSerDes:

    def serialize(self, petri_net: SerializablePetriNet) -> string:
        petri_net_dict = dict()
        petri_net_dict["places"] = list(petri_net.places)
        petri_net_dict["transitions"] = list(petri_net.transitions)
        petri_net_dict["silent_transitions"] = list(petri_net.silent_transitions)
        petri_net_dict["arc_place_transition"] = list(petri_net.arc_place_transition)
        petri_net_dict["arc_transition_place"] = list(petri_net.arc_transition_place)
        petri_net_dict["start_activities"] = list(petri_net.start_activities)
        petri_net_dict["end_activities"] = list(petri_net.end_activities)
        return json.dumps(petri_net_dict)

    def list_to_tuple(self, transitions):
        transitions_as_tuple = []
        for transition in transitions:
            transitions_as_tuple.append((transition[0], transition[1]))
        return transitions_as_tuple

    def transform_to_net(self, net_dict):
        return SerializablePetriNet(
            places=net_dict["places"],
            silent_transitions=net_dict["silent_transitions"],
            transitions=net_dict["transitions"],
            arc_place_transition=self.list_to_tuple(net_dict["arc_place_transition"]),
            arc_transition_place=self.list_to_tuple(net_dict["arc_transition_place"]),
            start_activities=net_dict["start_activities"],
            end_activities=net_dict["end_activities"]
        )

    def deserialize(self, petri_net_str: string):
        petri_net_dict = json.loads(petri_net_str)
        return self.transform_to_net(petri_net_dict)
