import dataclasses
from abc import ABC, abstractmethod
import importlib

from utils.utils import load_config_file

ATTACKER_CONFIG_MAPPING = {
    'WordSubs': "./config/attacker/WordSubs.json",
    'WordDelete': "./config/attacker/WordDelete.json",
    'Dipper': "./config/attacker/Dipper.json",
}

ATTACKER_CLASS_MAPPING = {
    'WordSubs': "utils.attacker.WordSubsAttacker",
    'WordDelete': "utils.attacker.WordDeleteAttacker",
    'Dipper': "utils.attacker.DipperAttacker",
}


@dataclasses.dataclass
class BaseAttackerReturn:
    text_original: str
    text_attacked: str | dict


class BaseAttacker(ABC):
    def __init__(self, config_path: str = None, **kwargs):
        if config_path is None:
            self.config_dict = load_config_file(f"../config/attacker/{self.attacker_method_name}.json")
        else:
            self.config_dict = load_config_file(config_path)

        # Update config with kwargs
        if kwargs:
            self.config_dict.update(kwargs)

    @property
    def attacker_method_name(self) -> str:
        return "Base"

    @abstractmethod
    def attack(self, *args, **kwargs) -> BaseAttackerReturn:
        pass


class AutoAttacker:
    @classmethod
    def load_attacker(cls, attack_method_name: str) -> BaseAttacker:
        if attack_method_name in ATTACKER_CONFIG_MAPPING:
            config_path = ATTACKER_CONFIG_MAPPING[attack_method_name]
        else:
            raise ValueError(f"Invalid attacker name: {attack_method_name}")

        if attack_method_name in ATTACKER_CLASS_MAPPING:
            class_path = ATTACKER_CLASS_MAPPING[attack_method_name]
        else:
            raise ValueError(f"Invalid attacker name: {attack_method_name}")

        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        attacker_class = getattr(module, class_name)
        attacker_instance = attacker_class(config_path)

        return attacker_instance
