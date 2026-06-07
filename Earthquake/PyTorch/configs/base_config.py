"""
Base configuration class for all config dataclasses.

Provides common serialization methods (to_dict, from_dict, save, load)
to reduce code duplication across config classes.

Usage:
    from configs.base_config import BaseConfig

    @dataclass
    class MyConfig(BaseConfig):
        param1: int = 10
        param2: str = 'default'

    # MyConfig has to_dict(), from_dict(), save(), load() methods
    config = MyConfig(param1=20)
    config.save('config.json')
    loaded = MyConfig.load('config.json')
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Type, TypeVar

# Type variable for class methods that return the same type
T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base class for dataclass configs with JSON serialization helpers."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the config.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create configuration from dictionary.

        Ignores unknown keys that are not fields of this config class.

        Args:
            config_dict: Dictionary with configuration parameters.

        Returns:
            Config instance with values from the dictionary.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_dict)

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Creates parent directories if they don't exist.

        Args:
            path: Output JSON file path.
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """
        Load configuration from JSON file.

        Args:
            path: Input JSON file path.

        Returns:
            Config instance loaded from file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """
        String representation showing all fields.

        Override in subclasses for custom formatting.
        """
        class_name = self.__class__.__name__
        fields = self.to_dict()
        field_strs = [f"  {k}={v!r}" for k, v in fields.items()]
        return f"{class_name}(\n" + ",\n".join(field_strs) + "\n)"

    def copy(self: T, **updates) -> T:
        """
        Create a copy of the config with optional field updates.

        Args:
            **updates: Field values to override in the copy.

        Returns:
            New config instance with updated values.

        Example:
            config = MyConfig(param1=10, param2='a')
            new_config = config.copy(param1=20)  # param2 stays 'a'
        """
        current = self.to_dict()
        current.update(updates)
        return self.__class__.from_dict(current)
