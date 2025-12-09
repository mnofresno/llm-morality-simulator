"""Automatic scenario registry for discovering and managing scenarios."""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Any, Optional, Tuple
from scenarios.base import BaseScenario


class ScenarioRegistry:
    """Registry for automatically discovering and managing scenarios."""
    
    _scenarios: Dict[str, Type[BaseScenario]] = {}  # Display name -> Class
    _scenarios_by_name: Dict[str, Type[BaseScenario]] = {}  # Internal name -> Class
    _scenarios_loaded = False
    
    @classmethod
    def discover_scenarios(cls) -> Dict[str, Type[BaseScenario]]:
        """
        Automatically discover all scenario classes in the scenarios directory.
        
        Returns:
            Dictionary mapping scenario display names to scenario classes
        """
        if cls._scenarios_loaded:
            return cls._scenarios
        
        scenarios_dir = Path(__file__).parent
        scenario_modules = {}
        
        # Find all Python files in scenarios directory
        # Include both *_scenario.py and other Python files (like cold_room_relay.py)
        scenario_files = list(scenarios_dir.glob("*_scenario.py"))
        scenario_files.extend(scenarios_dir.glob("*.py"))
        scenario_files = list(set(scenario_files))  # Remove duplicates
        
        # Exclude special files
        excluded = {"__init__.py", "base.py", "registry.py"}
        
        for file_path in scenario_files:
            if file_path.name in excluded:
                continue
                
            module_name = file_path.stem
            try:
                # Import the module
                module = importlib.import_module(f"scenarios.{module_name}")
                
                # Find all classes that inherit from BaseScenario
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseScenario) and 
                        obj is not BaseScenario and
                        obj.__module__ == module.__name__):
                        
                        # Get display name from metadata or class name
                        display_name = cls._get_display_name(obj)
                        scenario_modules[display_name] = obj
                        
                        # Also index by internal name (from metadata)
                        try:
                            # Create a temporary instance with default parameters to get metadata
                            sig = inspect.signature(obj.__init__)
                            default_params = {}
                            for param_name, param in sig.parameters.items():
                                if param_name != 'self' and param.default != inspect.Parameter.empty:
                                    default_params[param_name] = param.default
                            temp_instance = obj(**default_params)
                            internal_name = temp_instance.name
                            cls._scenarios_by_name[internal_name] = obj
                        except Exception:
                            # If we can't create instance, use display name as fallback
                            pass
                        
            except Exception as e:
                print(f"Warning: Could not load scenario from {module_name}: {e}")
                continue
        
        cls._scenarios = scenario_modules
        cls._scenarios_loaded = True
        return cls._scenarios
    
    @classmethod
    def _get_display_name(cls, scenario_class: Type[BaseScenario]) -> str:
        """
        Get display name for a scenario class from metadata or class name.
        
        Args:
            scenario_class: The scenario class
            
        Returns:
            Human-readable display name
        """
        # Try to get title from metadata first
        try:
            sig = inspect.signature(scenario_class.__init__)
            default_params = {}
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param.default != inspect.Parameter.empty:
                    default_params[param_name] = param.default
            temp_instance = scenario_class(**default_params)
            metadata = temp_instance.metadata()
            # Check for title in metadata
            if 'title' in metadata:
                return metadata['title']
            if 'display_name' in metadata:
                return metadata['display_name']
        except Exception:
            pass
        
        # Fallback: Convert class name like "ColdRoomRelayScenario" to "Cold Room Relay"
        class_name = scenario_class.__name__
        if class_name.endswith("Scenario"):
            class_name = class_name[:-8]  # Remove "Scenario"
        
        # Convert CamelCase to Title Case with spaces
        import re
        display_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name)
        return display_name
    
    @classmethod
    def get_scenario_class(cls, identifier: str) -> Optional[Type[BaseScenario]]:
        """
        Get scenario class by display name or internal name.
        
        Args:
            identifier: Display name or internal name (name field) of the scenario
            
        Returns:
            Scenario class or None if not found
        """
        scenarios = cls.discover_scenarios()
        # Try display name first
        if identifier in scenarios:
            return scenarios[identifier]
        # Try internal name
        if identifier in cls._scenarios_by_name:
            return cls._scenarios_by_name[identifier]
        return None
    
    @classmethod
    def get_all_scenarios(cls) -> Dict[str, Type[BaseScenario]]:
        """
        Get all discovered scenarios.
        
        Returns:
            Dictionary of display names to scenario classes
        """
        return cls.discover_scenarios()
    
    @classmethod
    def get_scenario_parameters(cls, scenario_class: Type[BaseScenario]) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter information for a scenario class.
        
        Args:
            scenario_class: The scenario class
            
        Returns:
            Dictionary mapping parameter names to their metadata:
            {
                'param_name': {
                    'type': type,
                    'default': default_value,
                    'annotation': annotation,
                    'description': description_from_docstring
                }
            }
        """
        sig = inspect.signature(scenario_class.__init__)
        parameters = {}
        docstring = inspect.getdoc(scenario_class.__init__) or ""
        
        # Parse docstring for parameter descriptions
        param_docs = {}
        if docstring:
            for line in docstring.split('\n'):
                if 'Args:' in line:
                    continue
                if ':' in line and line.strip().startswith(('Args:', 'param', 'Args')):
                    # Extract parameter description
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        param_name = parts[0].strip().replace('*', '').strip()
                        description = parts[1].strip()
                        param_docs[param_name] = description
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_info = {
                'type': param.annotation if param.annotation != inspect.Parameter.empty else type(None),
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None,
                'description': param_docs.get(param_name, '')
            }
            parameters[param_name] = param_info
        
        return parameters
    
    @classmethod
    def create_scenario_instance(cls, identifier: str, **kwargs) -> Optional[BaseScenario]:
        """
        Create an instance of a scenario with given parameters.
        
        Args:
            identifier: Display name or internal name of the scenario
            **kwargs: Parameters to pass to scenario __init__
            
        Returns:
            Scenario instance or None if not found
        """
        scenario_class = cls.get_scenario_class(identifier)
        if not scenario_class:
            return None
        
        try:
            # Filter kwargs to only include valid parameters
            sig = inspect.signature(scenario_class.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            
            return scenario_class(**filtered_kwargs)
        except Exception as e:
            print(f"Error creating scenario instance: {e}")
            return None
    
    @classmethod
    def get_scenario_metadata(cls, identifier: str) -> Dict[str, Any]:
        """
        Get metadata for a scenario by display name or internal name.
        
        Args:
            identifier: Display name or internal name of the scenario
            
        Returns:
            Dictionary with scenario metadata
        """
        scenario_class = cls.get_scenario_class(identifier)
        if not scenario_class:
            return {}
        
        # Create a default instance to get metadata
        try:
            # Try with default parameters
            sig = inspect.signature(scenario_class.__init__)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param.default != inspect.Parameter.empty:
                    params[param_name] = param.default
            
            instance = scenario_class(**params)
            return instance.metadata()
        except Exception as e:
            print(f"Warning: Could not create instance for metadata: {e}")
            return {
                'name': scenario_class.__name__,
                'class': scenario_class.__name__
            }
    
    @classmethod
    def list_scenario_names(cls) -> List[str]:
        """
        List all available scenario internal names.
        
        Returns:
            List of internal scenario names
        """
        cls.discover_scenarios()
        return list(cls._scenarios_by_name.keys())

