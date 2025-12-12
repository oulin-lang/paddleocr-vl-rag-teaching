import json 
import os 
import threading 
from typing import Dict, Any 
 
class RuntimeConfig: 
    _instance = None 
    _lock = threading.Lock() 
    _config_file = "runtime_config.json" 
 
    # Default values 
    _defaults = { 
        "enable_mcp_access": True 
    } 
 
    def __new__(cls): 
        if cls._instance is None: 
            with cls._lock: 
                if cls._instance is None: 
                    cls._instance = super(RuntimeConfig, cls).__new__(cls) 
                    cls._instance._load_config() 
        return cls._instance 
 
    def _load_config(self): 
        """Load configuration from file, or create with defaults if not exists.""" 
        self._config = self._defaults.copy() 
        if os.path.exists(self._config_file): 
            try: 
                with open(self._config_file, 'r', encoding='utf-8') as f: 
                    saved_config = json.load(f) 
                    self._config.update(saved_config) 
            except Exception as e: 
                print(f"Error loading runtime config: {e}") 
        else: 
            self._save_config() 
 
    def _save_config(self): 
        """Save current configuration to file.""" 
        try: 
            with open(self._config_file, 'w', encoding='utf-8') as f: 
                json.dump(self._config, f, indent=4, ensure_ascii=False) 
        except Exception as e: 
            print(f"Error saving runtime config: {e}") 
 
    def get(self, key: str, default: Any = None) -> Any: 
        """Get a configuration value.""" 
        return self._config.get(key, default) 
 
    def set(self, key: str, value: Any): 
        """Set a configuration value and persist it.""" 
        with self._lock: 
            self._config[key] = value 
            self._save_config() 
            # Log the change (simple print for now, can be expanded to logging module) 
            print(f"[RuntimeConfig] Set {key} = {value}") 
 
    @property 
    def enable_mcp_access(self) -> bool: 
        return self.get("enable_mcp_access", True) 
 
    @enable_mcp_access.setter 
    def enable_mcp_access(self, value: bool): 
        self.set("enable_mcp_access", bool(value)) 
 
# Global instance 
runtime_config = RuntimeConfig() 