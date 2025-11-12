import os
import yaml
import logging
from typing import Dict, Any


logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Resolve to Piepeline/config.yaml
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')


def load_config() -> Dict[str, Any]:
	"""
	Load the YAML configuration safely. Returns empty dict on error.
	"""
	try:
		with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
			config = yaml.safe_load(f) or {}
		return config
	except Exception as e:
		logger.error(f'Error loading configuration: {e}')
		return {}


def get_data() -> Dict[str, Any]:
	config = load_config()
	return config.get('data', {}) or {}


def get_ingestion() -> Dict[str, Any]:
	config = load_config()
	return config.get('ingestion', {}) or {}


def get_missing_values() -> Dict[str, Any]:
	config = load_config()
	return config.get('missing_values', {}) or {}


def get_outliers() -> Dict[str, Any]:
	config = load_config()
	return config.get('outliers', {}) or {}


def get_feature_engineering() -> Dict[str, Any]:
	config = load_config()
	return config.get('feature_engineering', {}) or {}


def get_preprocessing(task: str) -> Dict[str, Any]:
	"""
	Get preprocessing section for a task, e.g. 'regression' or 'classification'.
	"""
	config = load_config()
	return (config.get('preprocessing', {}) or {}).get(task, {}) or {}


def get_splitting(task: str) -> Dict[str, Any]:
	"""
	Get splitting section for a task, e.g. 'regression' or 'classification'.
	"""
	config = load_config()
	return (config.get('splitting', {}) or {}).get(task, {}) or {}

