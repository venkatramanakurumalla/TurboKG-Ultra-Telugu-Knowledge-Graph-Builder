#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurboKG ULTRA v7.3 - Optimized Large File Processor & Enhanced Performance
Production-grade Telugu Knowledge Graph Builder with Robust File Processing
Updates:
- Fixed cache_key error in LargeFileProcessor
- Enhanced memory management for large files
- Improved error handling and checkpointing
- Optimized parallel processing with better backpressure
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import argparse
import logging
import re
import gc
import tempfile
import shutil
import atexit
import string
import threading
import fcntl
import hashlib
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set, Callable, Union, Protocol
from collections import defaultdict, Counter
from pathlib import Path
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import lru_cache
import numpy as np
from enum import Enum

# ==================== IMPORT ENHANCED LINGUISTIC DATA ====================
try:
    from Tgm import (
        _TELUGU_SUFFIX_DATA as ENHANCED_SUFFIX_DATA,
        _SUFFIX_LOOKUP as ENHANCED_SUFFIX_LOOKUP, 
        _SORTED_SUFFIXES as ENHANCED_SORTED_SUFFIXES,
        _PLACE_OVERRIDE as ENHANCED_PLACE_OVERRIDE,
        _PERSON_OVERRIDE as ENHANCED_PERSON_OVERRIDE,
        _ORG_OVERRIDE as ENHANCED_ORG_OVERRIDE,
        _CONJUNCTIONS as ENHANCED_CONJUNCTIONS,
        _VERB_ROOTS as ENHANCED_VERB_ROOTS,
        _KNOWN_STEMS as ENHANCED_KNOWN_STEMS,
        _TEMPORAL_WORDS, _ORGANIZATION_WORDS, _PERSON_WORDS, _PLACE_WORDS,
        _ARTIFACT_WORDS, _ABSTRACT_WORDS, _NATURE_WORDS,
        _PRONOUNS_EXPANDED as ENHANCED_PRONOUNS_EXPANDED,
        _COMMON_EXCEPTIONS as ENHANCED_COMMON_EXCEPTIONS
        
    )
    ENHANCED_DATA_LOADED = True
except ImportError as e:
    # print(f"⚠️ Could not load enhanced linguistic data: {e}")
    # print("ℹ️ Using built-in data only")
    ENHANCED_DATA_LOADED = False

# ==================== OPTIONAL IMPORTS ====================
_HAS_FAISS = False
_HAS_STMODEL = False
_HAS_PROM = False
_HAS_NEO4J = False
_HAS_PSUTIL = False
_HAS_YAML = False
_HAS_STANZA = True

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_STMODEL = True
except ImportError:
    _HAS_STMODEL = False

try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False

try:
    from neo4j import GraphDatabase
    _HAS_NEO4J = True
except ImportError:
    _HAS_NEO4J = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import stanza
    _HAS_STANZA = True
except ImportError:
    _HAS_STANZA = False

# ==================== ENHANCED LINGUISTIC DATA SETUP ====================
if ENHANCED_DATA_LOADED:
    # Override with enhanced data
    _TELUGU_SUFFIX_DATA = ENHANCED_SUFFIX_DATA
    _SUFFIX_LOOKUP = ENHANCED_SUFFIX_LOOKUP
    _SORTED_SUFFIXES = ENHANCED_SORTED_SUFFIXES
    _PLACE_OVERRIDE = ENHANCED_PLACE_OVERRIDE
    _PERSON_OVERRIDE = ENHANCED_PERSON_OVERRIDE
    _BUILTIN_VERB_ROOTS = ENHANCED_VERB_ROOTS
    _BUILTIN_KNOWN_STEMS = ENHANCED_KNOWN_STEMS
    _TEMPORAL_WORDS = _TEMPORAL_WORDS
    _ORGANIZATION_WORDS = _ORGANIZATION_WORDS
    _PERSON_WORDS = _PERSON_WORDS
    _PLACE_WORDS = _PLACE_WORDS
    _ABSTRACT_WORDS = _ABSTRACT_WORDS
    _ARTIFACT_WORDS = _ARTIFACT_WORDS
    _NATURE_WORDS = _NATURE_WORDS
    _ENHANCED_PRONOUNS_EXPANDED = ENHANCED_PRONOUNS_EXPANDED
    _CONJUNCTIONS = ENHANCED_CONJUNCTIONS
    _COMMON_EXCEPTIONS = ENHANCED_COMMON_EXCEPTIONS
    

else:
    # Fallback to original built-in data
    _BUILTIN_VERB_ROOTS = {
        "ఉండు", "రా", "పో", "తిను", "తాగు", "చెప్పు", "చూడు", "విను", "ఎఱుగు",
        "నడచు", "ఓడు", "గెల్చు", "వ్రాయు", "చదువు", "నిల్చు", "కూర్చు", "పడు",
        "లేచు", "ఇచ్చు", "తీసు", "పెట్టు", "తెరచు", "మూయు", "కొను", "వేయు",
        "తెలుసు", "అర్థము", "ఆగు", "మారు", "మాట్లాడు", "ఆడు", "పాడు", "అడుగు",
        "చేయు", "కలుగు", "తోచు", "కనిపెట్టు", "తెచ్చు", "పుచ్చు", "వద్దు", "అయి"
    }

    _BUILTIN_KNOWN_STEMS = {
        "తిన్నాడు": "తిను",
        "తాగాడు": "తాగు", 
        "చూశాడు": "చూడు",
        "చెప్పాడు": "చెప్పు",
        "వ్రాశాడు": "వ్రాయు",
        "చదివాడు": "చదువు",
        "నడిచాడు": "నడచు",
        "పడ్డాడు": "పడు",
        "లేచాడు": "లేచు",
        "ఇచ్చాడు": "ఇచ్చు",
        "తీసాడు": "తీసు",
        "పెట్టాడు": "పెట్టు",
        "మాట్లాడాడు": "మాట్లాడు",
        "ఆడాడు": "ఆడు",
        "పాడాడు": "పాడు",
        "చేశాడు": "చేయు",
        "అయ్యాడు": "అయి"
    }

    _PLACE_OVERRIDE = {
        "హైదరాబాద్": "place_city",
        "విజయవాడ": "place_city", 
        "విశాఖపట్నం": "place_city",
        "తిరుపతి": "place_temple",
        "వారంగల్": "place_city",
        "గుంటూరు": "place_city",
        "నెల్లూరు": "place_city",
        "కడప": "place_city",
        "అనంతపురం": "place_city",
        "కర్నూలు": "place_city"
    }

    _PERSON_OVERRIDE = {
        "రాముడు": "person",
        "సీత": "person",
        "కృష్ణుడు": "person",
        "శివుడు": "person",
        "విష్ణువు": "person",
        "లక్ష్మి": "person",
        "పార్వతి": "person",
        "బాలయ్య": "person",
        "వెంకటేశ్వరుడు": "person",
        "హనుమంతుడు": "person"
    }

    _COMMON_EXCEPTIONS = {
        'పుస్తకానికి': ('పుస్తకం', 'కి'),
        'పుస్తకంలో': ('పుస్తకం', 'లో'),
        'ఇంట్లో': ('ఇల్లు', 'లో'),
        'ఇంటికి': ('ఇల్లు', 'కి'),
        'పిల్లలు': ('పిల్ల', 'లు'),
    }

    _TELUGU_SUFFIX_DATA = {
      "nominative": {
        "డు": {"gender": "masculine", "number": "singular"},
        "ము": {"gender": "neuter", "number": "singular"},
        "వు": {"gender": "neuter", "number": "singular"},
        "లు": {"number": "plural"},
        "వారు": {"honorific": True, "number": "plural"},
        "గారు": {"honorific": True},
        "మండి": {"collective": True}
      },
      "accusative": {
        "ని": {"case": "accusative"},
        "ను": {"case": "accusative"},
        "నున్": {"case": "accusative"},
        "నిన్": {"case": "accusative"},
        "లన్": {"number": "plural"},
        "గురించి": {"meaning": "regarding"},
        "కూర్చి": {"meaning": "about/concerning"},
        "పైన": {"meaning": "about"},
        "మీద": {"meaning": "on/about"}
      },
      "instrumental": {
        "తో": {"meaning": "with/by"},
        "చేత": {"meaning": "by/through"},
        "చేతన": {"meaning": "by means of"},
        "ద్వారా": {"meaning": "through"},
        "బట్టి": {"meaning": "according to"},
        "వల్ల": {"meaning": "because of"}
      },
      "dative": {
        "కు": {"meaning": "to/for"},
        "కి": {"meaning": "to/for"},
        "కై": {"meaning": "for"},
        "కొరకు": {"meaning": "for"},
        "కోసం": {"meaning": "for/sake of"},
        "కొఱకున్": {"archaic": True},
        "కోఱకు": {"archaic": True}
      },
      "ablative": {
        "నుంచి": {"meaning": "from"},
        "నుండి": {"meaning": "from"},
        "వలన": {"meaning": "due to/from"},
        "వల్ల": {"meaning": "because of"},
        "కంటే": {"meaning": "than"},
        "కంటె": {"meaning": "than"},
        "పట్టి": {"meaning": "regarding/from"}
      },
      "genitive": {
        "యొక్క": {"meaning": "of"},
        "యొక్కది": {"meaning": "belonging to"},
        "ది": {"possessive_suffix": True}
      },
      "locative": {
        "లో": {"meaning": "in"},
        "లోపల": {"meaning": "inside"},
        "లోన్": {"archaic": True},
        "పై": {"meaning": "on/above"},
        "క్రింద": {"meaning": "below"},
        "వద్ద": {"meaning": "at/near"},
        "దగ్గర": {"meaning": "near"}
      },
      "vocative": {
        "ఓ": {"meaning": "address"},
        "ఓయీ": {"meaning": "address"},
        "ఓరీ": {"meaning": "address"},
        "అయ్యా": {"meaning": "sir"},
        "అమ్మా": {"meaning": "mother"},
        "బాబూ": {"meaning": "dear"}
      },
      "verbal": {
        "తున్నాడు": {"tense": "present_continuous", "person": 3},
        "తున్నాను": {"tense": "present_continuous", "person": 1},
        "తున్నావు": {"tense": "present_continuous", "person": 2},
        "తున్నారు": {"tense": "present_continuous", "person": 3},
        "తున్నాయి": {"tense": "present_continuous", "number": "plural"},
        "తే": {"function": "conditional"},
        "తూ": {"function": "while"},
        "క": {"function": "negative_participle"},
        "డం": {"function": "nominalize"},
        "డం వల్ల": {"function": "causal_clause"},
        "ఇ": {"tense": "past"},
        "తా": {"tense": "future/habitual"},
        "వటానికి": {"function": "purpose_infinitive"},
        "వడానికి": {"function": "purpose_infinitive"},
        "కూడదని": {"function": "prohibitive"},
        "లేక": {"function": "conjunctive_negative"},
        "మంటే": {"function": "quotative"},
        "సేది": {"tense": "future_potential"}
      },
      "nominal": {
        "తనం": {"function": "abstract", "meaning": "-ness"},
        "తనము": {"function": "abstract"},
        "రికం": {"function": "abstract"},
        "పు": {"function": "adjectival"},
        "గా": {"function": "adverbial"},
        "గానూ": {"function": "emphatic_adverbial"},
        "గా ఉండి": {"function": "state_marker"},
        "గానీ": {"function": "disjunctive"},
        "గానే": {"function": "immediate"},
        "గానేనూ": {"function": "emphatic_immediate"}
      },
      "place": {
        "పురం": {"type": "city"},
        "పట్నం": {"type": "city"},
        "వూరు": {"type": "village"},
        "పల్లె": {"type": "village"},
        "గూడెం": {"type": "hamlet"},
        "వాడ": {"type": "settlement"},
        "పాళెం": {"type": "settlement"},
        "చెర్ల": {"type": "village"},
        "పాడు": {"type": "settlement"},
        "పూడి": {"type": "settlement"}
      },
      "temporal": {
        "తర్వాత": {"meaning": "after"},
        "ముందు": {"meaning": "before"},
        "తరువాత": {"meaning": "after"},
        "లోపు": {"meaning": "within"},
        "వరకు": {"meaning": "until"},
        "నాటికి": {"meaning": "by (time)"}
      },
      "quantitative": {
        "ంత": {"meaning": "as much as"},
        "ంతటి": {"meaning": "of that size"},
        "లాంటిది": {"meaning": "similar to"},
        "లాంటి": {"meaning": "like/similar"},
        "కొంత": {"meaning": "some"},
        "ఎంత": {"meaning": "how much"},
        "చాలా": {"meaning": "much/many"}
      },
      "archaic": {
        "మంబు": {"function": "nominal"},
        "ఇఁడి": {"meaning": "without"},
        "మాలు": {"meaning": "without"},
        "అమి": {"function": "negative_nom"},
        "ఇండి": {"meaning": "one_without"}
      }
    }

    # Flatten suffixes into a single confidence-weighted lookup dict
    _SUFFIX_LOOKUP: Dict[str, Tuple[str, float, Dict[str, Any]]] = {}
    for category, suffixes in _TELUGU_SUFFIX_DATA.items():
        for suf, meta in suffixes.items():
            conf = 0.95
            if meta.get("archaic"):
                conf = 0.75
            elif category in ("verbal", "nominal"):
                conf = 0.92
            elif category == "place":
                conf = 0.98
            elif category == "vocative":
                conf = 0.90
            _SUFFIX_LOOKUP[suf] = (category, conf, meta)

    _SORTED_SUFFIXES = sorted(_SUFFIX_LOOKUP.keys(), key=lambda x: -len(x))

    # Initialize other word lists as empty if enhanced data not available
    _TEMPORAL_WORDS = set()
    _ORGANIZATION_WORDS = set()
    _PERSON_WORDS = set()
    _PLACE_WORDS = set()
    _ABSTRACT_WORDS = set()
    _ARTIFACT_WORDS = set()
    _NATURE_WORDS = set()

# ==================== CUSTOM JSON ENCODER ====================
class TurboKGJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for TurboKG objects"""
    
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)

# ==================== ENHANCED CONFIGURATION ====================
@dataclass
class TurboKGConfig:
    """Enhanced configuration with validation"""
    min_confidence: float = 0.7
    context_window: int = 5
    enable_sandhi: bool = True
    sandhi_mode: str = "adaptive"
    enable_parallel: bool = True
    num_workers: Optional[int] = None
    max_cache_size: int = 10000
    batch_size: int = 100
    relation_min_frequency: int = 1
    enable_performance_monitoring: bool = True
    max_document_size_mb: int = 10
    shards: int = 1
    faiss_enabled: bool = False
    prom_port: Optional[int] = None
    use_neo4j: bool = False
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_pass: Optional[str] = None
    neo4j_database: str = "neo4j"
    # Enhanced features
    verb_roots_path: Optional[str] = None
    stems_path: Optional[str] = None
    enable_compound_splitting: bool = True
    enable_verb_morphology: bool = True
    enable_embeddings: bool = False
    embedding_model_name: Optional[str] = "all-MiniLM-L6-v2"
    use_new_sandhi_engine: bool = False
    enable_syntactic_analysis: bool = True
    enable_semantic_roles: bool = True
    max_relation_distance: int = 5
    min_cooccurrence_frequency: int = 3
    relation_confidence_threshold: float = 0.6
    # Stanza Configuration
    use_stanza: bool = False
    stanza_use_gpu: bool = True
    stanza_model_dir: Optional[str] = None
    
    def __post_init__(self):
        """Enhanced validation with file path checks"""
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.context_window < 1:
            raise ValueError("context_window must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_document_size_mb < 1:
            raise ValueError("max_document_size_mb must be >= 1")
        if self.sandhi_mode not in {"adaptive", "strict", "permissive"}:
            raise ValueError("sandhi_mode must be 'adaptive', 'strict', or 'permissive'")
        if self.use_neo4j and not all([self.neo4j_uri, self.neo4j_user, self.neo4j_pass]):
            raise ValueError("Neo4j configuration requires URI, user, and password")
        if self.num_workers is None:
            self.num_workers = max(1, (os.cpu_count() or 4) - 1)
        
        # Validate file paths exist if provided
        if self.verb_roots_path and not Path(self.verb_roots_path).exists():
            logger.warning(f"Verb roots path does not exist: {self.verb_roots_path}")
        if self.stems_path and not Path(self.stems_path).exists():
            logger.warning(f"Stems path does not exist: {self.stems_path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TurboKGConfig':
        """Create config from dictionary"""
        from dataclasses import fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

# ==================== PERFORMANCE MONITORING ====================
class PerformanceMonitor:
    """Real-time performance monitoring with alerts"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.alert_thresholds = {
            'memory_usage_mb': 2000,
            'processing_time_sec': 10.0,
            'error_rate': 0.05,
            'cache_hit_rate': 0.3
        }
        self._lock = threading.RLock()
    
    def track_metric(self, metric_name: str, value: float):
        """Track metric and check for alerts"""
        with self._lock:
            self.metrics_history[metric_name].append(value)
            
            if len(self.metrics_history[metric_name]) > 100:
                self.metrics_history[metric_name].pop(0)
                
            threshold = self.alert_thresholds.get(metric_name)
            if threshold and value > threshold:
                logger.warning(f"⚠️ Performance alert: {metric_name} = {value:.1f} (threshold: {threshold})")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        with self._lock:
            for metric, values in self.metrics_history.items():
                if values:
                    summary[metric] = {
                        'current': values[-1],
                        'average': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values),
                        'trend': 'up' if len(values) > 1 and values[-1] > values[-2] else 'down'
                    }
        return summary

# ==================== OPTIMIZED LOGGING ====================
class TurboKGFormatter(logging.Formatter):
    """Custom formatter with colors and performance metrics"""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[41m',
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        message = super().format(record)
        return f"{log_color}{message}{self.COLORS['RESET']}"

def setup_logger():
    """Setup optimized logger"""
    logger = logging.getLogger("TurboKG")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(TurboKGFormatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    logger.addHandler(handler)
    return logger

logger = setup_logger()

# ==================== CONSTANTS & CONFIG ====================
_TELUGU_BLOCK_START = 0x0C00
_TELUGU_BLOCK_END = 0x0C7F
_TELUGU_CHARS = r"\u0C00-\u0C7F"
_PUNCTUATION = string.punctuation + "‘'\"'“”–—…«»\u2013\u2014\u2018\u2019\u201c\u201d\u00a0"
_TOKEN_RE = re.compile(rf"([{_TELUGU_CHARS}]+|\d+|[{re.escape(_PUNCTUATION)}]|\S)", re.UNICODE)
_SENTENCE_ENDINGS = re.compile(r'[.!?।॥\u0964]+')
_SYLLABLE_RE = re.compile(rf"(?:[{_TELUGU_CHARS}][\u0C3E-\u0C56\u0C81\u0C82\u0C83]*)+", re.UNICODE)
_DEFAULT_CONTEXT_WINDOW = 3
_DEFAULT_BATCH_SIZE = 10000
_DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)
_MAX_ENTITY_LENGTH = 100
_MIN_COOCCURRENCE_FREQ = 2
_SCHEMA_VERSION = "2.1"

# Precompiled sanitize regex
_SANITIZE_RE = re.compile(rf'[^\w\u0C00-\u0C7F\-_\.]')

# ==================== ENHANCED LEXICON MANAGER ====================
class LexiconManager:
    """Enhanced lexicon manager with verb roots and known stems"""
    
    def __init__(self, verb_roots_path: Optional[Path] = None, stems_path: Optional[Path] = None):
        self.verb_roots_path = Path(verb_roots_path) if verb_roots_path else None
        self.stems_path = Path(stems_path) if stems_path else None
        self.verb_roots: Set[str] = set()
        self.known_stems: Dict[str, str] = {}
        self._load()
        
    def _load(self):
        """Load lexicons from files or use built-in data"""
        # Load verb roots
        if self.verb_roots_path and self.verb_roots_path.exists():
            try:
                with open(self.verb_roots_path, 'r', encoding='utf-8') as f:
                    self.verb_roots = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(self.verb_roots)} verb roots from {self.verb_roots_path}")
            except Exception as e:
                logger.warning(f"Failed to load verb roots: {e}, using built-in data")
                self.verb_roots = _BUILTIN_VERB_ROOTS
        else:
            self.verb_roots = _BUILTIN_VERB_ROOTS
            
        # Load known stems
        if self.stems_path and self.stems_path.exists():
            try:
                with open(self.stems_path, 'r', encoding='utf-8') as f:
                    self.known_stems = json.load(f)
                logger.info(f"Loaded {len(self.known_stems)} known stems from {self.stems_path}")
            except Exception as e:
                logger.warning(f"Failed to load known stems: {e}, using built-in data")
                self.known_stems = dict(_BUILTIN_KNOWN_STEMS)
        else:
            self.known_stems = dict(_BUILTIN_KNOWN_STEMS)
            
    def reload(self):
        """Reload lexicons"""
        self._load()
        logger.info("Lexicons reloaded successfully")
        
    def add_verb_root(self, root: str):
        """Add a verb root dynamically"""
        self.verb_roots.add(root)
        
    def add_known_stem(self, surface_form: str, stem: str):
        """Add a known stem mapping dynamically"""
        self.known_stems[surface_form] = stem

# ==================== TRIE-BASED COMPOUND SPLITTER ====================
class TrieNode:
    """Trie node for efficient compound splitting"""
    __slots__ = ('children', 'is_end')
    
    def __init__(self):
        self.children = {}
        self.is_end = False

class CompoundSplitter:
    """Enhanced compound splitter using trie data structure"""
    
    def __init__(self, word_list: Set[str]):
        self.root = TrieNode()
        self._build_trie(word_list)
        
    def _build_trie(self, word_list: Set[str]):
        """Build trie from word list"""
        for word in word_list:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            
    def split(self, word: str) -> List[List[str]]:
        """Split compound word using trie"""
        n = len(word)
        if n == 0:
            return [[]]
            
        # Dynamic programming approach
        dp = [[] for _ in range(n + 1)]
        dp[0] = [[]]
        
        for i in range(1, n + 1):
            node = self.root
            for j in range(i - 1, -1, -1):
                char = word[j]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.is_end and dp[j]:
                    for prev_split in dp[j]:
                        dp[i].append(prev_split + [word[j:i]])
                        
        return dp[n] if dp[n] else [[word]]

# ==================== ENHANCED VERB MORPHOLOGY ====================
class TeluguVerbMorphology:
    """Enhanced verb morphology analyzer"""
    
    TENSE_ASPECT_MARKERS = {
        "past": {
            "ాడు", "ారు", "ాను", "ింది", "చాడు", "శాడు", "యాడు", "కాడు", 
            "గాడు", "టాడు", "డాడు", "దాడు", "బాడు", "మాడు", "నాడు"
        },
        "present_continuous": {
            "తున్నాడు", "తున్నారు", "తున్నాను", "స్తున్నాడు", "స్తున్నారు",
            "స్తున్నాను", "కుంటున్నాడు", "కుంటున్నారు", "గుతున్నాడు"
        },
        "future": {
            "తాడు", "తారు", "తాను", "స్తాడు", "స్తారు", "స్తాను",
            "కుంటాడు", "కుంటారు", "గుతాడు"
        },
        "perfective": {
            "ించాడు", "ించారు", "పాడు", "య్యాడు", "క్కాడు", "గ్గాడు"
        },
        "habitual": {
            "తుంటాడు", "తుంటారు", "స్తుంటాడు", "కుంటుంటాడు"
        }
    }
    
    @classmethod
    def detect_tense_aspect(cls, word: str) -> Optional[str]:
        """Detect tense and aspect from verb form"""
        best_match = (None, 0)
        for tense, markers in cls.TENSE_ASPECT_MARKERS.items():
            for marker in markers:
                if word.endswith(marker) and len(marker) > best_match[1]:
                    best_match = (tense, len(marker))
        return best_match[0]
        
    @classmethod
    def extract_verb_root(cls, word: str, known_roots: Set[str]) -> Optional[str]:
        """Extract verb root from conjugated form"""
        best_root = None
        for root in known_roots:
            variants = {root}
            if root.endswith('ు'):
                variants.add(root[:-1])
            if len(root) > 2:
                variants.add(root[:-1])
            for v in variants:
                if word.startswith(v) and (best_root is None or len(v) > len(best_root)):
                    best_root = root
                    break
        if best_root:
            return best_root
                
        longest_marker = ''
        for markers in cls.TENSE_ASPECT_MARKERS.values():
            for marker in markers:
                if word.endswith(marker) and len(marker) > len(longest_marker):
                    longest_marker = marker
        if longest_marker:
            stem = word[:-len(longest_marker)]
            if len(stem) >= 2:
                return stem
        return None

# ==================== ENHANCED SANDHI ENGINE ====================
class EnhancedTeluguSandhiEngine:
    """Optimized Sandhi engine with statistical learning and rule caching"""
    
    def __init__(self, mode: str = "adaptive"):
        if mode not in {"adaptive", "strict", "permissive"}:
            raise ValueError("mode must be 'adaptive', 'strict', or 'permissive'")
            
        self.mode = mode
        self.rules: Dict[str, Tuple[Callable, str, int]] = {}
        self.rule_stats = defaultdict(int)
        self.vowels = 'అఆఇఈఉఊఋఌఎఏఐఒఓఔ'
        self.consonants = 'కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహళ'
        self.parushams = {'క': 'గ', 'చ': 'స', 'ట': 'డ', 'త': 'ద', 'ప': 'వ'}
        self._rule_cache = {}
        self._cache_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self._cache_hits = 0
        self._total_requests = 0
        self.performance_monitor = PerformanceMonitor()
        
        self._register_enhanced_rules()

    def _register_enhanced_rules(self):
        """Register comprehensive Sandhi rules with priorities"""
        rules = [
            ("ఉత్వ సంధి", self._apply_utva_sandhi, 10),
            ("ఇత్వ సంధి", self._apply_itva_sandhi, 20),
            ("అత్వ సంధి", self._apply_atva_sandhi, 30),
            ("యడాగమ సంధి", self._apply_yadagama_sandhi, 40),
            ("గసడదవాదేశ సంధి", self._apply_gasadadava_sandhi, 50),
            ("ఆమ్రేడిత సంధి", self._apply_amredita_sandhi, 5),
            ("త్రిక సంధి", self._apply_trika_sandhi, 15),
            ("విభక్తి సంధి", self._apply_vibhakti_sandhi, 25),
            ("సవర్ణదీర్ఘ సంధి", self._apply_savarna_sandhi, 35),
            ("అనునాసిక సంధి", self._apply_anuswara_sandhi, 45),
        ]
        
        for name, func, priority in rules:
            self.add_rule(name, func, f"Enhanced {name}", priority)

    def add_rule(self, name: str, func: Callable, description: str, priority: int = 50):
        """Add a new sandhi rule"""
        self.rules[name] = (func, description, priority)

    @lru_cache(maxsize=20000)
    def join_words_cached(self, word1: str, word2: str) -> List[str]:
        """Cached word joining for performance"""
        if not isinstance(word1, str) or not isinstance(word2, str):
            raise TypeError("word1 and word2 must be strings")
        if not word1.strip() or not word2.strip():
            return [f"{word1} {word2}"]
        return self._join_words_uncached(word1, word2)

    def _join_words_uncached(self, word1: str, word2: str) -> List[str]:
        """Actual word joining logic with enhanced analysis"""
        pūrva, para = word1.strip(), word2.strip()
        if not pūrva or not para:
            return [f"{pūrva} {para}"]

        cache_key = (pūrva, para)
        with self._cache_lock:
            if cache_key in self._rule_cache:
                self._cache_hits += 1
                return self._rule_cache[cache_key]

        self._total_requests += 1
        start_time = time.time()
        
        sorted_rules = sorted(self.rules.items(), key=lambda x: x[1][2])
        results = set()
        
        for name, (func, desc, prio) in sorted_rules:
            try:
                forms = func(pūrva, para)
                if forms:
                    results.update(forms)
                    with self._stats_lock:
                        self.rule_stats[name] += 1
                    if self.mode == "strict":
                        break
            except Exception as e:
                logger.debug(f"Rule {name} failed on ({pūrva}, {para}): {e}")
                continue
        final_results = list(results) if results else []

        processing_time = time.time() - start_time
        self.performance_monitor.track_metric('sandhi_processing_time', processing_time)

        if not final_results:
            final_results.append(f"{pūrva} {para}")
            if para and para[0] in self.vowels:
                final_results.append(f"{pūrva}{para}")
                try:
                    final_results.append(f"{pūrva}ా")
                except Exception:
                    pass

        with self._cache_lock:
            if len(self._rule_cache) < 50000:
                self._rule_cache[cache_key] = final_results

        return final_results

    def _apply_utva_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply ఉత్వ సంధి rules (Corrected: drop 'u')"""
        if not pūrva.endswith('ు') or not para:
            return []
        if para[0] in self.vowels:
            return [f"{pūrva[:-1]}{para}"] 
        return []

    def _apply_itva_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply ఇత్వ సంధి rules (Corrected: drop 'i' in standard cases)"""
        if not pūrva.endswith('ి') or not para:
            return []
        if para[0] in self.vowels:
            return [f"{pūrva[:-1]}{para}"]
        return []

    def _apply_atva_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply అత్వ సంధి rules"""
        # Note: requires deeper syllabification to detect implicit 'a'.
        # Assuming minimal implementation here.
        return []

    def _apply_yadagama_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply యడాగమ సంధి rules"""
        if not pūrva or not para:
            return []
        if pūrva[-1] in self.vowels and para[0] in self.vowels:
            return [f"{pūrva}య{para}"]
        return []

    def _apply_gasadadava_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply గసడదవాదేశ సంధి rules"""
        if not pūrva or not para:
            return []
        if pūrva[-1] in 'కచటతప' and para[0] in self.vowels:
            soft = self.parushams.get(pūrva[-1])
            if soft:
                return [f"{pūrva[:-1]}{soft}{para}"]
        return []

    def _apply_amredita_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply ఆమ్రేడిత సంధి rules"""
        if not pūrva or not para:
            return []
        if pūrva == para:
            return [f"{pūrva}మ{para}"]
        return []

    def _apply_trika_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply త్రిక సంధి rules"""
        if not pūrva or not para:
            return []
        if pūrva[-1] in 'ఇఈ' and para[0] in self.vowels:
            return [f"{pūrva[:-1]}య{para}"]
        return []

    def _apply_vibhakti_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply విభక్తి సంధి rules"""
        if not pūrva or not para:
            return []
        vibhaktis = ['ను', 'కి', 'కు', 'లో', 'తో', 'చే', 'వల్ల', 'కోసం']
        if any(pūrva.endswith(v) for v in vibhaktis):
            if para[0] in self.vowels:
                return [f"{pūrva[:-1]}వ{para}"]
        return []

    def _apply_savarna_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply సవర్ణదీర్ఘ సంధి rules"""
        if not pūrva or not para:
            return []
        if pūrva[-1] in self.vowels and para[0] in self.vowels:
            if pūrva[-1] == para[0]:
                return [f"{pūrva[:-1]}{para}"]
        return []

    def _apply_anuswara_sandhi(self, pūrva: str, para: str) -> List[str]:
        """Apply అనునాసిక సంధి rules"""
        if not pūrva or not para:
            return []
        if pūrva.endswith('ం') and para[0] in self.consonants:
            return [f"{pūrva[:-1]}న{para}"]
        return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            cache_size = len(self._rule_cache)
        return {
            'cache_size': cache_size,
            'cache_hits': self._cache_hits,
            'total_requests': self._total_requests,
            'hit_rate': self._cache_hits / max(1, self._total_requests),
            'rule_stats': dict(self.rule_stats)
        }

    def clear_cache(self):
        """Clear the rule cache"""
        with self._cache_lock:
            self._rule_cache.clear()
        self.join_words_cached.cache_clear()

# ==================== STANZA ENTITY EXTRACTOR ====================
class StanzaEntityExtractor:
    """
    Neural Entity Extractor using Stanford Stanza for high-accuracy POS & NER.
    Replaces rule-based logic with pre-trained Telugu models.
    """
    def __init__(self, config: TurboKGConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        
        if not _HAS_STANZA:
            raise ImportError("Stanza is not installed. Run: pip install stanza")
            
        logger.info("🧠 Initializing Stanza Neural Pipeline for Telugu...")
        
        # 1. Prepare arguments dynamically
        stanza_kwargs = {}
        if self.config.stanza_model_dir:
            stanza_kwargs['model_dir'] = self.config.stanza_model_dir
            
        # 2. Define Processor Lists
        # DOWNLOAD list: Exclude 'lemma' (it's identity/logic only) and 'mwt'
        download_processors = 'tokenize,pos,ner'
        
        # PIPELINE list: Include 'lemma' so the identity logic runs
        pipeline_processors = 'tokenize,pos,lemma,ner'
            
        # 3. Download Telugu models (Robustly)
        try:
            # Try downloading with NER first
            stanza.download('te', processors=download_processors, 
                          verbose=False, **stanza_kwargs)
        except Exception as e:
            # If NER model is missing (common for te), fallback to basic models
            error_msg = str(e).lower()
            if "ner" in error_msg or "default" in error_msg:
                logger.warning("⚠️ Stanza NER/Default model not found for Telugu. Falling back to POS-only.")
                download_processors = 'tokenize,pos'
                pipeline_processors = 'tokenize,pos,lemma' # Drop NER from pipeline
                try:
                    stanza.download('te', processors=download_processors, 
                                  verbose=False, **stanza_kwargs)
                except Exception as inner_e:
                    logger.warning(f"Stanza download warning: {inner_e}")
            else:
                logger.warning(f"Stanza download warning: {e}")

        # 4. Initialize pipeline
        try:
            self.nlp = stanza.Pipeline(
                'te', 
                processors=pipeline_processors,
                use_gpu=self.config.stanza_use_gpu,
                verbose=False,
                **stanza_kwargs
            )
        except Exception as e:
            # Final fallback if pipeline creation fails (e.g., NER missing in pipeline)
            logger.warning(f"Pipeline creation failed with NER. Retrying without NER.")
            self.nlp = stanza.Pipeline(
                'te',
                processors='tokenize,pos,lemma',
                use_gpu=self.config.stanza_use_gpu,
                verbose=False,
                **stanza_kwargs
            )

        logger.info("✅ Stanza Pipeline Ready")
        
        # Mapping Stanza UPOS tags to TurboKG entity types
        self.pos_map = {
            'PROPN': 'person',      # Proper Noun
            'NOUN': 'noun',         # Common Noun
            'VERB': 'verb',         # Verb
            'ADJ': 'adjective',     # Adjective
            'ADV': 'adverb',        # Adverb
            'NUM': 'number',        # Number
            'PRON': 'person',       # Pronoun
            'INTJ': 'interjection',
            'ADP': 'postposition'
        }
        self._cache = {} 

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Stanza Neural Pipeline"""
        start_time = time.time()
        entities = []
        
        if not text or not text.strip():
            return []

        # Run Neural Inference
        try:
            doc = self.nlp(text)
        except Exception as e:
            logger.error(f"Stanza processing failed: {e}")
            return []

        global_idx = 0
        
        for sent in doc.sentences:
            for word in sent.words:
                # 1. Base Type Inference from POS
                entity_type = self.pos_map.get(word.upos, 'unknown')
                
                # 2. Refine with Stanza NER (if available and detected)
                if hasattr(word, 'parent') and hasattr(word.parent, 'ner'):
                    ner = word.parent.ner
                    if ner and ner != 'O':
                        if 'PERSON' in ner:
                            entity_type = 'person'
                        elif 'LOC' in ner or 'GPE' in ner:
                            entity_type = 'place'
                        elif 'ORG' in ner:
                            entity_type = 'organization'

                # 3. Construct Verb Analysis using Stanza features
                verb_analysis = {}
                if word.upos == 'VERB':
                    verb_analysis = {
                        'is_verb': True,
                        'verb_root': word.lemma if word.lemma else word.text,
                        'tense_aspect': word.feats if word.feats else "unknown",
                        'confidence': 0.95
                    }

                # 4. Create Entity Object
                if entity_type not in ['unknown', 'postposition', 'interjection', 'symbol', 'punct']:
                    entity = {
                        'text': word.text,
                        'position': global_idx,
                        'confidence': 0.95,
                        'entity_type': entity_type,
                        'morphology': {
                            'lemma': word.lemma if word.lemma else word.text,
                            'upos': word.upos,
                            'feats': word.feats
                        },
                        'sandhi_forms': [],
                        'compound_analysis': [],
                        'verb_analysis': verb_analysis
                    }
                    entities.append(entity)
                
                global_idx += 1

        processing_time = time.time() - start_time
        self.performance_monitor.track_metric('entity_extraction_time', processing_time)
        return entities
        
    def clear_cache(self):
        import gc
        gc.collect()

# ==================== ULTRA ENTITY EXTRACTOR ====================
class UltraEntityExtractor:
    """Ultra entity extractor with balanced enhanced linguistic knowledge"""
    
    def __init__(self, lexicon_manager: LexiconManager, config: TurboKGConfig):
        self.lexicon = lexicon_manager
        self.config = config
        self.sandhi_engine = EnhancedTeluguSandhiEngine(config.sandhi_mode)
        self.verb_morphology = TeluguVerbMorphology()
        self.compound_splitter = None
        self._init_compound_splitter()
        self._cache = {}
        self._cache_lock = threading.RLock()
        self.performance_monitor = PerformanceMonitor()
        self.common_words = self._load_common_words()
        
    def _init_compound_splitter(self):
        """Initialize compound splitter with known words"""
        known_words = set(self.lexicon.known_stems.keys()) | self.lexicon.verb_roots
        known_words.update(_PERSON_OVERRIDE.keys())
        known_words.update(_PLACE_OVERRIDE.keys())
        self.compound_splitter = CompoundSplitter(known_words)
        
    def _load_common_words(self) -> Set[str]:
        """Load common words that shouldn't be entities"""
        return {
            'పని', 'విద్య', 'ఉద్యోగం', 'వ్యాపారం', 'కుటుంబం', 'సమయం',
            'ప్రేమ', 'స్నేహం', 'ఆరోగ్యం', 'ధనం', 'భాష', 'సంస్కృతి',
            'చదువు', 'రాయడం', 'మాట్లాడడం', 'వినడం', 'చూడడం', 'అనుభవం',
            'ఆహారం', 'నీరు', 'గాలి', 'భూమి', 'ఆకాశం', 'సూర్యుడు',
            'తల్లి', 'తండ్రి', 'అక్క', 'చెల్లి', 'సోదరుడు', 'సోదరి',
            'మనిషి', 'ప్రాణి', 'జంతువు', 'పక్షి', 'మృగం', 'పుష్పం'
        }
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities with ultra-enhanced analysis"""
        start_time = time.time()
        entities = []
        tokens = _TOKEN_RE.findall(text)
        
        for i, token in enumerate(tokens):
            if not self._is_telugu_word(token):
                continue
                
            entity = self._analyze_token(token, i, tokens)
            if entity:
                entities.append(entity)
                
        processing_time = time.time() - start_time
        self.performance_monitor.track_metric('entity_extraction_time', processing_time)
        
        return entities
        
    def _analyze_token(self, token: str, position: int, tokens: List[str]) -> Optional[Dict[str, Any]]:
        """Enhanced token analysis with better person/verb distinction"""
        # Check cache first
        cache_key = (token, position)
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
                
        # Exclude common words early
        if self._is_common_word(token):
            return None
            
        # Enhanced analysis pipeline
        analysis = {
            'text': token,
            'position': position,
            'confidence': 0.0,
            'entity_type': 'unknown',
            'morphology': {},
            'sandhi_forms': [],
            'compound_analysis': [],
            'verb_analysis': {}
        }
        
        # 1. Check for known exceptions and proper names FIRST
        if (token in _PERSON_OVERRIDE or token in _PERSON_WORDS) and len(token) > 2:
            if not self._looks_like_verb(token):
                analysis.update({
                    'entity_type': 'person',
                    'confidence': 0.95,
                    'is_proper_name': True
                })
                # Cache and return early for proper names
                with self._cache_lock:
                    if len(self._cache) < self.config.max_cache_size:
                        self._cache[cache_key] = analysis
                return analysis
        
        # 2. Check for known places
        if token in _PLACE_OVERRIDE and len(token) > 2:
            analysis.update({
                'entity_type': _PLACE_OVERRIDE[token],
                'confidence': 0.95
            })
            with self._cache_lock:
                if len(self._cache) < self.config.max_cache_size:
                    self._cache[cache_key] = analysis
            return analysis
            
        # 3. Check for known organizations
        if token in _ORGANIZATION_WORDS and len(token) > 3:
            analysis.update({
                'entity_type': 'organization',
                'confidence': 0.90
            })
            with self._cache_lock:
                if len(self._cache) < self.config.max_cache_size:
                    self._cache[cache_key] = analysis
            return analysis
        
        # 4. Check for other known word categories
        if token in _TEMPORAL_WORDS and len(token) > 2:
            analysis.update({
                'entity_type': 'temporal',
                'confidence': 0.85
            })
            with self._cache_lock:
                if len(self._cache) < self.config.max_cache_size:
                    self._cache[cache_key] = analysis
            return analysis
            
        if token in _ABSTRACT_WORDS and len(token) > 2:
            analysis.update({
                'entity_type': 'abstract',
                'confidence': 0.80
            })
            with self._cache_lock:
                if len(self._cache) < self.config.max_cache_size:
                    self._cache[cache_key] = analysis
            return analysis
            
        if token in _ARTIFACT_WORDS and len(token) > 2:
            analysis.update({
                'entity_type': 'artifact',
                'confidence': 0.85
            })
            with self._cache_lock:
                if len(self._cache) < self.config.max_cache_size:
                    self._cache[cache_key] = analysis
            return analysis
            
        if token in _NATURE_WORDS and len(token) > 2:
            analysis.update({
                'entity_type': 'nature',
                'confidence': 0.85
            })
            with self._cache_lock:
                if len(self._cache) < self.config.max_cache_size:
                    self._cache[cache_key] = analysis
            return analysis
    
        # 5. Continue with normal analysis for other tokens
        return self._continue_analysis(token, position, tokens, analysis)
    
    def _continue_analysis(self, token: str, position: int, tokens: List[str], analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Continue with normal entity analysis"""
        # 1. Check for known exceptions
        if token in _COMMON_EXCEPTIONS:
            analysis.update(self._handle_known_exception(token))
            
        # 2. Suffix analysis
        suffix_analysis = self._analyze_suffixes(token)
        analysis.update(suffix_analysis)
        
        # 3. Compound splitting
        if self.config.enable_compound_splitting:
            compound_analysis = self._analyze_compounds(token)
            analysis['compound_analysis'] = compound_analysis
            
        # 4. Verb morphology analysis
        if self.config.enable_verb_morphology:
            verb_analysis = self._analyze_verb(token)
            analysis['verb_analysis'] = verb_analysis
            
        # 5. Entity type inference
        entity_type = self._infer_entity_type(analysis)
        analysis['entity_type'] = entity_type
        
        # 6. Confidence calculation
        confidence = self._calculate_confidence(analysis)
        analysis['confidence'] = confidence
        
        # Cache the result
        with self._cache_lock:
            if len(self._cache) < self.config.max_cache_size:
                self._cache[(token, position)] = analysis
                
        return analysis if confidence >= self.config.min_confidence else None
    
    def _looks_like_verb(self, text: str) -> bool:
        """More accurate verb detection - exclude proper names"""
        # Common verb endings in Telugu (more specific patterns)
        verb_endings = {
            '్తున్నాడు', '్తున్నారు', '్తున్నాను', '్తున్నావు',  # present continuous
            '్తాడు', '్తారు', '్తాను', '్తావు',              # future
            '్చాడు', '్చారు', '్చాను', '్చావు',              # past
            'ించాడు', 'ించారు', 'ించాను',                   # causative past
            'పడతాడు', 'పడతారు', 'పడతాను',                   # potential
            'వుతున్న', 'వుతాడు', 'వుతారు',                   # becoming
            'అవుతున్న', 'అవుతాడు', 'అవుతారు',               # becoming
            'కుంటున్న', 'కుంటాడు', 'కుంటారు',               # doing continuously
            'గుతున్న', 'గుతాడు', 'గుతారు',                   # going
        }
        
        # Specific verb patterns (not just endings)
        verb_patterns = [
            any(pattern in text for pattern in ['తున్న', '్తున్న', 'ుతున్న']),  # present continuous
            any(pattern in text for pattern in ['్తాడు', '్తారు', '్తుంది']),   # future
            any(pattern in text for pattern in ['్చాడు', '్చారు', '్చింది']),   # past
            any(pattern in text for pattern in ['కుంట', 'గుత', 'పడత']),        # aspect markers
            text.endswith('ు') and len(text) > 3 and text not in _PERSON_OVERRIDE,  # verb roots but not names
        ]
        
        # Check for verb endings (exact matches)
        has_verb_ending = any(text.endswith(ending) for ending in verb_endings)
        
        # Check for verb patterns
        has_verb_pattern = any(verb_patterns)
        
        # Proper names that should NEVER be classified as verbs
        proper_names = {'రాముడు', 'కృష్ణుడు', 'బాలయ్య', 'సీత', 'లక్ష్మి', 'హనుమంతుడు'}
        if text in proper_names:
            return False
        
        # Known person names from override list
        if text in _PERSON_OVERRIDE:
            # Only classify as verb if it has clear verb morphology
            return has_verb_ending and len(text) > 5
        
        return has_verb_ending or has_verb_pattern
    
    def _is_common_word(self, token: str) -> bool:
        """Check if token is a common word that shouldn't be an entity"""
        return token in self.common_words
    
    def _infer_entity_type(self, analysis: Dict[str, Any]) -> str:
        """Enhanced but balanced entity type inference"""
        text = analysis['text']
        
        # 1. Check verb analysis FIRST (verbs should stay as verbs)
        verb_analysis = analysis.get('verb_analysis', {})
        if verb_analysis.get('is_verb', False) and verb_analysis.get('confidence', 0) > 0.7:
            return 'verb'
        
        # 2. Check against expanded word lists with CONTEXT
        if (text in _PERSON_OVERRIDE or text in _PERSON_WORDS) and len(text) > 2:
            # Additional check: person names are usually not verbs
            if not self._looks_like_verb(text):
                return 'person'
                
        elif text in _PLACE_OVERRIDE and len(text) > 2:
            return _PLACE_OVERRIDE[text]
            
        elif text in _ORGANIZATION_WORDS and len(text) > 3:
            return 'organization'
            
        elif text in _TEMPORAL_WORDS and len(text) > 2:
            return 'temporal'
            
        elif text in _ABSTRACT_WORDS and len(text) > 2:
            return 'abstract'
            
        elif text in _ARTIFACT_WORDS and len(text) > 2:
            return 'artifact'
            
        elif text in _NATURE_WORDS and len(text) > 2:
            return 'nature'
        
        # 3. Fall back to suffix-based inference
        suffix_cat = analysis.get('suffix_category')
        if suffix_cat == 'place':
            return 'place'
        elif suffix_cat == 'person':
            return 'person'
        elif suffix_cat == 'verbal':
            return 'verb_derived'
            
        # Default to noun
        return 'noun'
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Balanced confidence calculation"""
        text = analysis['text']
        base_confidence = 0.5
        
        # Boost for suffix analysis
        if analysis.get('suffix_confidence'):
            base_confidence = max(base_confidence, analysis['suffix_confidence'])
            
        # Boost for verb analysis
        if analysis.get('verb_analysis', {}).get('confidence', 0) > base_confidence:
            base_confidence = analysis['verb_analysis']['confidence']
            
        # Boost confidence for known words (but be conservative)
        if (text in _PERSON_OVERRIDE or text in _PERSON_WORDS) and len(text) > 2:
            return min(base_confidence + 0.2, 0.95)
        elif text in _PLACE_OVERRIDE and len(text) > 2:
            return min(base_confidence + 0.15, 0.95)
        elif text in _ORGANIZATION_WORDS and len(text) > 3:
            return min(base_confidence + 0.1, 0.90)
        elif text in _TEMPORAL_WORDS and len(text) > 2:
            return min(base_confidence + 0.1, 0.85)
        elif text in _ARTIFACT_WORDS and len(text) > 2:
            return min(base_confidence + 0.1, 0.85)
        
        # Penalize unknown types
        if analysis['entity_type'] == 'unknown':
            base_confidence *= 0.7
            
        return min(base_confidence, 1.0)
    
    def _is_telugu_word(self, token: str) -> bool:
        """Enhanced filtering - exclude common particles and short words"""
        if not token or len(token) < 2:
            return False
            
        # Exclude common particles that shouldn't be entities
        common_particles = {
            'లో', 'కు', 'కి', 'నుండి', 'నుంచి', 'తో', 'గురించి', 
            'కోసం', 'వల్ల', 'చేత', 'ద్వారా', 'వద్ద', 'గా', 'అయితే',
            'మరియు', 'కానీ', 'అందువల్ల', 'అయినప్పటికీ', 'ఎందుకంటే',
            'కాబట్టి', 'ముందు', 'తర్వాత', 'పైన', 'కింద', 'లోపల', 'బయట',
            'వెంట', 'గుర్తు', 'చేతన', 'వలన', 'బట్టి', 'కంటే', 'కంటె'
        }
        if token in common_particles:
            return False
        
        # Exclude single character words (except in compounds)
        if len(token) == 1 and token not in _PLACE_OVERRIDE and token not in _PERSON_OVERRIDE:
            return False
            
        # Exclude numbers and punctuation
        if any(c in _PUNCTUATION for c in token):
            return False
            
        # Must contain Telugu characters
        return any('\u0C00' <= c <= '\u0C7F' for c in token)
    
    def _handle_known_exception(self, token: str) -> Dict[str, Any]:
        """Enhanced exception handling with higher confidence"""
        if token in _PERSON_OVERRIDE:
            return {
                'entity_type': _PERSON_OVERRIDE[token], 
                'confidence': 0.98,
                'is_proper_name': True
            }
        elif token in _PLACE_OVERRIDE:
            return {
                'entity_type': _PLACE_OVERRIDE[token], 
                'confidence': 0.98,
                'is_known_place': True
            }
        elif token in _COMMON_EXCEPTIONS:
            stem, suffix = _COMMON_EXCEPTIONS[token]
            return {
                'entity_type': 'noun',
                'stem': stem,
                'suffix': suffix,
                'confidence': 0.95,
                'is_inflected_form': True
            }
        return {}
    
    def _analyze_suffixes(self, token: str) -> Dict[str, Any]:
        """Enhanced suffix analysis with better confidence"""
        best_match = (None, 0.0, {})
        
        for suffix in _SORTED_SUFFIXES:
            if token.endswith(suffix):
                category, conf, meta = _SUFFIX_LOOKUP[suffix]
                stem = token[:-len(suffix)]
                
                # Enhanced stem validation
                if len(stem) >= 2 and self._is_valid_stem(stem):
                    if len(suffix) > best_match[1]:
                        best_match = (suffix, len(suffix), {
                            'stem': stem,
                            'suffix_category': category,
                            'suffix_confidence': conf,
                            'suffix_metadata': meta,
                            'is_inflected': True
                        })
                        
        if best_match[0]:
            return best_match[2]
        return {}
    
    def _is_valid_stem(self, stem: str) -> bool:
        """Check if stem is valid (not just particles)"""
        if len(stem) < 2:
            return False
            
        # Exclude stems that are just common particles
        common_particles = {'లో', 'కు', 'కి', 'తో', 'గా', 'వల్ల'}
        if stem in common_particles:
            return False
            
        return True
    
    def _analyze_compounds(self, token: str) -> List[List[str]]:
        """Enhanced compound splitting with better validation"""
        if len(token) <= 6:  # Too short for meaningful compounds
            return [[token]]
            
        try:
            splits = self.compound_splitter.split(token)
            
            # Validate splits - ensure components make sense
            validated_splits = []
            for split in splits:
                if len(split) > 1 and all(len(part) >= 2 for part in split):
                    # Check if components are valid words
                    if all(self._is_valid_component(part) for part in split):
                        validated_splits.append(split)
            
            return validated_splits if validated_splits else [[token]]
            
        except Exception as e:
            logger.debug(f"Compound splitting failed for {token}: {e}")
            return [[token]]
    
    def _is_valid_component(self, component: str) -> bool:
        """Check if a compound component is valid"""
        if len(component) < 2:
            return False
            
        # Component should contain Telugu characters
        if not any('\u0C00' <= c <= '\u0C7F' for c in component):
            return False
            
        # Should not be just common particles
        common_particles = {'లో', 'కు', 'కి', 'తో', 'గా'}
        if component in common_particles:
            return False
            
        return True
    
    def _analyze_verb(self, token: str) -> Dict[str, Any]:
        """Enhanced verb analysis with better root extraction"""
        tense = self.verb_morphology.detect_tense_aspect(token)
        root = self.verb_morphology.extract_verb_root(token, self.lexicon.verb_roots)
        
        # Enhanced root validation
        if root and len(root) < 2:
            root = None
            
        confidence = 0.0
        if root:
            confidence = 0.9
        elif tense:
            confidence = 0.7
        elif self._looks_like_verb(token):
            confidence = 0.6
            
        return {
            'is_verb': tense is not None or root is not None or confidence > 0.5,
            'tense_aspect': tense,
            'verb_root': root,
            'confidence': confidence
        }
    
    def clear_cache(self):
        """Clear the entity cache"""
        with self._cache_lock:
            self._cache.clear()
        logger.debug("UltraEntityExtractor cache cleared")

# ==================== ULTRA RELATION EXTRACTOR ====================
class UltraRelationExtractor:
    """Ultra relation extractor with better filtering"""
    
    def __init__(self, config: TurboKGConfig, lexicon_manager: LexiconManager):
        self.config = config
        self.lexicon = lexicon_manager
        self.relation_patterns = self._build_enhanced_patterns()
        self.cooccurrence_threshold = 3
        self.distance_threshold = 5
        self.performance_monitor = PerformanceMonitor()
        
    def _build_enhanced_patterns(self) -> List[Dict[str, Any]]:
        """Build comprehensive relation patterns for Telugu including new SOV rules"""
        return [
            # NEW: Telugu SOV (Subject-Object-Verb) Pattern
            {
                'name': 'telugu_sov_pattern',
                'pattern': [
                    {'type': 'person', 'role': 'subject'},   # Subject
                    {'type': 'noun', 'role': 'object'},      # Object
                    {'type': 'verb', 'role': 'action'}       # Verb
                ],
                'relation': 'performs_action_on',
                'confidence': 0.90,
                'syntax': 'SOV'
            },
            
            # NEW: Instrumental Relation (with/by - తో)
            {
                'name': 'instrumental_relation',
                'pattern': [
                    {'type': 'noun', 'role': 'instrument'},
                    {'text': 'తో', 'required': True},
                    {'type': 'verb', 'role': 'action'}
                ],
                'relation': 'used_instrument',
                'confidence': 0.80,
                'syntax': 'INSTRUMENTAL'
            },

            # NEW: Ablative Relation (from - నుండి/నుంచి)
            {
                'name': 'ablative_relation',
                'pattern': [
                    {'type': 'place', 'role': 'source'},
                    {'text': 'నుండి', 'required': True},
                    {'type': 'verb', 'role': 'action'}
                ],
                'relation': 'originated_from',
                'confidence': 0.85,
                'syntax': 'ABLATIVE'
            },

            # Existing patterns refined
            {
                'name': 'subject_verb_object',
                'pattern': [
                    {'type': 'person', 'role': 'subject'},
                    {'type': 'verb', 'role': 'action'}, 
                    {'type': 'noun', 'role': 'object'}
                ],
                'relation': 'performs_action_on',
                'confidence': 0.85,
                'syntax': 'SVO'
            },
            {
                'name': 'subject_verb',
                'pattern': [
                    {'type': 'person', 'role': 'subject'},
                    {'type': 'verb', 'role': 'action'}
                ],
                'relation': 'performs',
                'confidence': 0.80,
                'syntax': 'SV'
            },
            
            # POSSESSION patterns
            {
                'name': 'possession_genitive',
                'pattern': [
                    {'type': 'person', 'role': 'owner'},
                    {'text': 'యొక్క', 'required': True},
                    {'type': 'noun', 'role': 'possession'}
                ],
                'relation': 'owns',
                'confidence': 0.90,
                'syntax': 'GENITIVE'
            },
            
            # LOCATION patterns  
            {
                'name': 'person_location',
                'pattern': [
                    {'type': 'person', 'role': 'entity'},
                    {'type': 'place', 'role': 'location'}
                ],
                'relation': 'located_at',
                'confidence': 0.75,
                'max_distance': 3
            },
            {
                'name': 'entity_in_location',
                'pattern': [
                    {'type': 'noun', 'role': 'entity'},
                    {'text': 'లో', 'required': True},
                    {'type': 'place', 'role': 'location'}
                ],
                'relation': 'located_in',
                'confidence': 0.85,
                'syntax': 'POSTPOSITION'
            },
            
            # TEMPORAL patterns
            {
                'name': 'event_time',
                'pattern': [
                    {'type': 'verb', 'role': 'event'},
                    {'type': 'temporal', 'role': 'time'}
                ],
                'relation': 'occurs_at',
                'confidence': 0.70
            },
            
            # ACTION-PURPOSE patterns
            {
                'name': 'action_purpose',
                'pattern': [
                    {'type': 'verb', 'role': 'action'},
                    {'text': 'కోసం', 'required': True},
                    {'type': 'noun', 'role': 'purpose'}
                ],
                'relation': 'for_purpose_of',
                'confidence': 0.88
            }
        ]
    
    def extract_relations(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Enhanced relation extraction with better filtering"""
        # Filter out low-quality entities first
        filtered_entities = [
            ent for ent in entities 
            if ent.get('confidence', 0) > 0.6  # Higher threshold
            and ent.get('entity_type') != 'unknown'
            and len(ent['text']) > 1  # Exclude single characters
        ]
        
        if len(filtered_entities) < 2:
            return []
            
        start_time = time.time()
        relations = []
        
        # Strategy 1: Pattern-based extraction (highest priority)
        pattern_relations = self._extract_using_patterns(filtered_entities, text)
        relations.extend(pattern_relations)
        
        # Strategy 2: Conservative co-occurrence (only for frequent, close pairs)
        cooccurrence_relations = self._extract_conservative_cooccurrence(filtered_entities, text)
        relations.extend(cooccurrence_relations)
        
        # Remove duplicates and low-confidence relations
        relations = self._deduplicate_relations(relations)
        
        processing_time = time.time() - start_time
        self.performance_monitor.track_metric('relation_extraction_time', processing_time)
        
        return relations
    
    def _extract_using_patterns(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Extract relations using predefined linguistic patterns"""
        relations = []
        tokens = _TOKEN_RE.findall(text)
        
        for pattern in self.relation_patterns:
            for i in range(len(entities) - len(pattern['pattern']) + 1):
                candidate_entities = entities[i:i + len(pattern['pattern'])]
                
                if self._matches_pattern(candidate_entities, pattern, tokens):
                    relation = self._create_relation_from_pattern(candidate_entities, pattern, text)
                    if relation and relation['confidence'] >= self.config.min_confidence:
                        relations.append(relation)
        
        return relations
    
    def _matches_pattern(self, entities: List[Dict], pattern: Dict, tokens: List[str]) -> bool:
        """Check if entities match a relation pattern"""
        if len(entities) != len(pattern['pattern']):
            return False
            
        for entity, pattern_item in zip(entities, pattern['pattern']):
            # Check entity type
            if 'type' in pattern_item and entity.get('entity_type') != pattern_item['type']:
                return False
                
            # Check specific text (for relational words)
            if 'text' in pattern_item:
                if entity.get('text') != pattern_item['text']:
                    return False
                if pattern_item.get('required', False) and entity.get('confidence', 0) < 0.8:
                    return False
            
            # Check distance constraints
            max_distance = pattern.get('max_distance')
            if max_distance and len(entities) > 1:
                positions = [e.get('position', 0) for e in entities]
                if max(positions) - min(positions) > max_distance:
                    return False
        
        return True
    
    def _create_relation_from_pattern(self, entities: List[Dict], pattern: Dict, text: str) -> Dict[str, Any]:
        """Create a relation from matched pattern"""
        # Find source and target based on roles
        source = None
        target = None
        
        for entity, pattern_item in zip(entities, pattern['pattern']):
            role = pattern_item.get('role', '')
            if role in ['subject', 'owner', 'entity', 'part']:
                source = entity
            elif role in ['object', 'possession', 'location', 'time', 'purpose', 'whole']:
                target = entity
        
        if not source or not target:
            return None
            
        return {
            'source': source['text'],
            'target': target['text'],
            'source_type': source.get('entity_type', 'unknown'),
            'target_type': target.get('entity_type', 'unknown'),
            'relation_type': pattern['relation'],
            'confidence': pattern['confidence'],
            'pattern': pattern['name'],
            'syntax': pattern.get('syntax', 'unknown'),
            'context': self._get_context(text, source.get('position', 0)),
            'evidence': 'pattern_matching'
        }
    
    def _extract_conservative_cooccurrence(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Conservative co-occurrence with strict filters"""
        relations = []
        
        if len(entities) < 2:
            return relations
            
        # Only consider entities that are close to each other
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i+1:], i+1):
                pos1 = ent1.get('position', 0)
                pos2 = ent2.get('position', 0)
                
                # Strict distance filter
                if abs(pos1 - pos2) > self.distance_threshold:
                    continue
                    
                # Type compatibility filter
                if not self._are_types_compatible(ent1, ent2):
                    continue
                
                # Create relation with low confidence
                relations.append({
                    'source': ent1['text'],
                    'target': ent2['text'],
                    'relation_type': 'possibly_related_to',
                    'confidence': 0.3,  # Much lower confidence
                    'pattern': 'conservative_cooccurrence',
                    'distance': abs(pos1 - pos2),
                    'evidence': 'proximity'
                })
        
        return relations
    
    def _are_types_compatible(self, ent1: Dict, ent2: Dict) -> bool:
        """Stricter type compatibility checking"""
        type1 = ent1.get('entity_type', 'unknown')
        type2 = ent2.get('entity_type', 'unknown')
        
        # More specific compatibility rules
        compatible_pairs = {
            ('person', 'verb'), ('person', 'place'), ('person', 'organization'),
            ('verb', 'noun'), ('verb', 'place'), ('verb', 'temporal'),
            ('place', 'organization'), ('person', 'artifact'),
        }
        
        pair = (type1, type2)
        reverse_pair = (type2, type1)
        
        return pair in compatible_pairs or reverse_pair in compatible_pairs
    
    def _get_context(self, text: str, position: int, window: int = 3) -> str:
        """Get context around a position"""
        tokens = _TOKEN_RE.findall(text)
        start = max(0, position - window)
        end = min(len(tokens), position + window + 1)
        return ' '.join(tokens[start:end])
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """Remove duplicate relations, keeping highest confidence"""
        relation_key = lambda r: (r['source'], r['target'], r['relation_type'])
        
        unique_relations = {}
        for rel in relations:
            key = relation_key(rel)
            if key not in unique_relations or rel['confidence'] > unique_relations[key]['confidence']:
                unique_relations[key] = rel
        
        return list(unique_relations.values())

# ==================== ENHANCED TURBOKG CLASS ====================
class EnhancedTurboKG:
    """Enhanced TurboKG with balanced linguistic accuracy"""
    
    def __init__(self, config: Optional[Union[TurboKGConfig, Dict[str, Any]]] = None):
        if config is None:
            config = TurboKGConfig()
            
        self.config = config
        self.lexicon_manager = LexiconManager(config.verb_roots_path, config.stems_path)
        
        # SELECT EXTRACTOR BASED ON CONFIG
        if self.config.use_stanza and _HAS_STANZA:
            logger.info("🤖 Using Stanza Entity Extractor")
            self.entity_extractor = StanzaEntityExtractor(config)
        else:
            if self.config.use_stanza and not _HAS_STANZA:
                logger.warning("⚠️ Stanza requested but not found. Falling back to Ultra Extractor.")
            logger.info("⚡ Using Ultra (Rule-based) Entity Extractor")
            self.entity_extractor = UltraEntityExtractor(self.lexicon_manager, config)

        self.relation_extractor = UltraRelationExtractor(config, self.lexicon_manager)
        self.sandhi_engine = EnhancedTeluguSandhiEngine(config.sandhi_mode)
        
        # Initialize optional components
        self.faiss_index = None
        self.embedding_model = None
        self.neo4j_driver = None
        self.prom_metrics = None
        
        self._setup_optional_components()
        self.performance_monitor = PerformanceMonitor()
        
        enhanced_status = "ENHANCED" if ENHANCED_DATA_LOADED else "STANDARD"
        logger.info(f"🚀 TurboKG ULTRA v7.3 {enhanced_status} initialized with {self.config.num_workers} workers")
        if ENHANCED_DATA_LOADED:
            logger.info(f"✅ Enhanced linguistic data: {len(_PLACE_OVERRIDE)} places, {len(_PERSON_OVERRIDE)} persons")
    
    def _setup_optional_components(self):
        """Setup optional components"""
        # FAISS setup
        if self.config.faiss_enabled and _HAS_FAISS:
            try:
                dimension = 384
                self.faiss_index = faiss.IndexFlatIP(dimension)
                logger.info("✅ FAISS index initialized")
            except Exception as e:
                logger.warning(f"⚠️ FAISS initialization failed: {e}")
                self.config.faiss_enabled = False
                
        # Embeddings setup
        if self.config.enable_embeddings and _HAS_STMODEL:
            try:
                self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
                logger.info(f"✅ Embedding model loaded: {self.config.embedding_model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Embedding model loading failed: {e}")
                self.config.enable_embeddings = False
                
        # Neo4j setup
        if self.config.use_neo4j and _HAS_NEO4J:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_pass)
                )
                with self.neo4j_driver.session() as session:
                    session.run("RETURN 1")
                logger.info("✅ Neo4j connection established")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j connection failed: {e}")
                self.config.use_neo4j = False
                
        # Prometheus setup
        if self.config.prom_port and _HAS_PROM:
            try:
                start_http_server(self.config.prom_port)
                self.prom_metrics = {
                    'documents_processed': Counter('turbo_kg_documents_processed', 'Number of documents processed'),
                    'entities_extracted': Counter('turbo_kg_entities_extracted', 'Number of entities extracted'),
                    'relations_extracted': Counter('turbo_kg_relations_extracted', 'Number of relations extracted'),
                    'processing_time': Histogram('turbo_kg_processing_time', 'Document processing time'),
                    'memory_usage': Gauge('turbo_kg_memory_usage', 'Memory usage in MB')
                }
                logger.info(f"✅ Prometheus metrics on port {self.config.prom_port}")
            except Exception as e:
                logger.warning(f"⚠️ Prometheus setup failed: {e}")
    
    def process_document(self, text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single document and extract knowledge graph"""
        start_time = time.time()
        
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            
        # Validate input
        if not text or len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty")
            
        if len(text) > self.config.max_document_size_mb * 1024 * 1024:
            raise ValueError(f"Document size exceeds {self.config.max_document_size_mb}MB limit")
            
        # Enhanced processing pipeline
        entities = self.entity_extractor.extract_entities(text)
        relations = self.relation_extractor.extract_relations(entities, text)
        
        # Generate embeddings if enabled
        embeddings = {}
        if self.config.enable_embeddings and self.embedding_model:
            embeddings = self._generate_embeddings(entities)
            
        # Build knowledge graph
        kg = {
            'doc_id': doc_id,
            'entities': entities,
            'relations': relations,
            'embeddings': embeddings,
            'metadata': {
                'processing_time': time.time() - start_time,
                'entity_count': len(entities),
                'relation_count': len(relations),
                'text_length': len(text),
                'timestamp': time.time()
            }
        }
        
        # Update metrics
        self._update_metrics(kg)
        
        # Export to Neo4j if enabled
        if self.config.use_neo4j:
            self._export_to_neo4j(kg)
            
        return kg
        
    def _generate_embeddings(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings for entities"""
        embeddings = {}
        try:
            entity_texts = [e['text'] for e in entities]
            if entity_texts:
                embeds = self.embedding_model.encode(entity_texts)
                for i, entity in enumerate(entities):
                    embeddings[entity['text']] = embeds[i].tolist()
                    
                # Update FAISS index if enabled
                if self.config.faiss_enabled and self.faiss_index is not None:
                    embeds_np = np.array(embeds).astype('float32')
                    self.faiss_index.add(embeds_np)
                    
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            
        return embeddings
        
    def _update_metrics(self, kg: Dict[str, Any]):
        """Update performance metrics"""
        if self.prom_metrics:
            self.prom_metrics['documents_processed'].inc()
            self.prom_metrics['entities_extracted'].inc(len(kg['entities']))
            self.prom_metrics['relations_extracted'].inc(len(kg['relations']))
            self.prom_metrics['processing_time'].observe(kg['metadata']['processing_time'])
            
            if _HAS_PSUTIL:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.prom_metrics['memory_usage'].set(memory_mb)
                
        # Update performance monitor
        self.performance_monitor.track_metric('memory_usage_mb', 
            psutil.Process().memory_info().rss / 1024 / 1024 if _HAS_PSUTIL else 0)
        self.performance_monitor.track_metric('processing_time_sec', kg['metadata']['processing_time'])
        
    def _export_to_neo4j(self, kg: Dict[str, Any]):
        """Export knowledge graph to Neo4j"""
        if not self.neo4j_driver:
            return
            
        try:
            with self.neo4j_driver.session(database=self.config.neo4j_database) as session:
                # Create entities
                for entity in kg['entities']:
                    query = """
                    MERGE (e:Entity {text: $text})
                    SET e.entity_type = $entity_type,
                        e.confidence = $confidence,
                        e.doc_id = $doc_id
                    """
                    session.run(query, {
                        'text': entity['text'],
                        'entity_type': entity.get('entity_type', 'unknown'),
                        'confidence': entity.get('confidence', 0.0),
                        'doc_id': kg['doc_id']
                    })
                    
                # Create relations
                for relation in kg['relations']:
                    query = """
                    MATCH (source:Entity {text: $source})
                    MATCH (target:Entity {text: $target})
                    MERGE (source)-[r:RELATION {type: $relation_type}]->(target)
                    SET r.confidence = $confidence,
                        r.pattern = $pattern,
                        r.doc_id = $doc_id
                    """
                    session.run(query, {
                        'source': relation['source'],
                        'target': relation['target'],
                        'relation_type': relation['relation_type'],
                        'confidence': relation['confidence'],
                        'pattern': relation.get('pattern', 'unknown'),
                        'doc_id': kg['doc_id']
                    })
                    
        except Exception as e:
            logger.warning(f"Neo4j export failed: {e}")
            
    def batch_process(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Batch process multiple documents"""
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have the same length")
            
        results = []
        
        if self.config.enable_parallel:
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                future_to_doc = {
                    executor.submit(self.process_document, text, doc_id): (text, doc_id)
                    for text, doc_id in zip(texts, doc_ids)
                }
                
                for future in as_completed(future_to_doc):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Document processing failed: {e}")
        else:
            for text, doc_id in zip(texts, doc_ids):
                try:
                    result = self.process_document(text, doc_id)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Document processing failed: {e}")
                    
        return results
        
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar entities using FAISS"""
        if not self.config.faiss_enabled or self.faiss_index is None:
            raise RuntimeError("FAISS is not enabled or initialized")
            
        if not self.embedding_model:
            raise RuntimeError("Embedding model is not available")
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        # Return results (note: this is simplified - you'd need to map indices back to entities)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < self.faiss_index.ntotal:
                results.append({
                    'index': idx,
                    'score': float(score)
                })
                
        return results
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = self.performance_monitor.get_summary()
        
        # Add component-specific stats
        summary['sandhi_engine'] = self.sandhi_engine.get_cache_stats()
        if hasattr(self.entity_extractor, 'performance_monitor'):
            summary['entity_extractor'] = self.entity_extractor.performance_monitor.get_summary()
        summary['relation_extractor'] = self.relation_extractor.performance_monitor.get_summary()
        
        # Add memory info
        if _HAS_PSUTIL:
            process = psutil.Process()
            summary['memory'] = {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'vms_mb': process.memory_info().vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
            
        return summary
        
    def clear_caches(self):
        """Clear all internal caches"""
        if hasattr(self.entity_extractor, '_cache'):
             self.entity_extractor._cache.clear()
        if hasattr(self.entity_extractor, 'clear_cache'):
             self.entity_extractor.clear_cache()
             
        self.sandhi_engine.clear_cache()
        gc.collect()
        logger.info("✅ All caches cleared")
        
    def __del__(self):
        """Enhanced cleanup"""
        try:
            if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
                self.neo4j_driver.close()
            if hasattr(self, 'faiss_index'):
                del self.faiss_index
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
        except Exception:
            pass  # Avoid exceptions in destructor

# ==================== OPTIMIZED FILE PROCESSING UTILITIES ====================
class JsonStreamWriter:
    """
    Writes items to a JSON file incrementally while maintaining valid JSON array syntax.
    """
    def __init__(self, filename: Path, mode: str = "json"):
        self.filename = filename
        self.mode = mode  # 'json' (array) or 'jsonl' (lines)
        self.file = None
        self.first_item = True

    def __enter__(self):
        self.file = open(self.filename, 'w', encoding='utf-8')
        if self.mode == "json":
            self.file.write("[\n")
        return self

    def write(self, item: Dict[str, Any]):
        if self.mode == "json":
            if not self.first_item:
                self.file.write(",\n")
            json.dump(item, self.file, cls=TurboKGJSONEncoder, ensure_ascii=False, indent=2)
            self.first_item = False
        else:
            # JSONL mode
            json.dump(item, self.file, cls=TurboKGJSONEncoder, ensure_ascii=False)
            self.file.write("\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "json" and self.file:
            self.file.write("\n]")
        if self.file:
            self.file.close()


class LargeFileProcessor:
    """
    Memory-efficient file processor for massive datasets using streaming and parallel processing.
    """
    def __init__(self, turbo_kg: EnhancedTurboKG, output_dir: Path, batch_size: int = 100):
        self.turbo_kg = turbo_kg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        # Queue control: Allow 4x pending tasks per worker before pausing reading
        self.max_queue_size = self.turbo_kg.config.num_workers * 4
        self._shutdown_event = threading.Event() 

    def _read_text_generator(self, file_path: Path, chunk_size_lines: int = 50):
        """Yields text chunks from a large text file without loading everything."""
        buffer = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    buffer.append(line.strip())
                if len(buffer) >= chunk_size_lines:
                    yield " ".join(buffer)
                    buffer = []
            if buffer:
                yield " ".join(buffer)

    def _read_jsonl_generator(self, file_path: Path):
        """Yields objects from a JSONL file line by line."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load processing checkpoint safely."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_checkpoint(self, doc_idx: int, stats: Dict[str, Any]):
        """Save processing checkpoint atomically."""
        try:
            checkpoint_data = {
                'last_processed': doc_idx,
                'stats': stats,
                'timestamp': time.time()
            }
            # Write to temp then rename to ensure atomic write (prevents corruption if crash during write)
            temp_path = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.replace(self.checkpoint_file)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _handle_result(self, future, writer, stats):
        """Callback to handle a finished processing task and write to disk."""
        try:
            result = future.result()
            writer.write(result)
            
            # Update stats
            stats['processed'] += 1
            stats['entities'] += len(result['entities'])
            stats['relations'] += len(result['relations'])
            
            if stats['processed'] % 100 == 0:
                logger.info(f"⏳ Progress: {stats['processed']} chunks processed...")
        except Exception as e:
            stats['failed'] += 1
            logger.error(f"❌ Task failed: {e}")

    def _show_progress(self, processed: int, stats: Dict):
        """Show processing summary log."""
        logger.info(f"📊 Status: {processed} docs | Entities: {stats['entities']} | Relations: {stats['relations']}")

    def process_large_file(self, input_file: Path, output_format: str = "json"):
        """
        Main entry point. Uses Producer-Consumer pattern with Backpressure.
        """
        start_time = time.time()
        input_path = Path(input_file)
        output_file = self.output_dir / f"{input_path.stem}_kg.{output_format}"
        
        logger.info(f"🚀 Starting processing: {input_file}")
        
        # 1. Load Checkpoint
        checkpoint = self._load_checkpoint()
        start_from_idx = checkpoint.get('last_processed', -1) + 1
        if start_from_idx > 0:
            logger.info(f"🔄 Resuming from index: {start_from_idx}")

        # 2. Determine Iterator
        if input_path.suffix.lower() == '.jsonl':
            generator = self._read_jsonl_generator(input_path)
            is_jsonl = True
        else:
            generator = self._read_text_generator(input_path)
            is_jsonl = False

        stats = checkpoint.get('stats', {'processed': 0, 'failed': 0, 'entities': 0, 'relations': 0})

        # 3. Start Stream Writer
        mode = 'a' if is_jsonl and start_from_idx > 0 else 'w'
        
        with JsonStreamWriter(output_file, mode=output_format) as writer:
            futures = set()
            
            # 4. Thread Pool Execution
            with ThreadPoolExecutor(max_workers=self.turbo_kg.config.num_workers) as executor:
                
                doc_idx = 0
                for item in generator:
                    # Skip already processed items (Fast-forward)
                    if doc_idx < start_from_idx:
                        doc_idx += 1
                        continue

                    if self._shutdown_event.is_set():
                        break
                    
                    # Prepare Data
                    if is_jsonl:
                        text = item.get('text', '') if isinstance(item, dict) else str(item)
                        doc_id = item.get('id', f"doc_{doc_idx}")
                    else:
                        text = item
                        doc_id = f"chunk_{doc_idx}"

                    if not text or len(text) < 5:
                        doc_idx += 1
                        continue

                    # Submit Task
                    future = executor.submit(self.turbo_kg.process_document, text, doc_id)
                    futures.add(future)
                    doc_idx += 1

                    # 5. Backpressure (Flow Control)
                    # If queue is full, wait for tasks to finish before reading more
                    if len(futures) >= self.max_queue_size:
                        # Wait for at least one task to finish
                        done, _ = wait(futures, timeout=None, return_when=FIRST_COMPLETED)
                        
                        for f in done:
                            self._handle_result(f, writer, stats)
                            futures.remove(f)
                            
                            # Periodic Checkpoint
                            if stats['processed'] % 100 == 0:
                                self._save_checkpoint(doc_idx, stats)
                                self._show_progress(stats['processed'], stats)

                # 6. Drain remaining tasks
                for f in as_completed(futures):
                    self._handle_result(f, writer, stats)

        # Final Save
        self._save_checkpoint(doc_idx, stats)
        duration = time.time() - start_time
        logger.info(f"✅ Done in {duration:.2f}s. Entities: {stats['entities']}, Relations: {stats['relations']}")
        
    def stop_processing(self):
        """Gracefully stop processing"""
        self._shutdown_event.set()
        logger.info("🛑 Processing stopped by user request")

# ==================== VERIFICATION AND TESTING ====================
def verify_large_file_processor():
    """Verify that the large file processor works correctly"""
    logger.info("🧪 Verifying Large File Processor...")
    
    # Create test data
    test_dir = Path("test_large_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create a test file with various content
    test_file = test_dir / "test_input.jsonl"
    test_data = [
        {"id": "doc1", "text": "రాముడు పుస్తకం చదివాడు. అతను గ్రంథాలయంలో ఉన్నాడు."},
        {"id": "doc2", "text": "సీత ఫలాలు తిన్నాది. ఆమె బజారులో కొన్నాది."},
        {"id": "doc3", "text": "కృష్ణుడు గోపికలతో నృత్యం చేశాడు. అతను ఫ్లూట్ వాయించాడు."},
        {"id": "doc4", "text": "అర్జునుడు విలువిద్యలో నిపుణుడు. అతను కురుక్షేత్ర యుద్ధంలో భాగస్వామి."},
        {"id": "doc5", "text": "భీష్ముడు ప్రతిజ్ఞ చేశాడు. అతను జీవితాంతం బ్రహ్మచర్యం పాటించాడు."}
    ]
    
    # Write test data
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    # Test configuration
    config = TurboKGConfig(
        min_confidence=0.6,
        num_workers=2,
        enable_sandhi=True,
        use_stanza=False,  # Use rule-based for faster testing
        max_document_size_mb=10
    )
    
    try:
        # Initialize TurboKG
        kg_engine = EnhancedTurboKG(config)
        
        # Initialize processor
        processor = LargeFileProcessor(kg_engine, test_dir)
        
        # Process the test file
        logger.info("🔄 Processing test file...")
        processor.process_large_file(test_file, output_format="jsonl")
        
        # Verify output
        output_file = test_dir / "test_input_kg.jsonl"
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            logger.info(f"✅ Verification successful! Processed {len(results)} documents")
            logger.info(f"📊 Entities extracted: {sum(len(r['entities']) for r in results)}")
            logger.info(f"🔗 Relations extracted: {sum(len(r['relations']) for r in results)}")
            
            # Show sample results
            if results:
                sample = results[0]
                logger.info(f"📝 Sample - Entities: {[e['text'] for e in sample['entities']]}")
                logger.info(f"🔗 Sample - Relations: {[(r['source'], r['relation_type'], r['target']) for r in sample['relations']]}")
            
            return True
        else:
            logger.error("❌ Output file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        output_file = test_dir / "test_input_kg.jsonl"
        if output_file.exists():
            output_file.unlink()
        checkpoint_file = test_dir / "processing_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if test_dir.exists():
            test_dir.rmdir()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="TurboKG ULTRA v7.3 - Large File Handler")
    parser.add_argument("-i", "--input", required=True, help="Input file path (.txt or .jsonl)")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--format", choices=['json', 'jsonl'], default='json', help="Output format")
    parser.add_argument("--enhanced", action='store_true', help="Force use enhanced dictionary")
    parser.add_argument("--stanza", action='store_true', help="Enable Stanza Neural Pipeline")
    parser.add_argument("--verify", action='store_true', help="Run verification test before processing")
    
    args = parser.parse_args()
    
    try:
        # Run verification if requested
        if args.verify:
            if verify_large_file_processor():
                logger.info("✅ Verification passed! Starting main processing...")
            else:
                logger.error("❌ Verification failed! Exiting.")
                sys.exit(1)
        
        # 1. Configuration
        config = TurboKGConfig(
            min_confidence=0.7,
            num_workers=args.workers,
            enable_sandhi=True,
            sandhi_mode="adaptive",
            # Disable Neo4j/Embeddings by default for raw file processing speed unless configured
            use_neo4j=False, 
            enable_embeddings=False,
            # Enable Stanza via flag
            use_stanza=args.stanza
        )
        
        # 2. Initialize Engine
        logger.info("⚙️ Initializing TurboKG Engine...")
        kg_engine = EnhancedTurboKG(config)
        
        # 3. Initialize Large File Processor
        processor = LargeFileProcessor(kg_engine, args.output)
        
        # 4. Run Processing
        input_path = Path(args.input)
        if not input_path.exists():
            logger.critical(f"❌ Input file not found: {input_path}")
            sys.exit(1)
            
        processor.process_large_file(input_path, output_format=args.format)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Processing interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)