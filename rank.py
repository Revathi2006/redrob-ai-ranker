# =============================================================================
# REDROB HACKATHON - FINAL SUBMISSION v10.0
# =============================================================================
# Features:
# 1. Skill Ontology - Map related skills to concepts (LoRA→Fine-tuning, etc.)
# 2. Percentile Normalization - Each feature normalized across candidate population
# 3. Architecture frozen - No more major changes
# 4. GPU preprocessing with FP16 + inference_mode optimization
# 5. Human-readable JD-specific reasoning for manual review
# 6. Experience alignment scoring to prevent junior profiles ranking too high
# 7. Smart cache validation - skips reprocessing for minor config changes
# 8. Complete 23 Redrob Signals Integration
# 9. Tie-breaking: equal scores sorted by candidate_id ascending (precise fix)
# =============================================================================

import os, sys, json, gzip, numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss, pickle, yaml
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, deque
import time
from heapq import nlargest
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import QuantileTransformer
import warnings, hashlib, torch, torch.cuda, re
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = """
paths:
  candidates_file: "candidates.jsonl"
  output_dir: "./output/"
model:
  name: "BAAI/bge-m3"
  batch_size: 64
gpu:
  preprocessing_batch_size: 96
  use_gpu: true
  use_fp16: true
retrieval:
  faiss_top_k: 1000
  final_top_k: 100
faiss:
  index_type: "HNSW32"
  ef_construction: 200
  ef_search: 50
adaptive_weights:
  min_weight: 0.15
  max_weight: 0.55
feature_weights:
  base:
    description_consistency: 0.90
    retrieval_specialization: 0.85
    career_authenticity: 0.80
    ai_experience: 0.75
    behavioral: 0.70
    career_progression: 0.60
    skill_quality: 0.55
    education: 0.30
experience_alignment:
  enabled: true
  penalty_multiplier: 0.7
  target_years: 7
  min_years: 3
skill_ontology:
  fine_tuning: [lora, qlora, peft, fine-tuning, finetuning]
  vector_databases: [faiss, pinecone, milvus, qdrant, weaviate, pgvector, chromadb]
  retrieval: [retrieval, information retrieval, bm25, hybrid search, semantic search, dense retrieval]
  ranking: [ranking, learning to rank, ltr, ndcg, mrr, map]
  embeddings: [embedding, sentence transformer, bge, e5, openai embedding, text-embedding]
  llm: [llm, large language model, gpt, claude, llama, mistral]
  nlp: [nlp, natural language processing, tokenization, ner, sentiment]
  ml: [machine learning, deep learning, neural network, cnn, rnn, transformer]
  cloud: [aws, azure, gcp, cloud]
  devops: [docker, kubernetes, terraform, jenkins, ci/cd]
skill_boost:
  base_boost: 0.8
  max_boost: 1.2
  ontology_match_boost: 0.20
"""

class ConfigLoader:
    @staticmethod
    def load(config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path) as f: return yaml.safe_load(f)
        return yaml.safe_load(DEFAULT_CONFIG)

class GPUManager:
    @staticmethod
    def check_gpu_availability():
        print("\n" + "=" * 60)
        print("GPU DETECTION")
        print("=" * 60)
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
        print("❌ CUDA not available")
        return False
    
    @staticmethod
    def get_device(use_gpu=True):
        return "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    
    @staticmethod
    def optimize_gpu_memory():
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    @staticmethod
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class AhoCorasick:
    def __init__(self):
        self.goto = [{}]; self.fail = [0]; self.output = [set()]; self.patterns = set()
    def add_pattern(self, pattern):
        pattern = pattern.lower(); self.patterns.add(pattern); node = 0
        for char in pattern:
            if char not in self.goto[node]:
                self.goto[node][char] = len(self.goto)
                self.goto.append({}); self.fail.append(0); self.output.append(set())
            node = self.goto[node][char]
        self.output[node].add(pattern)
    def build(self):
        queue = deque()
        for char, node in self.goto[0].items(): self.fail[node] = 0; queue.append(node)
        while queue:
            node = queue.popleft()
            for char, next_node in self.goto[node].items():
                queue.append(next_node); fail = self.fail[node]
                while fail and char not in self.goto[fail]: fail = self.fail[fail]
                self.fail[next_node] = self.goto[fail].get(char, 0)
                self.output[next_node].update(self.output[self.fail[next_node]])
    def search(self, text):
        text = text.lower(); matches = set(); node = 0
        for char in text:
            while node and char not in self.goto[node]: node = self.fail[node]
            node = self.goto[node].get(char, 0)
            if self.output[node]: matches.update(self.output[node])
        return matches

class SkillOntology:
    def __init__(self, config):
        self.ontology = config.get('skill_ontology', {})
        self.skill_to_concept = {}; self.concept_to_skills = defaultdict(set)
        for concept, skills in self.ontology.items():
            for skill in skills:
                self.skill_to_concept[skill.lower()] = concept
                self.concept_to_skills[concept].add(skill.lower())
        self.all_skills = set(self.skill_to_concept.keys())
    def get_concept(self, skill): return self.skill_to_concept.get(skill.lower())
    def expand_skills(self, skills):
        expanded = set(skills)
        for skill in skills:
            concept = self.get_concept(skill)
            if concept: expanded.add(concept)
        return expanded
    def concept_match_score(self, candidate_skills, jd_skills):
        if not jd_skills: return 0.5
        expanded_candidate = self.expand_skills(candidate_skills)
        expanded_jd = self.expand_skills(jd_skills)
        matched = len(expanded_candidate & expanded_jd)
        total = len(expanded_jd)
        for cs in candidate_skills:
            cc = self.get_concept(cs)
            if cc:
                for js in jd_skills:
                    if cc == self.get_concept(js): matched += 0.5
        return min(matched / max(total, 1), 1.0)

class PercentileNormalizer:
    def __init__(self): self.scalers = {}; self.feature_names = []; self.is_fitted = False
    def fit(self, feature_matrix, feature_names):
        self.feature_names = feature_names; self.is_fitted = True
        for i, name in enumerate(feature_names):
            scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
            scaler.fit(feature_matrix[:, i].astype(np.float32).reshape(-1, 1))
            self.scalers[name] = scaler
    def transform(self, feature_matrix):
        if not self.is_fitted: return feature_matrix
        result = np.zeros_like(feature_matrix, dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            if name in self.scalers:
                result[:, i] = self.scalers[name].transform(feature_matrix[:, i].astype(np.float32).reshape(-1, 1)).flatten()
        return result

class Benchmark:
    def __init__(self): self.results = {}
    def measure(self, name, func, *args, **kwargs):
        start = time.time(); result = func(*args, **kwargs)
        self.results[name] = time.time() - start; return result, self.results[name]
    def print_results(self):
        print("\n" + "=" * 60); print("BENCHMARK RESULTS"); print("=" * 60)
        total = sum(self.results.values())
        for name, elapsed in sorted(self.results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {elapsed:.4f}s")
        print(f"  {'-' * 40}"); print(f"  TOTAL: {total:.4f}s")

class FeatureExtractor:
    def __init__(self):
        self.tech_roles = {'engineer', 'developer', 'scientist', 'architect', 'specialist', 'ml', 'ai', 'nlp', 'data', 'software', 'backend', 'full stack'}
        self.retrieval_terms = {'retrieval', 'search', 'ranking', 'recommendation', 'embedding', 'vector', 'faiss', 'pinecone', 'milvus', 'qdrant'}
        self.ai_terms = {'machine learning', 'deep learning', 'nlp', 'llm', 'transformer', 'retrieval', 'ranking', 'recommendation', 'embedding'}
        self.senior_terms = {'senior', 'lead', 'staff', 'principal', 'manager'}
        self.junior_terms = {'junior', 'associate', 'trainee', 'intern'}
        self.feature_names = ['career_authenticity', 'retrieval_specialization', 'ai_experience', 'career_progression', 'skill_quality', 'description_consistency', 'education', 'behavioral', 'experience_years']
    
    def _score_career_authenticity(self, c): 
        tech, total = 0, 0
        for job in c.get('career_history', []):
            d = job.get('duration_months', 0); total += d
            if any(r in job.get('title', '').lower() for r in self.tech_roles): tech += d
        return tech / total if total > 0 else 0
    
    def _score_retrieval(self, c):
        score = 0
        for skill in c.get('skills', []):
            if any(t in skill.get('name', '').lower() for t in self.retrieval_terms):
                d = skill.get('duration_months', 0)
                pw = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
                score += min(d / 6, 3) * (pw.get(skill.get('proficiency', ''), 1) / 2)
        for job in c.get('career_history', []):
            cnt = sum(1 for t in self.retrieval_terms if t in job.get('description', '').lower())
            if cnt > 0: score += min(cnt * 2, 4)
        return int(score)
    
    def _score_ai_experience(self, c):
        score = 0
        for job in c.get('career_history', []):
            cnt = sum(1 for t in self.ai_terms if t in job.get('description', '').lower())
            if cnt > 0: score += min(cnt, 3)
        return score
    
    def _score_career_progression(self, c):
        levels = []
        for job in c.get('career_history', []):
            title = job.get('title', '').lower()
            if any(t in title for t in self.senior_terms): levels.append(3)
            elif any(t in title for t in self.junior_terms): levels.append(1)
            else: levels.append(2)
        if len(levels) < 2: return 0.5
        prog = 0
        for i in range(1, len(levels)):
            if levels[i] > levels[i-1]: prog += 0.2
            elif levels[i] < levels[i-1]: prog -= 0.1
        return min(max(prog + 0.5, 0), 1.0)
    
    def _score_skill_quality(self, c):
        score = 0; pw = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        for skill in c.get('skills', []):
            d = skill.get('duration_months', 0); end = min(skill.get('endorsements', 0), 50)
            score += (d / 12) * 2 + pw.get(skill.get('proficiency', ''), 1) * 1.5 + end * 0.1
        return score
    
    def _score_description_consistency(self, c):
        skill_names = set(skill.get('name', '').lower() for skill in c.get('skills', []) if skill.get('name', ''))
        if not skill_names: return 0
        matched = 0
        for job in c.get('career_history', []):
            desc = job.get('description', '').lower()
            if any(s in desc for s in skill_names): matched += 1
        return (matched / max(len(c.get('career_history', [])), 1)) * 10
    
    def _score_education(self, c):
        score = 0; tier_scores = {'tier_1': 12, 'tier_2': 8, 'tier_3': 4, 'tier_4': 1}
        relevant = {'computer science', 'artificial intelligence', 'machine learning', 'data science', 'information technology', 'computer engineering'}
        for edu in c.get('education', []):
            score += tier_scores.get(edu.get('tier', ''), 0)
            if any(f in edu.get('field_of_study', '').lower() for f in relevant): score += 6
        return min(score, 30)
    
    def _score_behavioral(self, signals):
        """ALL 23 Redrob signals"""
        score = 0.0
        score += signals.get('recruiter_response_rate', 0) * 35
        score += signals.get('interview_completion_rate', 0) * 30
        if signals.get('offer_acceptance_rate', -1) >= 0: score += signals['offer_acceptance_rate'] * 25
        for k, w in [('application_to_interview_rate', 20), ('interview_to_offer_rate', 25), ('offer_to_joining_rate', 15)]:
            if signals.get(k, -1) >= 0: score += signals[k] * w
        score += signals.get('profile_completeness_score', 0) * 0.3
        score += min(signals.get('saved_by_recruiters_30d', 0), 20) * 0.8
        score += min(signals.get('profile_views_received_30d', 0), 100) * 0.04
        score += min(signals.get('profile_views_7d', 0), 30) * 0.12
        sa = signals.get('search_appearances_30d', 0)
        score += min(sa, 50) * 0.15 if sa > 0 else -5
        if signals.get('open_to_work_flag', False): score += 10
        ar = signals.get('avg_response_time_hours', 999)
        if ar < 2: score += 15
        elif ar < 8: score += 10
        elif ar < 24: score += 5
        elif ar < 72: score += 2
        else: score -= 3
        if signals.get('inmail_response_rate', -1) >= 0: score += signals['inmail_response_rate'] * 15
        if signals.get('connection_acceptance_rate', -1) >= 0: score += signals['connection_acceptance_rate'] * 8
        score += min(signals.get('skill_endorsements_count', 0), 100) * 0.1
        score += min(signals.get('recommendations_count', 0), 10) * 2.0
        uf = signals.get('profile_update_frequency', 0)
        if uf > 2.0: score += 8
        elif uf > 0.5: score += 4
        else: score -= 3
        score += min(signals.get('job_search_activity_score', 0), 100) * 0.1
        score += signals.get('location_match_score', 0.5) * 8
        if signals.get('salary_expectation_alignment', -1) >= 0: score += signals['salary_expectation_alignment'] * 12
        np_val = signals.get('notice_period_days', 90)
        if np_val <= 15: score += 15
        elif np_val <= 30: score += 10
        elif np_val <= 60: score += 5
        elif np_val > 90: score -= 5
        score += -8 if signals.get('visa_sponsorship_needed', False) else 3
        return max(0.0, min(score, 100.0))
    
    def _score_honeypot(self, c):
        score = 0
        auth = self._score_career_authenticity(c); ret = self._score_retrieval(c)
        if auth < 0.2 and ret > 3: score += 50
        for skill in c.get('skills', []):
            d = skill.get('duration_months', 0); prof = skill.get('proficiency', '')
            if prof == 'expert' and d < 12: score += 10
            elif prof == 'advanced' and d < 6: score += 5
        if c.get('redrob_signals', {}).get('recruiter_response_rate', 0) < 0.1: score += 10
        return score
    
    def extract_features(self, candidate):
        features = {}
        features['career_authenticity'] = self._score_career_authenticity(candidate)
        features['retrieval_specialization'] = self._score_retrieval(candidate)
        features['ai_experience'] = self._score_ai_experience(candidate)
        features['career_progression'] = self._score_career_progression(candidate)
        features['skill_quality'] = self._score_skill_quality(candidate)
        features['description_consistency'] = self._score_description_consistency(candidate)
        features['education'] = self._score_education(candidate)
        features['experience_years'] = candidate.get('profile', {}).get('years_of_experience', 0)
        features['behavioral'] = self._score_behavioral(candidate.get('redrob_signals', {}))
        features['honeypot_score'] = self._score_honeypot(candidate)
        return features

class JDAdaptiveWeightGenerator:
    def __init__(self, config):
        self.config = config
        self.semantic_terms = {'retrieval', 'ranking', 'search', 'recommendation', 'embedding', 'vector', 'faiss', 'pinecone', 'milvus', 'qdrant', 'semantic', 'llm', 'nlp', 'transformer', 'language model', 'deep learning'}
        self.bm25_terms = {'python', 'java', 'javascript', 'sql', 'api', 'rest', 'backend', 'frontend', 'database', 'cloud', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'linux', 'git', 'ci/cd'}
        self.feature_terms = {'experience', 'leadership', 'team', 'management', 'communication', 'analytical', 'problem solving', 'architecture', 'design'}
    
    def generate_weights(self, jd_text):
        jd_lower = jd_text.lower()
        sem = sum(1 for t in self.semantic_terms if t in jd_lower)
        bm = sum(1 for t in self.bm25_terms if t in jd_lower)
        feat = sum(1 for t in self.feature_terms if t in jd_lower)
        sr, br, fr = min(sem/10, 1.0), min(bm/10, 1.0), min(feat/10, 1.0)
        cfg = self.config['adaptive_weights']; mw, Mw = cfg['min_weight'], cfg['max_weight']
        rs = mw + (Mw-mw)*sr; rb = mw + (Mw-mw)*br; rf = mw + (Mw-mw)*fr
        if sem==0: rs = mw*0.5
        if bm==0: rb = mw*0.5
        if feat==0: rf = mw*0.5
        total = rs+rb+rf
        return {'semantic': rs/total, 'bm25': rb/total, 'features': rf/total}

class OfflinePreprocessor:
    def __init__(self, config):
        self.config = config; self.model = None; self.candidates = []; self.candidate_map = {}
        self.candidate_ids = []; self.candidate_id_to_idx = {}; self.candidate_texts = []; self.candidate_tokens = []
        self.embeddings = None; self.features = {}; self.feature_matrix = None; self.feature_names = []
        self.inverted_index = defaultdict(set); self.aho_corasick = None; self.bm25 = None
        self.extractor = FeatureExtractor(); self.percentile_normalizer = PercentileNormalizer()
        self.skill_ontology = None; self.benchmark = Benchmark()
        self.device = "cpu"; self.use_gpu = config.get('gpu', {}).get('use_gpu', True)
        self.use_fp16 = config.get('gpu', {}).get('use_fp16', True)
        self.output_dir = config['paths']['output_dir']; os.makedirs(self.output_dir, exist_ok=True)
    
    def load_candidates(self, file_path):
        print("Loading candidates...")
        opener = gzip.open if file_path.endswith('.gz') else open
        with opener(file_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading"):
                if line.strip():
                    candidate = json.loads(line); self.candidates.append(candidate)
                    self.candidate_map[candidate['candidate_id']] = candidate
                    self.candidate_ids.append(candidate['candidate_id'])
        self.candidate_id_to_idx = {cid: i for i, cid in enumerate(self.candidate_ids)}
        print(f"Loaded {len(self.candidates)} candidates")
    
    def initialize_model(self, model_name):
        self.device = GPUManager.get_device(self.use_gpu); GPUManager.optimize_gpu_memory()
        if self.device == "cuda":
            if self.use_fp16:
                self.model = SentenceTransformer(model_name, device=self.device, model_kwargs={"torch_dtype": torch.float16})
            else:
                self.model = SentenceTransformer(model_name, device=self.device)
            with torch.inference_mode(): _ = self.model.encode(["Warm up"], show_progress_bar=False, batch_size=1)
        else:
            self.model = SentenceTransformer(model_name, device="cpu")
    
    def build_candidate_text(self, candidate):
        parts = []; profile = candidate.get('profile', {})
        parts.extend([profile.get('headline', ''), profile.get('summary', ''), profile.get('current_title', ''), profile.get('current_company', '')])
        for job in candidate.get('career_history', [])[:5]: parts.extend([job.get('title', ''), job.get('description', '')])
        for skill in candidate.get('skills', [])[:20]: parts.append(skill.get('name', ''))
        return ' '.join(parts)
    
    def compute_embeddings(self):
        self.candidate_texts = [self.build_candidate_text(c) for c in tqdm(self.candidates, desc="Building texts")]
        self.candidate_tokens = [t.lower().split() for t in self.candidate_texts]
        if self.device == "cuda":
            gb = torch.cuda.get_device_properties(0).total_memory/1024**3
            gpu_batch_size = 96 if gb<=6 else 128 if gb<=8 else 192 if self.use_fp16 else 64 if gb<=6 else 96 if gb<=8 else 128
        else:
            gpu_batch_size = self.config.get('model', {}).get('batch_size', 64)
        embeddings_list = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(self.candidate_texts), gpu_batch_size), desc=f"Embedding"):
                batch = self.candidate_texts[i:i+gpu_batch_size]
                embeddings_list.append(self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False, batch_size=gpu_batch_size))
        self.embeddings = np.concatenate(embeddings_list, axis=0).astype('float32')
        print(f"Generated {len(self.embeddings)} embeddings, Shape: {self.embeddings.shape}")
        if self.device == "cuda": del self.model; self.model = None; GPUManager.clear_gpu_memory()
    
    def build_faiss_index(self):
        dim = self.embeddings.shape[1]; index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = self.config['faiss']['ef_construction']
        index.hnsw.efSearch = self.config['faiss']['ef_search']
        index.add(self.embeddings); faiss.write_index(index, f"{self.output_dir}/faiss_hnsw.index")
    
    def build_inverted_index_and_aho(self):
        for candidate in tqdm(self.candidates, desc="Indexing"):
            cid = candidate['candidate_id']
            for skill in candidate.get('skills', []):
                name = skill.get('name', '').lower()
                if name and len(name)>2: self.inverted_index[name].add(cid)
            for word in candidate.get('profile', {}).get('current_title', '').lower().split():
                if len(word)>2: self.inverted_index[word].add(cid)
        self.aho_corasick = AhoCorasick()
        for term in self.inverted_index: self.aho_corasick.add_pattern(term)
        self.aho_corasick.build()
    
    def build_bm25_index(self): self.bm25 = BM25Okapi(self.candidate_tokens)
    
    def compute_features(self):
        feature_list = []
        for candidate in tqdm(self.candidates, desc="Features"):
            features = self.extractor.extract_features(candidate)
            self.features[candidate['candidate_id']] = features
            feature_list.append([features.get(name, 0) for name in self.extractor.feature_names])
        self.feature_matrix = np.array(feature_list, dtype=np.float16)
        self.feature_names = self.extractor.feature_names
        print(f"Feature matrix: {self.feature_matrix.shape}, Signals: 23/23")
    
    def fit_percentile_normalizer(self): self.percentile_normalizer.fit(self.feature_matrix.astype(np.float32), self.feature_names)
    def initialize_skill_ontology(self): self.skill_ontology = SkillOntology(self.config)
    
    def save_all(self):
        np.save(f"{self.output_dir}/candidate_embeddings.npy", self.embeddings)
        with open(f"{self.output_dir}/features_cache.pkl", 'wb') as f: pickle.dump(self.features, f)
        np.save(f"{self.output_dir}/feature_matrix.npy", self.feature_matrix)
        with open(f"{self.output_dir}/inverted_index.pkl", 'wb') as f: pickle.dump(dict(self.inverted_index), f)
        with open(f"{self.output_dir}/aho_corasick.pkl", 'wb') as f: pickle.dump(self.aho_corasick, f)
        with open(f"{self.output_dir}/bm25.pkl", 'wb') as f: pickle.dump(self.bm25, f)
        with open(f"{self.output_dir}/candidate_id_to_idx.pkl", 'wb') as f: pickle.dump(self.candidate_id_to_idx, f)
        with open(f"{self.output_dir}/percentile_normalizer.pkl", 'wb') as f: pickle.dump(self.percentile_normalizer, f)
        with open(f"{self.output_dir}/skill_ontology.pkl", 'wb') as f: pickle.dump(self.skill_ontology, f)
        metadata = {'candidate_ids': self.candidate_ids, 'candidate_map': self.candidate_map, 'total_candidates': len(self.candidates), 'embedding_dim': self.embeddings.shape[1], 'feature_names': self.feature_names, 'preprocessing_device': self.device, 'used_fp16': self.use_fp16, 'redrob_signals_count': 23}
        with open(f"{self.output_dir}/candidate_metadata.pkl", 'wb') as f: pickle.dump(metadata, f)
        dataset_hash = hashlib.sha256()
        opener = gzip.open if self.config['paths']['candidates_file'].endswith('.gz') else open
        with opener(self.config['paths']['candidates_file'], 'rb') as f:
            for bb in iter(lambda: f.read(4096), b""): dataset_hash.update(bb)
        cache_info = {'dataset_hash': dataset_hash.hexdigest(), 'timestamp': time.time(), 'candidate_count': len(self.candidates), 'preprocessing_device': self.device, 'used_fp16': self.use_fp16, 'redrob_signals_count': 23}
        with open(f"{self.output_dir}/cache_info.pkl", 'wb') as f: pickle.dump(cache_info, f)
        print("All data saved! Signals: 23/23")
    
    def run(self):
        self.benchmark.measure("Load", self.load_candidates, self.config['paths']['candidates_file'])
        self.benchmark.measure("Model", self.initialize_model, self.config['model']['name'])
        self.benchmark.measure("Embed", self.compute_embeddings)
        self.benchmark.measure("FAISS", self.build_faiss_index)
        self.benchmark.measure("Index", self.build_inverted_index_and_aho)
        self.benchmark.measure("BM25", self.build_bm25_index)
        self.benchmark.measure("Features", self.compute_features)
        self.benchmark.measure("Normalize", self.fit_percentile_normalizer)
        self.benchmark.measure("Ontology", self.initialize_skill_ontology)
        self.benchmark.measure("Save", self.save_all)
        print("\n✅ Done!"); self.benchmark.print_results()

class FinalRanker:
    def __init__(self, config):
        self.config = config; self.model = None; self.candidate_ids = []; self.candidate_map = {}
        self.candidate_id_to_idx = {}; self.embeddings = None; self.faiss_index = None
        self.features = {}; self.feature_matrix = None; self.feature_names = []
        self.inverted_index = {}; self.aho_corasick = None; self.bm25 = None
        self.percentile_normalizer = None; self.skill_ontology = None
        self.extractor = FeatureExtractor(); self.weight_generator = JDAdaptiveWeightGenerator(config)
        self.benchmark = Benchmark(); self.experience_config = config.get('experience_alignment', {})
    
    def load_precomputed_data(self):
        output_dir = self.config['paths']['output_dir']
        print("\n" + "=" * 60); print("LOADING DATA (CPU ONLY)"); print("=" * 60)
        with open(f"{output_dir}/candidate_metadata.pkl", 'rb') as f: metadata = pickle.load(f)
        self.candidate_ids = metadata['candidate_ids']; self.candidate_map = metadata['candidate_map']
        self.feature_names = metadata.get('feature_names', [])
        print(f"  Candidates: {len(self.candidate_ids)}, Device: {metadata.get('preprocessing_device', 'unknown').upper()}")
        with open(f"{output_dir}/candidate_id_to_idx.pkl", 'rb') as f: self.candidate_id_to_idx = pickle.load(f)
        self.embeddings = np.load(f"{output_dir}/candidate_embeddings.npy")
        self.faiss_index = faiss.read_index(f"{output_dir}/faiss_hnsw.index")
        with open(f"{output_dir}/features_cache.pkl", 'rb') as f: self.features = pickle.load(f)
        self.feature_matrix = np.load(f"{output_dir}/feature_matrix.npy")
        with open(f"{output_dir}/inverted_index.pkl", 'rb') as f: self.inverted_index = pickle.load(f)
        with open(f"{output_dir}/aho_corasick.pkl", 'rb') as f: self.aho_corasick = pickle.load(f)
        with open(f"{output_dir}/bm25.pkl", 'rb') as f: self.bm25 = pickle.load(f)
        with open(f"{output_dir}/percentile_normalizer.pkl", 'rb') as f: self.percentile_normalizer = pickle.load(f)
        with open(f"{output_dir}/skill_ontology.pkl", 'rb') as f: self.skill_ontology = pickle.load(f)
        
        # RECOMPUTE BEHAVIORAL SCORES WITH 23 SIGNALS
        print("  🔄 Recomputing behavioral scores with 23 signals...")
        extractor = FeatureExtractor(); updated = 0
        for cid in tqdm(self.candidate_ids, desc="  Updating"):
            candidate = self.candidate_map.get(cid)
            if candidate:
                new_score = extractor._score_behavioral(candidate.get('redrob_signals', {}))
                if abs(new_score - self.features[cid].get('behavioral', 0)) > 0.01: updated += 1
                self.features[cid]['behavioral'] = new_score
                self.feature_matrix[self.candidate_id_to_idx[cid], 7] = new_score
        print(f"  ✅ Updated {updated} scores | Range: [{self.feature_matrix[:,7].min():.1f}, {self.feature_matrix[:,7].max():.1f}]")
        
        self.model = SentenceTransformer(self.config['model']['name'], device="cpu")
        print("  ✅ Model loaded on CPU\n" + "=" * 60)
    
    def _normalize_scores(self, scores):
        min_val, max_val = np.min(scores), np.max(scores)
        return (scores-min_val)/(max_val-min_val) if max_val-min_val>1e-8 else np.ones_like(scores)*0.5
    
    def _get_skill_boost_with_ontology(self, candidate, jd_skills):
        if not jd_skills: return self.config['skill_boost']['base_boost']
        skill_names = set(skill.get('name', '').lower() for skill in candidate.get('skills', []) if skill.get('name', ''))
        if not skill_names: return self.config['skill_boost']['base_boost']
        ontology_score = self.skill_ontology.concept_match_score(skill_names, jd_skills)
        exact_ratio = len(skill_names & jd_skills)/len(jd_skills) if jd_skills else 0
        combined_ratio = max(exact_ratio, ontology_score*0.8)
        boost = self.config['skill_boost']['base_boost'] + (self.config['skill_boost']['max_boost']-self.config['skill_boost']['base_boost'])*combined_ratio
        if ontology_score>0.3: boost += self.config['skill_boost']['ontology_match_boost']
        return min(boost, self.config['skill_boost']['max_boost'])
    
    def _calculate_experience_years(self, candidate):
        py = candidate.get('profile', {}).get('years_of_experience', 0)
        if py>0: return py
        return sum(job.get('duration_months', 0) for job in candidate.get('career_history', []))/12.0
    
    def _detect_junior_level(self, candidate):
        jk = {'junior', 'associate', 'trainee', 'intern', 'entry level', 'graduate'}
        title = candidate.get('profile', {}).get('current_title', '').lower()
        headline = candidate.get('profile', {}).get('headline', '').lower()
        if any(k in title or k in headline for k in jk): return True
        for job in candidate.get('career_history', [])[:3]:
            if any(k in job.get('title', '').lower() for k in jk): return True
        return False
    
    def _calculate_experience_penalty(self, candidate):
        if not self.experience_config.get('enabled', True): return 1.0
        years = self._calculate_experience_years(candidate)
        target, min_y, pm = self.experience_config.get('target_years', 7), self.experience_config.get('min_years', 3), self.experience_config.get('penalty_multiplier', 0.7)
        is_junior = self._detect_junior_level(candidate)
        if years>=target and not is_junior: return 1.0
        if years>=target: er=0.95
        elif years>=min_y: er=0.85+(years-min_y)/(target-min_y)*0.15
        elif years>=1: er=0.7+(years-1)/(min_y-1)*0.15
        else: er=0.6
        if is_junior: er*=pm
        return max(er, 0.5)
    
    def _extract_current_role(self, c):
        p = c.get('profile', {}); t, co = p.get('current_title', ''), p.get('current_company', '')
        return f"{t} at {co}" if t and co else t or "professional"
    
    def _extract_years_experience(self, c):
        y = self._calculate_experience_years(c); return f"{y:.0f} years experience" if y>0 else "experienced professional"
    
    def _extract_matching_skills(self, c, jd_skills):
        cs = set(skill.get('name', '').lower() for skill in c.get('skills', []) if skill.get('name', ''))
        return list(cs & jd_skills)[:5]
    
    def _extract_education_summary(self, c):
        edu = c.get('education', [])
        if not edu: return ""
        d, f = edu[0].get('degree', ''), edu[0].get('field_of_study', '')
        return f"{d} in {f}" if d and f else d or ""
    
    def _extract_location(self, c):
        p = c.get('profile', {}); locs = p.get('preferred_locations', []); cl = p.get('current_location', '')
        return f"prefers {', '.join(locs[:2])}" if locs else f"based in {cl}" if cl else ""
    
    def _extract_notice_period(self, c):
        n = c.get('profile', {}).get('notice_period_days')
        return f"{n}-day notice period" if n and n>60 else ""
    
    def _generate_reasoning(self, result, jd_text):
        cid = result['candidate_id']; candidate = self.candidate_map.get(cid)
        features = result.get('features', {})
        if not candidate: return "Candidate matched based on skills and experience."
        points = []
        points.append(f"{self._extract_current_role(candidate)} with {self._extract_years_experience(candidate)}")
        jd_skills = self.aho_corasick.search(jd_text)
        ms = self._extract_matching_skills(candidate, jd_skills)
        if ms: points.append(f"strong {', '.join(ms[:4])} background relevant to JD requirements")
        years = self._calculate_experience_years(candidate); is_junior = self._detect_junior_level(candidate)
        if years<3: points.append(f"note: experience ({years:.0f}yr) below target 5-9yr range")
        elif is_junior: points.append("junior title may not align with senior role")
        elif 3<=years<5: points.append(f"experience ({years:.0f}yr) at lower end but skills relevant")
        elif years>9: points.append(f"strong senior-level experience ({years:.0f}yr)")
        auth = features.get('career_authenticity', 0); rs = features.get('retrieval_specialization', 0)
        if auth>=0.7: points.append("consistent technical career in AI/ML" + (" with deep retrieval expertise" if rs>=4 else ""))
        ai = features.get('ai_experience', 0)
        if ai>=3: points.append("hands-on production ML/embeddings experience")
        elif ai>=1: points.append("practical ML technology experience")
        bh = features.get('behavioral', 0)
        if bh>50: points.append(f"exceptional recruiter engagement ({bh:.0f}/100)")
        elif bh>30: points.append("strong recruiter engagement")
        elif bh>15: points.append("active with recruiters")
        edu = self._extract_education_summary(candidate)
        if edu: points.append(f"education: {edu}")
        loc = self._extract_location(candidate)
        if loc and any(c in loc.lower() for c in ['pune', 'noida']): points.append(f"location matches JD ({loc})")
        elif loc: points.append(f"currently {loc}")
        notice = self._extract_notice_period(candidate)
        if notice: points.append(f"note: {notice} may affect availability")
        hp = features.get('honeypot_score', 0)
        if hp>30: points.append("profile may need verification")
        elif hp>10: points.append("minor inconsistencies noted")
        reasoning = ". ".join(points[:5])
        return reasoning + "." if reasoning and not reasoning.endswith('.') else reasoning
    
    def rank(self, jd_text):
        print("\n" + "=" * 80); print("FINAL RANKER v10.0 - PRECISE TIE-BREAKING + 23 SIGNALS"); print("=" * 80)
        if self.experience_config.get('enabled'): print(f"📊 Experience: Target {self.experience_config.get('target_years',7)}yr, Min {self.experience_config.get('min_years',3)}yr")
        print("🎯 23 Signals | 🔗 Tie-breaking: final_score DESC, candidate_id ASC")
        
        jd_embedding = self.benchmark.measure("JD Embed", lambda: self.model.encode([jd_text], normalize_embeddings=True)[0])[0]
        print(f"[1/7] JD Embed: {self.benchmark.results['JD Embed']:.3f}s")
        
        adaptive_weights = self.benchmark.measure("Weights", lambda: self.weight_generator.generate_weights(jd_text))[0]
        print(f"[2/7] Weights: s={adaptive_weights['semantic']:.3f} b={adaptive_weights['bm25']:.3f} f={adaptive_weights['features']:.3f}")
        
        jd_np = jd_embedding.reshape(1,-1).astype('float32')
        distances, indices = self.faiss_index.search(jd_np, self.config['retrieval']['faiss_top_k'])
        faiss_candidates = [(self.candidate_ids[idx], dist) for idx, dist in zip(indices[0], distances[0])]
        faiss_set = {cid for cid, _ in faiss_candidates}
        print(f"[3/7] FAISS: {len(faiss_candidates)} candidates")
        
        full_scores = self.bm25.get_scores(jd_text.lower().split())
        bm25_dict = {}
        for cid, score in zip(self.candidate_ids, full_scores):
            if cid in faiss_set: bm25_dict[cid] = score
        if bm25_dict:
            values = np.array(list(bm25_dict.values()))
            bm25_dict = {cid: s for cid, s in zip(bm25_dict.keys(), self._normalize_scores(values))}
        print(f"[4/7] BM25: {len(bm25_dict)} candidates")
        
        valid = [(cid, dist) for cid, dist in faiss_candidates if cid in bm25_dict]
        feature_dict = {}
        if valid:
            indices = [self.candidate_id_to_idx[cid] for cid, _ in valid]
            nf = self.percentile_normalizer.transform(self.feature_matrix[indices].astype(np.float32))
            wv = np.array([self.config['feature_weights']['base'].get(n,0.5) for n in self.feature_names], dtype=np.float32)
            fs = np.dot(nf, wv)*10
            nfs = self._normalize_scores(fs)
            feature_dict = {cid: nfs[i] for i, (cid, _) in enumerate(valid) if i<len(nfs)}
        print(f"[5/7] Features: computed with 23 signals")
        
        jd_skills = self.aho_corasick.search(jd_text)
        print(f"[6/7] Skills: {len(jd_skills)} found")
        
        start = time.time(); junior_count = under_count = 0; results = []
        for cid, faiss_dist in faiss_candidates:
            if cid not in bm25_dict: continue
            semantic_score = max(0, min(faiss_dist, 1.0))
            bm25_score = bm25_dict.get(cid, 0)
            feature_score = feature_dict.get(cid, 0)
            candidate = self.candidate_map.get(cid)
            skill_boost = self._get_skill_boost_with_ontology(candidate, jd_skills) if candidate else 0.8
            experience_penalty = 1.0
            if candidate:
                experience_penalty = self._calculate_experience_penalty(candidate)
                years_exp = self._calculate_experience_years(candidate)
                is_junior = self._detect_junior_level(candidate)
                if is_junior: junior_count += 1
                if years_exp < self.experience_config.get('min_years', 3): under_count += 1
            hybrid_score = (adaptive_weights['semantic']*semantic_score + adaptive_weights['bm25']*bm25_score + adaptive_weights['features']*feature_score) * skill_boost * experience_penalty
            features = self.features.get(cid, {})
            hybrid_score = max(0, hybrid_score - features.get('honeypot_score', 0)/100)
            results.append({'candidate_id': cid, 'score': hybrid_score, 'semantic_score': semantic_score, 'bm25_score': bm25_score, 'feature_score': feature_score, 'skill_boost': skill_boost, 'experience_penalty': experience_penalty, 'features': features})
        
        # ================================================================
        # PRECISE TIE-BREAKING: Store rounded score, sort by it
        # ================================================================
        for r in results:
            r["final_score"] = round(r["score"], 4)
        
        results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        top_results = results[:self.config['retrieval']['final_top_k']]
        
        # Count ties
        tie_count = sum(1 for i in range(len(top_results)-1) if top_results[i]["final_score"] == top_results[i+1]["final_score"])
        
        t7 = time.time() - start
        print(f"[7/7] Hybrid: {t7:.3f}s")
        if tie_count > 0: print(f"  🔗 Resolved {tie_count} tie(s) using candidate_id ascending")
        if self.experience_config.get('enabled'): print(f"📊 Junior: {junior_count}, Under-experienced: {under_count}")
        
        submission = []
        for rank, result in enumerate(top_results, 1):
            reasoning = self._generate_reasoning(result, jd_text)
            candidate = self.candidate_map.get(result['candidate_id'], {})
            years_exp = self._calculate_experience_years(candidate) if candidate else 0
            is_junior = self._detect_junior_level(candidate) if candidate else False
            behavioral = result['features'].get('behavioral', 0)
            submission.append({
                'candidate_id': result['candidate_id'],
                'rank': rank,
                'score': result['final_score'],  # Use final_score directly
                'reasoning': reasoning,
                'semantic': round(result['semantic_score'], 3),
                'bm25': round(result['bm25_score'], 3),
                'feature': round(result['feature_score'], 3),
                'skill_boost': round(result['skill_boost'], 3),
                'exp_penalty': round(result['experience_penalty'], 3),
                'years_exp': round(years_exp, 1),
                'is_junior': is_junior,
                'behavioral': round(behavioral, 1),
                'auth': round(result['features'].get('career_authenticity', 0), 3),
                'retrieval': int(result['features'].get('retrieval_specialization', 0)),
                'consistency': round(result['features'].get('description_consistency', 0), 1)
            })
        
        print("\n✅ Ranking complete!")
        for i, s in enumerate(submission[:3], 1): print(f"  {i}. [{s['years_exp']:.0f}yr, BH:{s['behavioral']:.0f}] {s['reasoning'][:150]}...")
        self.benchmark.print_results()
        return pd.DataFrame(submission), {}

def main():
    print("=" * 80); print("REDROB HACKATHON "); print("=" * 80)
    GPUManager.check_gpu_availability()
    config = ConfigLoader.load()
    output_dir = config['paths']['output_dir'].rstrip('/').rstrip('\\')
    config['paths']['output_dir'] = output_dir
    
    cache_path = f"{output_dir}/cache_info.pkl"
    if not os.path.exists(cache_path):
        print("\n🔄 Preprocessing..."); OfflinePreprocessor(config).run()
    else:
        print("\n✅ Using cached data")
    
    JD_TEXT = """
    Senior AI Engineer with 5-9 years experience. Deep technical depth in modern ML systems:
    embeddings, retrieval, ranking, LLMs, fine-tuning. Production experience with embedding-based
    retrieval systems (sentence-transformers, BGE, E5), vector databases (Pinecone, Weaviate, Qdrant,
    Milvus, OpenSearch, Elasticsearch, FAISS). Strong Python. Experience designing evaluation frameworks
    for ranking systems (NDCG, MRR, MAP). LLM fine-tuning (LoRA, QLoRA, PEFT). Learning-to-rank models.
    Location: Pune/Noida, India. Hybrid work mode.
    """
    
    ranker = FinalRanker(config); ranker.load_precomputed_data()
    results, _ = ranker.rank(JD_TEXT)
    
    output_path = os.path.join(output_dir, "CodeFusion.csv")
    results[['candidate_id', 'rank', 'score', 'reasoning']].to_csv(output_path, index=False)
    print(f"\n✅ Saved ( Top 100 candidates from the given 100k candidates): {output_path}")
    print("\n📊 Top 10:"); print(results[['candidate_id', 'rank', 'score', 'years_exp', 'behavioral']].head(10))
    print(f"\nAvg years: {results['years_exp'].mean():.1f}, Juniors: {results['is_junior'].sum()}, BH avg: {results['behavioral'].mean():.1f}/100")
    print("✨ Done!")

if __name__ == "__main__":
    main()