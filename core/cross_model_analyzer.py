"""
Cross-Model Analysis Engine
Specialized system for comparing and analyzing relationships between different models
using the hybrid graph-RAG architecture.
"""
import logging
from typing import Dict, List, Any, Tuple
import time

from .hybrid_agent_runner import HybridAgentRunner

logger = logging.getLogger(__name__)


class CrossModelAnalyzer:
    """Specialized analyzer for cross-model comparisons and relationships."""
    
    def __init__(self, hybrid_runner: HybridAgentRunner):
        self.hybrid_runner = hybrid_runner
        self.comparison_cache = {}
    
    def comprehensive_model_comparison(self, model1: str, model2: str) -> Dict:
        """Perform comprehensive comparison between two models."""
        logger.info(f"🔬 Comprehensive model comparison: {model1} vs {model2}")
        
        # Check cache first
        cache_key = f"{model1}_vs_{model2}"
        if cache_key in self.comparison_cache:
            logger.info("📚 Using cached comparison result")
            return self.comparison_cache[cache_key]
        
        start_time = time.time()
        
        # Multi-dimensional comparison queries
        comparison_dimensions = [
            self._analyze_performance_comparison(model1, model2),
            self._analyze_methodology_comparison(model1, model2),
            self._analyze_dataset_usage(model1, model2),
            self._analyze_capability_differences(model1, model2),
            self._analyze_historical_relationship(model1, model2)
        ]
        
        # Synthesize comprehensive comparison
        synthesis = self._synthesize_model_comparison(model1, model2, comparison_dimensions)
        
        processing_time = time.time() - start_time
        
        result = {
            'model1': model1,
            'model2': model2,
            'performance_comparison': comparison_dimensions[0],
            'methodology_comparison': comparison_dimensions[1],
            'dataset_analysis': comparison_dimensions[2],
            'capability_analysis': comparison_dimensions[3],
            'historical_relationship': comparison_dimensions[4],
            'comprehensive_synthesis': synthesis,
            'processing_time': processing_time,
            'analysis_timestamp': time.time()
        }
        
        # Cache result
        self.comparison_cache[cache_key] = result
        
        logger.info(f"✅ Model comparison completed in {processing_time:.1f} seconds")
        return result
    
    def _analyze_performance_comparison(self, model1: str, model2: str) -> Dict:
        """Analyze performance differences between models."""
        query = f"Compare the performance metrics, accuracy, and evaluation results between {model1} and {model2}. Include specific numbers, benchmarks, and performance data."
        
        try:
            result = self.hybrid_runner.query(query)
            return {
                'dimension': 'performance',
                'analysis': result.answer,
                'confidence': result.confidence,
                'sources': result.citations
            }
        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            return {'dimension': 'performance', 'analysis': 'Analysis failed', 'confidence': 0.0}
    
    def _analyze_methodology_comparison(self, model1: str, model2: str) -> Dict:
        """Analyze methodological differences between models."""
        query = f"Compare the methodologies, algorithms, architectures, and technical approaches used in {model1} versus {model2}. What are the key technical differences?"
        
        try:
            result = self.hybrid_runner.query(query)
            return {
                'dimension': 'methodology',
                'analysis': result.answer,
                'confidence': result.confidence,
                'sources': result.citations
            }
        except Exception as e:
            logger.error(f"Methodology comparison failed: {e}")
            return {'dimension': 'methodology', 'analysis': 'Analysis failed', 'confidence': 0.0}
    
    def _analyze_dataset_usage(self, model1: str, model2: str) -> Dict:
        """Analyze dataset usage and data requirements."""
        query = f"Compare the datasets, data requirements, and data processing approaches used by {model1} and {model2}. Do they share datasets or use different data sources?"
        
        try:
            result = self.hybrid_runner.query(query)
            return {
                'dimension': 'datasets',
                'analysis': result.answer,
                'confidence': result.confidence,
                'sources': result.citations
            }
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            return {'dimension': 'datasets', 'analysis': 'Analysis failed', 'confidence': 0.0}
    
    def _analyze_capability_differences(self, model1: str, model2: str) -> Dict:
        """Analyze capability and feature differences."""
        query = f"What are the different capabilities, features, and use cases of {model1} compared to {model2}? What can each model do that the other cannot?"
        
        try:
            result = self.hybrid_runner.query(query)
            return {
                'dimension': 'capabilities',
                'analysis': result.answer,
                'confidence': result.confidence,
                'sources': result.citations
            }
        except Exception as e:
            logger.error(f"Capability analysis failed: {e}")
            return {'dimension': 'capabilities', 'analysis': 'Analysis failed', 'confidence': 0.0}
    
    def _analyze_historical_relationship(self, model1: str, model2: str) -> Dict:
        """Analyze historical relationship and evolution."""
        query = f"What is the historical relationship between {model1} and {model2}? Does one build on the other, are they competing approaches, or are they complementary?"
        
        try:
            result = self.hybrid_runner.query(query)
            return {
                'dimension': 'relationship',
                'analysis': result.answer,
                'confidence': result.confidence,
                'sources': result.citations
            }
        except Exception as e:
            logger.error(f"Relationship analysis failed: {e}")
            return {'dimension': 'relationship', 'analysis': 'Analysis failed', 'confidence': 0.0}
    
    def _synthesize_model_comparison(self, model1: str, model2: str, 
                                   dimensions: List[Dict]) -> Dict:
        """Synthesize comprehensive model comparison across all dimensions."""
        
        # Prepare synthesis context
        synthesis_context = []
        all_sources = set()
        
        for dim in dimensions:
            if dim.get('analysis') and dim['analysis'] != 'Analysis failed':
                synthesis_context.append(f"**{dim['dimension'].title()}:** {dim['analysis']}")
                all_sources.update(dim.get('sources', []))
        
        context_text = "\n\n".join(synthesis_context)
        
        synthesis_query = f"Based on this comprehensive analysis across multiple dimensions, provide a synthesized comparison of {model1} vs {model2}. Highlight the key differences, strengths, weaknesses, and overall relationship."
        
        try:
            # Use the hybrid runner for synthesis
            synthesis_prompt = f"""Synthesize this comprehensive model comparison:

Models: {model1} vs {model2}

Multi-dimensional Analysis:
{context_text[:4000]}

Provide a synthesized summary that:
1. Highlights key differences and similarities
2. Identifies strengths and weaknesses of each model
3. Explains the relationship between the models
4. Provides actionable insights for users

Synthesis:"""

            response = self.hybrid_runner.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=600,
                messages=[{"role": "user", "content": synthesis_prompt}]
            )
            
            synthesis = response.choices[0].message.content.strip()
            self.hybrid_runner.token_tracker.add_usage(response.usage.total_tokens)
            
            return {
                'synthesis': synthesis,
                'sources': list(all_sources),
                'confidence': self._calculate_synthesis_confidence(dimensions)
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {'synthesis': 'Synthesis failed', 'sources': [], 'confidence': 0.0}
    
    def _calculate_synthesis_confidence(self, dimensions: List[Dict]) -> float:
        """Calculate confidence for synthesis based on dimension confidences."""
        valid_dimensions = [d for d in dimensions if d.get('confidence', 0) > 0]
        
        if not valid_dimensions:
            return 0.0
        
        avg_confidence = sum(d['confidence'] for d in valid_dimensions) / len(valid_dimensions)
        
        # Boost confidence if we have multiple successful dimensions
        dimension_bonus = min(0.1, (len(valid_dimensions) - 1) * 0.02)
        
        return min(0.95, avg_confidence + dimension_bonus)
    
    def analyze_model_ecosystem(self, models: List[str]) -> Dict:
        """Analyze the entire ecosystem of models and their relationships."""
        logger.info(f"🌐 Analyzing model ecosystem: {models}")
        
        ecosystem_analysis = {
            'models': models,
            'pairwise_relationships': {},
            'ecosystem_insights': {},
            'model_clusters': [],
            'evolution_timeline': [],
            'performance_rankings': {}
        }
        
        # Analyze pairwise relationships
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                relationship = self._analyze_model_relationship(model1, model2)
                ecosystem_analysis['pairwise_relationships'][f"{model1}_vs_{model2}"] = relationship
        
        # Generate ecosystem insights
        ecosystem_insights = self._generate_ecosystem_insights(ecosystem_analysis['pairwise_relationships'])
        ecosystem_analysis['ecosystem_insights'] = ecosystem_insights
        
        return ecosystem_analysis
    
    def _analyze_model_relationship(self, model1: str, model2: str) -> Dict:
        """Analyze the relationship between two models."""
        query = f"What is the relationship between {model1} and {model2}? Are they competing approaches, does one build on the other, or are they complementary? Include any performance comparisons."
        
        try:
            result = self.hybrid_runner.query(query)
            return {
                'relationship_type': self._classify_relationship_type(result.answer),
                'description': result.answer,
                'confidence': result.confidence,
                'sources': result.citations
            }
        except Exception as e:
            logger.error(f"Relationship analysis failed: {e}")
            return {'relationship_type': 'unknown', 'description': 'Analysis failed', 'confidence': 0.0}
    
    def _classify_relationship_type(self, description: str) -> str:
        """Classify the type of relationship between models."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['builds on', 'extends', 'improves', 'successor']):
            return 'evolutionary'
        elif any(word in description_lower for word in ['competing', 'alternative', 'versus', 'compared']):
            return 'competitive'
        elif any(word in description_lower for word in ['complementary', 'together', 'combined', 'ensemble']):
            return 'complementary'
        elif any(word in description_lower for word in ['similar', 'same', 'equivalent']):
            return 'similar'
        else:
            return 'related'
    
    def _generate_ecosystem_insights(self, pairwise_relationships: Dict) -> Dict:
        """Generate insights about the overall model ecosystem."""
        
        relationship_types = {}
        for rel_data in pairwise_relationships.values():
            rel_type = rel_data.get('relationship_type', 'unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        insights = {
            'relationship_distribution': relationship_types,
            'dominant_relationship_type': max(relationship_types.items(), key=lambda x: x[1])[0] if relationship_types else 'unknown',
            'ecosystem_maturity': 'high' if relationship_types.get('evolutionary', 0) > 0 else 'emerging',
            'competition_level': 'high' if relationship_types.get('competitive', 0) > len(pairwise_relationships) * 0.5 else 'low'
        }
        
        return insights
    
    def find_similar_models(self, target_model: str, similarity_threshold: float = 0.7) -> List[Dict]:
        """Find models similar to the target model."""
        logger.info(f"🔍 Finding models similar to: {target_model}")
        
        query = f"Find models that are similar to {target_model} in terms of methodology, approach, or capabilities. What models solve similar problems or use similar techniques?"
        
        try:
            result = self.hybrid_runner.query(query)
            
            # Extract model names from the response (simple extraction)
            # In a more sophisticated version, this would use NER or structured extraction
            similar_models = self._extract_model_names_from_text(result.answer)
            
            return [{
                'model': model,
                'similarity_reason': result.answer,
                'confidence': result.confidence,
                'sources': result.citations
            } for model in similar_models if model.lower() != target_model.lower()]
            
        except Exception as e:
            logger.error(f"Similar model search failed: {e}")
            return []
    
    def _extract_model_names_from_text(self, text: str) -> List[str]:
        """Extract model names from text (simple implementation)."""
        # This is a simple implementation - could be enhanced with NER
        potential_models = []
        
        # Look for common model patterns
        import re
        
        # Pattern for models like "GPT-4", "BERT", "ResNet-50", etc.
        model_patterns = [
            r'\b[A-Z][a-zA-Z]*[-\s]*\d+[a-zA-Z]*\b',  # GPT-4, ResNet-50, etc.
            r'\b[A-Z]{2,}[a-zA-Z]*\b',  # BERT, GPT, etc.
            r'\b[A-Z][a-z]+[A-Z][a-z]+\b'  # CamelCase models
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, text)
            potential_models.extend(matches)
        
        # Filter and clean
        cleaned_models = []
        for model in potential_models:
            if len(model) > 2 and model not in cleaned_models:
                cleaned_models.append(model)
        
        return cleaned_models[:5]  # Return top 5 candidates
    
    def get_comparison_summary(self, model1: str, model2: str) -> str:
        """Get a quick comparison summary between two models."""
        cache_key = f"{model1}_vs_{model2}"
        
        if cache_key in self.comparison_cache:
            comparison = self.comparison_cache[cache_key]
            return comparison.get('comprehensive_synthesis', {}).get('synthesis', 'No summary available')
        
        # If not cached, perform quick comparison
        query = f"Provide a brief comparison summary of {model1} vs {model2} highlighting the key differences."
        
        try:
            result = self.hybrid_runner.query(query)
            return result.answer
        except Exception as e:
            logger.error(f"Quick comparison failed: {e}")
            return "Comparison unavailable"