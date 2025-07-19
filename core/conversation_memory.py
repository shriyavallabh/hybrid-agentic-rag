"""
Conversation Memory System for Hybrid Agentic RAG
Maintains conversation context and enables follow-up questions.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib


class ConversationMemory:
    """Manages conversation history and context for follow-up questions."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of turns to keep in memory
        """
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_summary: str = ""
        self.entities_mentioned: Dict[str, List[str]] = {
            'figures': [],
            'tables': [],
            'authors': [],
            'models': [],
            'datasets': [],
            'concepts': []
        }
        self.last_retrieval_results: Optional[Dict] = None
        self.session_id: str = self._generate_session_id()
        self.created_at: float = time.time()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def add_turn(self, user_query: str, system_response: str, 
                 retrieval_results: Optional[Dict] = None,
                 agent_trace: Optional[List[Dict]] = None):
        """
        Add a conversation turn to memory.
        
        Args:
            user_query: User's question
            system_response: System's answer
            retrieval_results: Retrieved chunks and entities
            agent_trace: Agent reasoning trace
        """
        turn = {
            'timestamp': time.time(),
            'user_query': user_query,
            'system_response': system_response,
            'retrieval_results': retrieval_results,
            'agent_trace': agent_trace,
            'turn_number': len(self.conversation_history) + 1
        }
        
        self.conversation_history.append(turn)
        
        # Update entities mentioned
        if retrieval_results:
            self._update_entities_mentioned(retrieval_results)
        
        # Store last retrieval results for follow-up queries
        self.last_retrieval_results = retrieval_results
        
        # Maintain max history limit
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Update context summary
        self._update_context_summary()
    
    def _update_entities_mentioned(self, retrieval_results: Dict):
        """Update tracked entities from retrieval results."""
        # Extract entities from graph results
        graph_entities = retrieval_results.get('graph_entities', [])
        for entity_info in graph_entities:
            entity_data = entity_info.get('entity_data', {})
            entity_type = entity_data.get('type', '').lower()
            entity_name = entity_data.get('name', '')
            
            if entity_type == 'figure' and entity_name not in self.entities_mentioned['figures']:
                self.entities_mentioned['figures'].append(entity_name)
            elif entity_type == 'table' and entity_name not in self.entities_mentioned['tables']:
                self.entities_mentioned['tables'].append(entity_name)
            elif entity_type == 'author' and entity_name not in self.entities_mentioned['authors']:
                self.entities_mentioned['authors'].append(entity_name)
            elif entity_type == 'model' and entity_name not in self.entities_mentioned['models']:
                self.entities_mentioned['models'].append(entity_name)
            elif entity_type == 'dataset' and entity_name not in self.entities_mentioned['datasets']:
                self.entities_mentioned['datasets'].append(entity_name)
    
    def _update_context_summary(self):
        """Update conversation context summary."""
        if not self.conversation_history:
            self.context_summary = ""
            return
        
        # Build context from recent turns
        recent_turns = self.conversation_history[-3:]  # Last 3 turns
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"User asked: {turn['user_query']}")
            # Include brief response summary
            response = turn['system_response']
            if len(response) > 200:
                response = response[:200] + "..."
            context_parts.append(f"System explained: {response}")
        
        self.context_summary = " ".join(context_parts)
    
    def get_context_for_query(self, current_query: str) -> Dict[str, Any]:
        """
        Get relevant context for the current query.
        
        Args:
            current_query: Current user query
            
        Returns:
            Context dictionary with relevant history and entities
        """
        context = {
            'session_id': self.session_id,
            'turn_number': len(self.conversation_history) + 1,
            'conversation_summary': self.context_summary,
            'entities_mentioned': self.entities_mentioned,
            'recent_queries': self._get_recent_queries(),
            'is_follow_up': self._is_follow_up_question(current_query),
            'referenced_entities': self._extract_referenced_entities(current_query),
            'last_retrieval_results': self.last_retrieval_results
        }
        
        return context
    
    def _get_recent_queries(self) -> List[str]:
        """Get list of recent user queries."""
        return [turn['user_query'] for turn in self.conversation_history[-3:]]
    
    def _is_follow_up_question(self, query: str) -> bool:
        """Determine if query is a follow-up question."""
        follow_up_indicators = [
            'it', 'this', 'that', 'these', 'those',
            'more', 'else', 'also', 'additionally',
            'what about', 'how about', 'tell me more',
            'explain further', 'elaborate', 'detail'
        ]
        
        query_lower = query.lower()
        
        # Check for pronouns referring to previous context
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True
        
        # Check if query references previously mentioned entities
        for entity_list in self.entities_mentioned.values():
            for entity in entity_list:
                if entity.lower() in query_lower:
                    return True
        
        return False
    
    def _extract_referenced_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract which previously mentioned entities are referenced."""
        referenced = {
            'figures': [],
            'tables': [],
            'authors': [],
            'models': [],
            'datasets': [],
            'concepts': []
        }
        
        query_lower = query.lower()
        
        # Check each category of mentioned entities
        for category, entities in self.entities_mentioned.items():
            for entity in entities:
                if entity.lower() in query_lower or self._pronoun_refers_to(query_lower, entity):
                    referenced[category].append(entity)
        
        return referenced
    
    def _pronoun_refers_to(self, query: str, entity: str) -> bool:
        """Check if pronouns in query might refer to entity."""
        if not self.conversation_history:
            return False
        
        # Check if entity was mentioned in last turn
        last_turn = self.conversation_history[-1]
        last_response = last_turn.get('system_response', '').lower()
        
        if entity.lower() in last_response:
            # Check for pronouns that might refer to it
            pronouns = ['it', 'this', 'that', 'its', 'their']
            for pronoun in pronouns:
                if pronoun in query:
                    return True
        
        return False
    
    def format_conversation_history(self) -> str:
        """Format conversation history for display or context."""
        if not self.conversation_history:
            return "No conversation history yet."
        
        formatted = []
        for i, turn in enumerate(self.conversation_history):
            formatted.append(f"Turn {turn['turn_number']}:")
            formatted.append(f"User: {turn['user_query']}")
            formatted.append(f"Assistant: {turn['system_response'][:200]}...")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversation memory."""
        return {
            'session_id': self.session_id,
            'total_turns': len(self.conversation_history),
            'session_duration': time.time() - self.created_at,
            'entities_tracked': {k: len(v) for k, v in self.entities_mentioned.items()},
            'memory_size': len(str(self.conversation_history))
        }
    
    def clear(self):
        """Clear conversation memory."""
        self.conversation_history.clear()
        self.context_summary = ""
        self.entities_mentioned = {k: [] for k in self.entities_mentioned}
        self.last_retrieval_results = None
        self.session_id = self._generate_session_id()
        self.created_at = time.time()


class MemoryAwareQueryProcessor:
    """Process queries with conversation memory context."""
    
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
    
    def enhance_query_with_context(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance query with conversation context.
        
        Args:
            query: User's current query
            
        Returns:
            Enhanced query and context dictionary
        """
        context = self.memory.get_context_for_query(query)
        
        # If it's a follow-up question, enhance the query
        if context['is_follow_up']:
            enhanced_query = self._enhance_follow_up_query(query, context)
        else:
            enhanced_query = query
        
        return enhanced_query, context
    
    def _enhance_follow_up_query(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance follow-up query with context."""
        enhanced = query
        
        # Replace pronouns with actual entities if clear
        if 'it' in query.lower() or 'that' in query.lower():
            # Get most recently mentioned entity
            last_entities = self._get_last_mentioned_entities(context)
            if last_entities:
                # Simple pronoun replacement
                enhanced = query
                for pronoun in ['it', 'that', 'this']:
                    if pronoun in enhanced.lower():
                        enhanced = enhanced.replace(pronoun, last_entities[0])
                        enhanced = enhanced.replace(pronoun.capitalize(), last_entities[0])
        
        # Add context if query is too vague
        if len(query.split()) < 5 and context['conversation_summary']:
            enhanced = f"{query} (in context of: {context['recent_queries'][-1] if context['recent_queries'] else 'previous discussion'})"
        
        return enhanced
    
    def _get_last_mentioned_entities(self, context: Dict[str, Any]) -> List[str]:
        """Get most recently mentioned entities."""
        entities = []
        
        # Collect all mentioned entities
        for category, entity_list in context['entities_mentioned'].items():
            entities.extend(entity_list)
        
        # Return most recent ones (reverse order)
        return entities[-3:] if entities else []
    
    def should_use_cached_results(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if cached results can be used."""
        # Use cached results for very similar follow-up queries
        if not context['is_follow_up'] or not context['last_retrieval_results']:
            return False
        
        # Check if query is asking for more details about same topic
        detail_indicators = ['more', 'details', 'elaborate', 'explain further', 'what else']
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in detail_indicators)
    
    def format_response_with_context(self, response: str, used_context: bool) -> str:
        """Format response with context awareness."""
        if used_context and self.memory.conversation_history:
            # Add subtle context indicator
            return f"{response}\n\n(This builds on our previous discussion about {self._get_topic_summary()})"
        return response
    
    def _get_topic_summary(self) -> str:
        """Get brief summary of conversation topics."""
        if not self.memory.entities_mentioned:
            return "this topic"
        
        # Find most discussed category
        max_category = max(self.memory.entities_mentioned.items(), 
                          key=lambda x: len(x[1]))
        
        if max_category[1]:
            return f"{max_category[0]}: {', '.join(max_category[1][:2])}"
        
        return "this topic"