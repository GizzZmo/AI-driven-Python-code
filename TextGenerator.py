import requests
import json
import logging
from typing import Dict, List, Optional, Union
import re
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextGenerator:
    """Advanced text generator with multiple AI providers and customization options."""
    
    def __init__(self, provider: str = "openai", api_key: str = "", **kwargs):
        """Initialize the TextGenerator with specified AI provider.
        
        Args:
            provider: AI provider ('openai', 'anthropic', 'gemini', 'local')
            api_key: API key for the service
            **kwargs: Additional provider-specific configuration
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.config = kwargs
        
        # Provider configurations
        self.providers = {
            'openai': {
                'base_url': 'https://api.openai.com/v1/',
                'models': ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003']
            },
            'anthropic': {
                'base_url': 'https://api.anthropic.com/v1/',
                'models': ['claude-3-opus', 'claude-3-sonnet', 'claude-instant-1']
            },
            'gemini': {
                'base_url': 'https://generativelanguage.googleapis.com/v1/',
                'models': ['gemini-pro', 'gemini-pro-vision']
            }
        }
        
        # Text generation presets
        self.style_presets = {
            'creative': {
                'temperature': 0.9,
                'top_p': 0.9,
                'max_tokens': 2000,
                'frequency_penalty': 0.3
            },
            'analytical': {
                'temperature': 0.3,
                'top_p': 0.8,
                'max_tokens': 1500,
                'frequency_penalty': 0.1
            },
            'conversational': {
                'temperature': 0.7,
                'top_p': 0.85,
                'max_tokens': 1000,
                'frequency_penalty': 0.2
            },
            'technical': {
                'temperature': 0.2,
                'top_p': 0.7,
                'max_tokens': 2500,
                'frequency_penalty': 0.0
            }
        }
        
    def generate_story(self, 
                      prompt: str,
                      genre: str = "adventure",
                      length: str = "short",
                      style: str = "creative",
                      characters: Optional[List[str]] = None,
                      setting: Optional[str] = None,
                      themes: Optional[List[str]] = None,
                      target_audience: str = "general") -> Dict:
        """Generate a story with customizable parameters.
        
        Args:
            prompt: Story prompt or concept
            genre: Genre (adventure, mystery, sci-fi, fantasy, romance, horror)
            length: Length (flash, short, medium, long)
            style: Writing style preset
            characters: List of character names/descriptions
            setting: Story setting description
            themes: List of themes to incorporate
            target_audience: Target audience (children, teen, adult, general)
            
        Returns:
            Dictionary containing story data
        """
        # Build enhanced prompt
        enhanced_prompt = self._build_story_prompt(
            prompt, genre, length, characters, setting, themes, target_audience
        )
        
        # Get length specifications
        length_specs = {
            'flash': {'max_tokens': 500, 'target_words': '100-300'},
            'short': {'max_tokens': 1500, 'target_words': '500-1500'},
            'medium': {'max_tokens': 3000, 'target_words': '1500-3000'},
            'long': {'max_tokens': 5000, 'target_words': '3000-5000'}
        }
        
        # Generate story
        result = self._generate_text(
            enhanced_prompt,
            style=style,
            max_tokens=length_specs[length]['max_tokens']
        )
        
        # Post-process and analyze
        story_data = {
            'story': result['text'],
            'metadata': {
                'genre': genre,
                'length': length,
                'style': style,
                'target_audience': target_audience,
                'word_count': len(result['text'].split()),
                'reading_time': self._estimate_reading_time(result['text']),
                'characters': characters or [],
                'setting': setting,
                'themes': themes or []
            },
            'analysis': self._analyze_story(result['text'])
        }
        
        return story_data
        
    def generate_code(self,
                     description: str,
                     language: str = "python",
                     style: str = "technical",
                     include_comments: bool = True,
                     include_tests: bool = False,
                     complexity: str = "intermediate",
                     frameworks: Optional[List[str]] = None) -> Dict:
        """Generate code with customizable parameters.
        
        Args:
            description: Description of what the code should do
            language: Programming language
            style: Code style preset
            include_comments: Whether to include detailed comments
            include_tests: Whether to include unit tests
            complexity: Code complexity (beginner, intermediate, advanced)
            frameworks: List of frameworks/libraries to use
            
        Returns:
            Dictionary containing code and metadata
        """
        # Build code generation prompt
        prompt = self._build_code_prompt(
            description, language, include_comments, include_tests, 
            complexity, frameworks
        )
        
        # Generate code
        result = self._generate_text(prompt, style=style, max_tokens=2500)
        
        # Extract code and comments
        code_data = self._parse_code_response(result['text'], language)
        
        return {
            'code': code_data['main_code'],
            'tests': code_data.get('tests', ''),
            'documentation': code_data.get('documentation', ''),
            'metadata': {
                'language': language,
                'complexity': complexity,
                'frameworks': frameworks or [],
                'includes_comments': include_comments,
                'includes_tests': include_tests,
                'lines_of_code': len(code_data['main_code'].split('\n'))
            }
        }
        
    def generate_article(self,
                        topic: str,
                        article_type: str = "informative",
                        target_audience: str = "general",
                        length: str = "medium",
                        tone: str = "professional",
                        include_sources: bool = False,
                        keywords: Optional[List[str]] = None) -> Dict:
        """Generate an article with customizable parameters.
        
        Args:
            topic: Article topic
            article_type: Type (informative, persuasive, entertainment, tutorial)
            target_audience: Target audience
            length: Article length
            tone: Writing tone (professional, casual, academic, friendly)
            include_sources: Whether to include citations
            keywords: SEO keywords to incorporate
            
        Returns:
            Dictionary containing article data
        """
        prompt = self._build_article_prompt(
            topic, article_type, target_audience, tone, include_sources, keywords
        )
        
        length_specs = {
            'short': 1000,
            'medium': 2000,
            'long': 3500
        }
        
        result = self._generate_text(
            prompt,
            style='analytical',
            max_tokens=length_specs[length]
        )
        
        article_data = self._parse_article(result['text'])
        
        return {
            'title': article_data.get('title', f"Article on {topic}"),
            'content': article_data['content'],
            'outline': article_data.get('outline', []),
            'metadata': {
                'topic': topic,
                'type': article_type,
                'target_audience': target_audience,
                'tone': tone,
                'word_count': len(article_data['content'].split()),
                'reading_time': self._estimate_reading_time(article_data['content']),
                'keywords': keywords or []
            }
        }
        
    def _generate_text(self, prompt: str, style: str = "creative", **kwargs) -> Dict:
        """Core text generation method."""
        # Get style parameters
        params = self.style_presets.get(style, self.style_presets['creative'])
        params.update(kwargs)
        
        # Simulate API call (replace with actual API integration)
        logger.info(f"Generating text with {self.provider} using {style} style...")
        
        # For demonstration purposes, return a mock response
        # In real implementation, this would call the actual API
        mock_responses = {
            'story': f"Once upon a time, in a world where {prompt.lower()}...",
            'code': f"# {prompt}\n\ndef example_function():\n    pass",
            'article': f"# {prompt}\n\nThis article explores the fascinating topic of {prompt.lower()}..."
        }
        
        # Determine response type based on prompt content
        if any(word in prompt.lower() for word in ['story', 'tale', 'narrative']):
            response_type = 'story'
        elif any(word in prompt.lower() for word in ['code', 'function', 'program']):
            response_type = 'code'
        else:
            response_type = 'article'
            
        return {
            'text': mock_responses.get(response_type, f"Generated content for: {prompt}"),
            'tokens_used': params.get('max_tokens', 1000),
            'model': self.providers.get(self.provider, {}).get('models', ['unknown'])[0]
        }
        
    def _build_story_prompt(self, prompt, genre, length, characters, setting, themes, audience):
        """Build enhanced story prompt."""
        enhanced = f"Write a {genre} story"
        
        if length:
            enhanced += f" of {length} length"
            
        enhanced += f" for a {audience} audience"
        
        if setting:
            enhanced += f" set in {setting}"
            
        if characters:
            enhanced += f" featuring characters: {', '.join(characters)}"
            
        if themes:
            enhanced += f" incorporating themes of: {', '.join(themes)}"
            
        enhanced += f"\n\nStory concept: {prompt}"
        
        return enhanced
        
    def _build_code_prompt(self, description, language, comments, tests, complexity, frameworks):
        """Build code generation prompt."""
        prompt = f"Write {complexity} level {language} code that {description}"
        
        if frameworks:
            prompt += f" using {', '.join(frameworks)}"
            
        if comments:
            prompt += ". Include detailed comments explaining the code."
            
        if tests:
            prompt += " Also include unit tests for the code."
            
        return prompt
        
    def _build_article_prompt(self, topic, article_type, audience, tone, sources, keywords):
        """Build article generation prompt."""
        prompt = f"Write a {article_type} article about {topic}"
        prompt += f" for a {audience} audience in a {tone} tone"
        
        if keywords:
            prompt += f". Include these keywords naturally: {', '.join(keywords)}"
            
        if sources:
            prompt += ". Include credible sources and citations."
            
        return prompt
        
    def _analyze_story(self, text: str) -> Dict:
        """Analyze story characteristics."""
        sentences = text.split('.')
        words = text.split()
        
        return {
            'sentence_count': len(sentences),
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'complexity_score': min(10, len(set(words)) / len(words) * 20) if words else 0,
            'dialogue_percentage': len(re.findall(r'"[^"]*"', text)) / len(sentences) * 100 if sentences else 0
        }
        
    def _parse_code_response(self, text: str, language: str) -> Dict:
        """Parse code from response."""
        # Simple parsing - in real implementation, use proper code parsing
        lines = text.split('\n')
        code_lines = [line for line in lines if not line.strip().startswith('#') or language == 'python']
        
        return {
            'main_code': '\n'.join(code_lines),
            'documentation': '',
            'tests': ''
        }
        
    def _parse_article(self, text: str) -> Dict:
        """Parse article structure."""
        lines = text.split('\n')
        title = lines[0].replace('#', '').strip() if lines else "Generated Article"
        content = '\n'.join(lines[1:]) if len(lines) > 1 else text
        
        return {
            'title': title,
            'content': content,
            'outline': []
        }
        
    def _estimate_reading_time(self, text: str) -> str:
        """Estimate reading time in minutes."""
        words = len(text.split())
        minutes = max(1, words // 200)  # Average reading speed: 200 words/minute
        return f"{minutes} minute{'s' if minutes != 1 else ''}"

# Example usage
if __name__ == "__main__":
    # Initialize text generator
    generator = TextGenerator(provider="openai", api_key="your-api-key")
    
    # Example 1: Generate a story
    print("Generating a story...")
    story = generator.generate_story(
        prompt="A robot discovers emotions",
        genre="sci-fi",
        length="short",
        characters=["ARIA-7", "Dr. Chen", "The Observer"],
        setting="Neo-Tokyo 2157",
        themes=["consciousness", "humanity", "discovery"],
        target_audience="teen"
    )
    print(f"Story generated: {story['metadata']['word_count']} words")
    print(f"Reading time: {story['metadata']['reading_time']}")
    
    # Example 2: Generate code
    print("\nGenerating code...")
    code = generator.generate_code(
        description="Create a REST API for a todo list application",
        language="python",
        include_comments=True,
        include_tests=True,
        frameworks=["flask", "sqlalchemy"]
    )
    print(f"Code generated: {code['metadata']['lines_of_code']} lines")
    
    # Example 3: Generate article
    print("\nGenerating article...")
    article = generator.generate_article(
        topic="The Future of Artificial Intelligence",
        article_type="informative",
        target_audience="tech professionals",
        length="medium",
        tone="professional",
        keywords=["AI", "machine learning", "automation", "future technology"]
    )
    print(f"Article generated: {article['metadata']['word_count']} words")
    print(f"Title: {article['title']}")