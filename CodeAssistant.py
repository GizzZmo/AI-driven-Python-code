import requests
import json
import ast
import logging
import re
import os
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import subprocess
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeReview:
    """Data class for code review results."""
    overall_score: int
    issues: List[Dict]
    suggestions: List[str]
    complexity_score: int
    maintainability_score: int
    
@dataclass
class CodeGeneration:
    """Data class for code generation results."""
    code: str
    explanation: str
    dependencies: List[str]
    tests: str
    documentation: str

class CodeAssistant:
    """Advanced code assistant with AI-powered analysis, generation, and optimization."""
    
    def __init__(self, api_key: str, provider: str = "openai", **kwargs):
        """Initialize the CodeAssistant with AI provider configuration.
        
        Args:
            api_key: API key for the AI service
            provider: AI provider ('openai', 'anthropic', 'github')
            **kwargs: Additional configuration options
        """
        self.api_key = api_key
        self.provider = provider.lower()
        self.config = kwargs
        
        # Supported languages and their configurations
        self.language_configs = {
            'python': {
                'extensions': ['.py'],
                'linters': ['pylint', 'flake8', 'black'],
                'test_frameworks': ['pytest', 'unittest'],
                'complexity_tools': ['radon', 'mccabe']
            },
            'javascript': {
                'extensions': ['.js', '.jsx', '.ts', '.tsx'],
                'linters': ['eslint', 'prettier'],
                'test_frameworks': ['jest', 'mocha'],
                'complexity_tools': ['jshint']
            },
            'java': {
                'extensions': ['.java'],
                'linters': ['checkstyle', 'spotbugs'],
                'test_frameworks': ['junit'],
                'complexity_tools': ['pmd']
            },
            'go': {
                'extensions': ['.go'],
                'linters': ['golint', 'gofmt'],
                'test_frameworks': ['testing'],
                'complexity_tools': ['gocyclo']
            }
        }
        
        # Code generation templates
        self.templates = {
            'python': {
                'class': self._get_python_class_template(),
                'function': self._get_python_function_template(),
                'api': self._get_python_api_template(),
                'cli': self._get_python_cli_template()
            },
            'javascript': {
                'component': self._get_js_component_template(),
                'function': self._get_js_function_template(),
                'api': self._get_js_api_template()
            }
        }
        
    def generate_code(self,
                     description: str,
                     language: str = "python",
                     code_type: str = "function",
                     style_guide: str = "pep8",
                     include_tests: bool = True,
                     include_docs: bool = True,
                     frameworks: Optional[List[str]] = None,
                     complexity_level: str = "intermediate") -> CodeGeneration:
        """Generate code based on description with extensive customization.
        
        Args:
            description: Description of what the code should do
            language: Programming language
            code_type: Type of code (function, class, api, cli, component)
            style_guide: Coding style guide to follow
            include_tests: Whether to generate unit tests
            include_docs: Whether to include documentation
            frameworks: List of frameworks/libraries to use
            complexity_level: Complexity level (beginner, intermediate, advanced)
            
        Returns:
            CodeGeneration object with generated code and metadata
        """
        # Build generation prompt
        prompt = self._build_generation_prompt(
            description, language, code_type, style_guide, 
            include_tests, include_docs, frameworks, complexity_level
        )
        
        # Generate code using AI
        response = self._call_ai_api(prompt, "code_generation")
        
        # Parse response
        code_data = self._parse_code_response(response, language)
        
        # Validate generated code
        validation_results = self._validate_code(code_data['code'], language)
        
        return CodeGeneration(
            code=code_data['code'],
            explanation=code_data.get('explanation', ''),
            dependencies=self._extract_dependencies(code_data['code'], language),
            tests=code_data.get('tests', ''),
            documentation=code_data.get('documentation', '')
        )
        
    def review_code(self,
                   code: str,
                   language: str = "python",
                   review_type: str = "comprehensive",
                   focus_areas: Optional[List[str]] = None,
                   severity_threshold: str = "medium") -> CodeReview:
        """Perform AI-powered code review with customizable focus.
        
        Args:
            code: Code to review
            language: Programming language
            review_type: Type of review (quick, comprehensive, security, performance)
            focus_areas: Specific areas to focus on
            severity_threshold: Minimum severity to report (low, medium, high, critical)
            
        Returns:
            CodeReview object with detailed analysis
        """
        # Analyze code structure
        structure_analysis = self._analyze_code_structure(code, language)
        
        # Perform static analysis
        static_issues = self._run_static_analysis(code, language)
        
        # AI-powered review
        ai_review = self._ai_code_review(code, language, review_type, focus_areas)
        
        # Calculate scores
        complexity_score = self._calculate_complexity_score(code, language)
        maintainability_score = self._calculate_maintainability_score(code, language)
        
        # Combine all issues and suggestions
        all_issues = static_issues + ai_review.get('issues', [])
        all_suggestions = ai_review.get('suggestions', [])
        
        # Filter by severity
        filtered_issues = self._filter_by_severity(all_issues, severity_threshold)
        
        overall_score = self._calculate_overall_score(
            complexity_score, maintainability_score, len(filtered_issues)
        )
        
        return CodeReview(
            overall_score=overall_score,
            issues=filtered_issues,
            suggestions=all_suggestions,
            complexity_score=complexity_score,
            maintainability_score=maintainability_score
        )
        
    def optimize_code(self,
                     code: str,
                     language: str = "python",
                     optimization_goals: List[str] = None,
                     preserve_functionality: bool = True) -> Dict:
        """Optimize code for performance, readability, or other goals.
        
        Args:
            code: Code to optimize
            language: Programming language
            optimization_goals: Goals like 'performance', 'readability', 'memory'
            preserve_functionality: Whether to preserve exact functionality
            
        Returns:
            Dictionary containing optimized code and analysis
        """
        if optimization_goals is None:
            optimization_goals = ['performance', 'readability']
            
        # Analyze current code
        original_analysis = self._analyze_performance(code, language)
        
        # Generate optimization suggestions
        optimizations = self._generate_optimizations(code, language, optimization_goals)
        
        # Apply optimizations
        optimized_code = self._apply_optimizations(code, optimizations, language)
        
        # Analyze optimized code
        optimized_analysis = self._analyze_performance(optimized_code, language)
        
        return {
            'original_code': code,
            'optimized_code': optimized_code,
            'optimizations_applied': optimizations,
            'performance_comparison': {
                'original': original_analysis,
                'optimized': optimized_analysis,
                'improvement_percentage': self._calculate_improvement(
                    original_analysis, optimized_analysis
                )
            },
            'preserves_functionality': preserve_functionality
        }
        
    def refactor_code(self,
                     code: str,
                     language: str = "python",
                     refactoring_type: str = "extract_methods",
                     target_pattern: str = None) -> Dict:
        """Refactor code using various techniques.
        
        Args:
            code: Code to refactor
            language: Programming language
            refactoring_type: Type of refactoring to apply
            target_pattern: Specific pattern to refactor
            
        Returns:
            Dictionary containing refactored code and changes made
        """
        refactoring_types = {
            'extract_methods': self._extract_methods,
            'extract_classes': self._extract_classes,
            'rename_variables': self._rename_variables,
            'simplify_conditionals': self._simplify_conditionals,
            'remove_duplicates': self._remove_duplicates
        }
        
        if refactoring_type not in refactoring_types:
            raise ValueError(f"Unsupported refactoring type: {refactoring_type}")
            
        # Apply refactoring
        refactoring_func = refactoring_types[refactoring_type]
        refactored_code, changes = refactoring_func(code, language)
        
        return {
            'original_code': code,
            'refactored_code': refactored_code,
            'refactoring_type': refactoring_type,
            'changes_made': changes,
            'improvement_score': self._calculate_refactoring_score(code, refactored_code, language)
        }
        
    def explain_code(self,
                    code: str,
                    language: str = "python",
                    explanation_level: str = "intermediate",
                    include_examples: bool = True) -> Dict:
        """Generate detailed explanations of code functionality.
        
        Args:
            code: Code to explain
            language: Programming language
            explanation_level: Level of explanation (beginner, intermediate, advanced)
            include_examples: Whether to include usage examples
            
        Returns:
            Dictionary containing detailed explanations
        """
        # Parse code structure
        structure = self._parse_code_structure(code, language)
        
        # Generate AI explanation
        ai_explanation = self._generate_ai_explanation(code, language, explanation_level)
        
        # Create examples if requested
        examples = []
        if include_examples:
            examples = self._generate_usage_examples(code, language)
            
        return {
            'code': code,
            'language': language,
            'structure_overview': structure,
            'detailed_explanation': ai_explanation,
            'key_concepts': self._extract_key_concepts(code, language),
            'usage_examples': examples,
            'complexity_explanation': self._explain_complexity(code, language),
            'best_practices': self._identify_best_practices(code, language)
        }
        
    def generate_tests(self,
                      code: str,
                      language: str = "python",
                      test_framework: str = None,
                      coverage_goal: int = 90,
                      test_types: List[str] = None) -> Dict:
        """Generate comprehensive test suites for given code.
        
        Args:
            code: Code to generate tests for
            language: Programming language
            test_framework: Testing framework to use
            coverage_goal: Target code coverage percentage
            test_types: Types of tests (unit, integration, edge_cases)
            
        Returns:
            Dictionary containing generated tests and metadata
        """
        if test_types is None:
            test_types = ['unit', 'edge_cases']
            
        if test_framework is None:
            test_framework = self.language_configs[language]['test_frameworks'][0]
            
        # Analyze code to identify testable components
        testable_components = self._identify_testable_components(code, language)
        
        # Generate tests for each component
        generated_tests = {}
        for component in testable_components:
            tests = self._generate_component_tests(
                component, language, test_framework, test_types
            )
            generated_tests[component['name']] = tests
            
        # Calculate estimated coverage
        estimated_coverage = self._estimate_test_coverage(code, generated_tests)
        
        return {
            'test_code': self._combine_test_code(generated_tests, test_framework),
            'test_framework': test_framework,
            'estimated_coverage': estimated_coverage,
            'test_breakdown': generated_tests,
            'setup_instructions': self._generate_test_setup_instructions(test_framework),
            'run_instructions': self._generate_test_run_instructions(test_framework)
        }
        
    # Helper methods for code generation
    def _get_python_class_template(self) -> str:
        return """
class {class_name}:
    \"\"\"
    {class_description}
    \"\"\"
    
    def __init__(self, {init_params}):
        {init_body}
        
    def {method_name}(self, {method_params}):
        \"\"\"
        {method_description}
        \"\"\"
        {method_body}
"""

    def _get_python_function_template(self) -> str:
        return """
def {function_name}({parameters}) -> {return_type}:
    \"\"\"
    {function_description}
    
    Args:
        {args_description}
        
    Returns:
        {return_description}
    \"\"\"
    {function_body}
"""

    def _get_python_api_template(self) -> str:
        return """
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/{endpoint}', methods=['{method}'])
def {function_name}():
    \"\"\"
    {endpoint_description}
    \"\"\"
    {endpoint_body}
    
if __name__ == '__main__':
    app.run(debug=True)
"""

    def _get_python_cli_template(self) -> str:
        return """
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='{description}')
    {argument_definitions}
    
    args = parser.parse_args()
    
    {main_logic}

if __name__ == '__main__':
    main()
"""

    def _get_js_component_template(self) -> str:
        return """
import React from 'react';

const {component_name} = ({props}) => {{
    {component_logic}
    
    return (
        <div>
            {component_jsx}
        </div>
    );
}};

export default {component_name};
"""

    def _get_js_function_template(self) -> str:
        return """
/**
 * {function_description}
 * @param {{type}} {param_name} - {param_description}
 * @returns {{return_type}} {return_description}
 */
function {function_name}({parameters}) {{
    {function_body}
}}

export default {function_name};
"""

    def _get_js_api_template(self) -> str:
        return """
const express = require('express');
const app = express();

app.use(express.json());

app.{method}('/{endpoint}', (req, res) => {{
    {endpoint_logic}
}});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {{
    console.log(`Server running on port ${{PORT}}`);
}});
"""

    # Mock implementations for complex operations
    def _call_ai_api(self, prompt: str, task_type: str) -> Dict:
        """Mock AI API call - replace with actual implementation."""
        logger.info(f"Calling AI API for {task_type}")
        
        # Mock responses based on task type
        if task_type == "code_generation":
            return {
                "code": "def example_function(x, y):\n    return x + y",
                "explanation": "This function adds two numbers together.",
                "tests": "def test_example_function():\n    assert example_function(1, 2) == 3"
            }
        elif task_type == "code_review":
            return {
                "issues": [
                    {"line": 1, "severity": "medium", "message": "Consider adding type hints"}
                ],
                "suggestions": ["Add docstrings", "Use more descriptive variable names"]
            }
        
        return {"response": "Mock AI response"}
        
    def _build_generation_prompt(self, description, language, code_type, style_guide, 
                                include_tests, include_docs, frameworks, complexity_level):
        """Build prompt for code generation."""
        prompt = f"Generate {complexity_level} level {language} {code_type} that {description}"
        
        if frameworks:
            prompt += f" using {', '.join(frameworks)}"
            
        prompt += f" following {style_guide} style guide"
        
        if include_tests:
            prompt += " with comprehensive unit tests"
            
        if include_docs:
            prompt += " with detailed documentation"
            
        return prompt
        
    def _parse_code_response(self, response: Dict, language: str) -> Dict:
        """Parse AI response to extract code components."""
        return {
            "code": response.get("code", ""),
            "explanation": response.get("explanation", ""),
            "tests": response.get("tests", ""),
            "documentation": response.get("documentation", "")
        }
        
    def _validate_code(self, code: str, language: str) -> Dict:
        """Validate generated code for syntax errors."""
        if language == "python":
            try:
                ast.parse(code)
                return {"valid": True, "errors": []}
            except SyntaxError as e:
                return {"valid": False, "errors": [str(e)]}
        return {"valid": True, "errors": []}
        
    def _extract_dependencies(self, code: str, language: str) -> List[str]:
        """Extract dependencies from code."""
        if language == "python":
            imports = re.findall(r'^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)', code, re.MULTILINE)
            return list(set(imports))
        return []
        
    def _analyze_code_structure(self, code: str, language: str) -> Dict:
        """Analyze code structure."""
        if language == "python":
            try:
                tree = ast.parse(code)
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                return {
                    "functions": functions,
                    "classes": classes,
                    "lines": len(code.split('\n'))
                }
            except:
                pass
        return {"functions": [], "classes": [], "lines": len(code.split('\n'))}
        
    def _run_static_analysis(self, code: str, language: str) -> List[Dict]:
        """Run static analysis tools."""
        # Mock static analysis results
        return [
            {"line": 1, "severity": "low", "message": "Line too long", "rule": "E501"},
            {"line": 5, "severity": "medium", "message": "Unused variable", "rule": "W0612"}
        ]
        
    def _ai_code_review(self, code: str, language: str, review_type: str, focus_areas: List[str]) -> Dict:
        """Perform AI-powered code review."""
        return self._call_ai_api(f"Review this {language} code: {code}", "code_review")
        
    def _calculate_complexity_score(self, code: str, language: str) -> int:
        """Calculate code complexity score."""
        lines = len(code.split('\n'))
        functions = len(re.findall(r'def\s+\w+', code)) if language == "python" else 0
        complexity = min(100, max(0, 100 - (lines // 10) - (functions * 5)))
        return complexity
        
    def _calculate_maintainability_score(self, code: str, language: str) -> int:
        """Calculate maintainability score."""
        # Simple heuristic based on code characteristics
        comments = len(re.findall(r'#.*', code)) if language == "python" else 0
        functions = len(re.findall(r'def\s+\w+', code)) if language == "python" else 0
        lines = len(code.split('\n'))
        
        score = 50 + (comments * 5) + (functions * 3) - (lines // 20)
        return max(0, min(100, score))
        
    def _filter_by_severity(self, issues: List[Dict], threshold: str) -> List[Dict]:
        """Filter issues by severity threshold."""
        severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        threshold_level = severity_order.get(threshold, 2)
        
        return [issue for issue in issues 
                if severity_order.get(issue.get("severity", "low"), 1) >= threshold_level]
        
    def _calculate_overall_score(self, complexity: int, maintainability: int, issue_count: int) -> int:
        """Calculate overall code quality score."""
        base_score = (complexity + maintainability) // 2
        penalty = min(30, issue_count * 5)
        return max(0, base_score - penalty)
        
    # Placeholder implementations for other methods
    def _analyze_performance(self, code: str, language: str) -> Dict:
        return {"execution_time": "0.1ms", "memory_usage": "1MB", "complexity": "O(n)"}
        
    def _generate_optimizations(self, code: str, language: str, goals: List[str]) -> List[Dict]:
        return [{"type": "loop_optimization", "description": "Replace nested loop with list comprehension"}]
        
    def _apply_optimizations(self, code: str, optimizations: List[Dict], language: str) -> str:
        return code  # Placeholder - would apply actual optimizations
        
    def _calculate_improvement(self, original: Dict, optimized: Dict) -> Dict:
        return {"speed": "15%", "memory": "10%"}
        
    def _extract_methods(self, code: str, language: str) -> Tuple[str, List[str]]:
        return code, ["Extracted method 'calculate_total'"]
        
    def _extract_classes(self, code: str, language: str) -> Tuple[str, List[str]]:
        return code, ["Extracted class 'DataProcessor'"]
        
    def _rename_variables(self, code: str, language: str) -> Tuple[str, List[str]]:
        return code, ["Renamed 'x' to 'user_count'"]
        
    def _simplify_conditionals(self, code: str, language: str) -> Tuple[str, List[str]]:
        return code, ["Simplified nested if statements"]
        
    def _remove_duplicates(self, code: str, language: str) -> Tuple[str, List[str]]:
        return code, ["Removed duplicate function 'validate_input'"]
        
    def _calculate_refactoring_score(self, original: str, refactored: str, language: str) -> int:
        return 85  # Mock score
        
    def _parse_code_structure(self, code: str, language: str) -> Dict:
        return self._analyze_code_structure(code, language)
        
    def _generate_ai_explanation(self, code: str, language: str, level: str) -> str:
        return f"This {language} code demonstrates {level} level programming concepts."
        
    def _generate_usage_examples(self, code: str, language: str) -> List[str]:
        return ["example = MyClass()", "result = example.process_data(input_data)"]
        
    def _extract_key_concepts(self, code: str, language: str) -> List[str]:
        return ["object-oriented programming", "error handling", "data processing"]
        
    def _explain_complexity(self, code: str, language: str) -> str:
        return "This code has O(n) time complexity and O(1) space complexity."
        
    def _identify_best_practices(self, code: str, language: str) -> List[str]:
        return ["Uses descriptive variable names", "Includes error handling", "Follows PEP 8 style guide"]
        
    def _identify_testable_components(self, code: str, language: str) -> List[Dict]:
        structure = self._analyze_code_structure(code, language)
        components = []
        for func in structure['functions']:
            components.append({"name": func, "type": "function", "complexity": "medium"})
        for cls in structure['classes']:
            components.append({"name": cls, "type": "class", "complexity": "high"})
        return components
        
    def _generate_component_tests(self, component: Dict, language: str, framework: str, test_types: List[str]) -> Dict:
        return {
            "test_code": f"def test_{component['name']}():\n    assert {component['name']}() is not None",
            "test_count": 3,
            "coverage_estimate": 85
        }
        
    def _combine_test_code(self, tests: Dict, framework: str) -> str:
        combined = f"# Generated tests using {framework}\n\n"
        for component, test_data in tests.items():
            combined += f"# Tests for {component}\n"
            combined += test_data['test_code'] + "\n\n"
        return combined
        
    def _estimate_test_coverage(self, code: str, tests: Dict) -> int:
        return 85  # Mock coverage estimate
        
    def _generate_test_setup_instructions(self, framework: str) -> List[str]:
        if framework == "pytest":
            return ["pip install pytest", "pip install pytest-cov"]
        return [f"Install {framework}"]
        
    def _generate_test_run_instructions(self, framework: str) -> List[str]:
        if framework == "pytest":
            return ["pytest tests/", "pytest --cov=. tests/"]
        return [f"Run tests with {framework}"]

# Example usage
if __name__ == "__main__":
    assistant = CodeAssistant(api_key="your-api-key")
    
    # Example 1: Generate code
    print("Generating code...")
    generated = assistant.generate_code(
        description="Create a function to calculate fibonacci numbers",
        language="python",
        code_type="function",
        include_tests=True,
        include_docs=True,
        complexity_level="intermediate"
    )
    print(f"Generated {len(generated.code.split())} lines of code")
    
    # Example 2: Review code
    sample_code = """
def calc(x, y):
    if x > y:
        return x + y
    else:
        return x - y
"""
    
    print("\nReviewing code...")
    review = assistant.review_code(
        sample_code,
        language="python",
        review_type="comprehensive"
    )
    print(f"Code quality score: {review.overall_score}/100")
    print(f"Found {len(review.issues)} issues")
    
    # Example 3: Generate tests
    print("\nGenerating tests...")
    tests = assistant.generate_tests(
        sample_code,
        language="python",
        test_framework="pytest"
    )
    print(f"Generated tests with {tests['estimated_coverage']}% coverage")
    
    # Example 4: Explain code
    print("\nExplaining code...")
    explanation = assistant.explain_code(
        sample_code,
        language="python",
        explanation_level="beginner"
    )
    print(f"Identified {len(explanation['key_concepts'])} key concepts")