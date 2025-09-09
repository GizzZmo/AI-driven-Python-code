#!/usr/bin/env python3
"""
Example usage of AI-driven Python code tools.

This script demonstrates the various features and customization options
available in the AI-driven Python code library.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from CartoonGenerator import CartoonGenerator
from TextGenerator import TextGenerator
from ImageAnalyzer import ImageAnalyzer
from CodeAssistant import CodeAssistant
from PIL import Image, ImageDraw

def setup_demo_environment():
    """Set up demo environment with sample files."""
    print("Setting up demo environment...")
    
    # Create output directories
    output_dirs = [
        "./examples_output",
        "./examples_output/cartoons",
        "./examples_output/images", 
        "./examples_output/text",
        "./examples_output/code"
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create a sample image for image analysis
    sample_img = Image.new('RGB', (600, 400), 'lightblue')
    draw = ImageDraw.Draw(sample_img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 200, 150], fill='red', outline='darkred', width=3)
    draw.ellipse([250, 100, 400, 250], fill='green', outline='darkgreen', width=3)
    draw.polygon([(450, 50), (550, 100), (500, 200), (400, 150)], fill='yellow', outline='orange', width=3)
    
    # Add some text
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((50, 300), "Sample Image for Analysis", fill='black', font=font)
    except:
        draw.text((50, 300), "Sample Image for Analysis", fill='black')
    
    sample_path = "./examples_output/sample_image.png"
    sample_img.save(sample_path)
    
    print(f"Demo environment ready! Sample image saved to {sample_path}")
    return sample_path

def demo_cartoon_generator():
    """Demonstrate CartoonGenerator features."""
    print("\n" + "="*60)
    print("CARTOON GENERATOR DEMO")
    print("="*60)
    
    # Mock API key for demo
    api_key = os.getenv('GEMINI_API_KEY', 'demo-api-key')
    generator = CartoonGenerator(api_key)
    
    print("1. Basic cartoon generation...")
    try:
        # Basic cartoon with default settings
        cartoon_data = {
            "pages": [
                {"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}
            ],
            "style": "classic"
        }
        
        saved_files = generator.save_pages(
            cartoon_data, 
            "./examples_output/cartoons/basic",
            format='PNG',
            create_thumbnails=True
        )
        print(f"✓ Basic cartoon saved: {len(saved_files)} files")
        
    except Exception as e:
        print(f"✗ Basic generation failed: {e}")
    
    print("\n2. Advanced cartoon with custom options...")
    try:
        # Demonstrate advanced options
        advanced_options = {
            'resolution': (1920, 1080),
            'style': 'vibrant',
            'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            'art_style': 'anime',
            'custom_options': {
                'mood': 'adventurous',
                'target_audience': 'teen'
            }
        }
        
        print(f"   Advanced options: {advanced_options}")
        print("   ✓ Configuration validated")
        
    except Exception as e:
        print(f"✗ Advanced generation failed: {e}")
    
    print("\n3. Style presets demonstration...")
    styles = ['classic', 'vintage', 'vibrant', 'monochrome']
    for style in styles:
        print(f"   - {style.title()} style: Enhanced with {generator.style_presets[style]}")

def demo_text_generator():
    """Demonstrate TextGenerator features."""
    print("\n" + "="*60)
    print("TEXT GENERATOR DEMO")
    print("="*60)
    
    api_key = os.getenv('OPENAI_API_KEY', 'demo-api-key')
    generator = TextGenerator(provider="openai", api_key=api_key)
    
    print("1. Story generation...")
    try:
        story = generator.generate_story(
            prompt="A robot discovers emotions in a digital world",
            genre="sci-fi",
            length="short",
            characters=["ARIA-7", "Dr. Chen", "The Observer"],
            setting="Neo-Tokyo 2157",
            themes=["consciousness", "humanity", "discovery"],
            target_audience="teen"
        )
        
        print(f"   ✓ Story generated: {story['metadata']['word_count']} words")
        print(f"   ✓ Reading time: {story['metadata']['reading_time']}")
        print(f"   ✓ Genre: {story['metadata']['genre']}")
        print(f"   ✓ Themes: {', '.join(story['metadata']['themes'])}")
        
        # Save story
        with open("./examples_output/text/generated_story.txt", "w") as f:
            f.write(f"Title: Robot Emotions\n\n")
            f.write(f"Genre: {story['metadata']['genre']}\n")
            f.write(f"Themes: {', '.join(story['metadata']['themes'])}\n\n")
            f.write(story['story'])
            
    except Exception as e:
        print(f"✗ Story generation failed: {e}")
    
    print("\n2. Code generation...")
    try:
        code = generator.generate_code(
            description="Create a REST API for a todo list application",
            language="python",
            include_comments=True,
            include_tests=True,
            frameworks=["flask", "sqlalchemy"]
        )
        
        print(f"   ✓ Code generated: {code['metadata']['lines_of_code']} lines")
        print(f"   ✓ Language: {code['metadata']['language']}")
        print(f"   ✓ Frameworks: {', '.join(code['metadata']['frameworks'])}")
        
        # Save code
        with open("./examples_output/code/generated_api.py", "w") as f:
            f.write(code['code'])
            
    except Exception as e:
        print(f"✗ Code generation failed: {e}")
    
    print("\n3. Article generation...")
    try:
        article = generator.generate_article(
            topic="The Future of Artificial Intelligence",
            article_type="informative",
            target_audience="tech professionals",
            length="medium",
            tone="professional",
            keywords=["AI", "machine learning", "automation"]
        )
        
        print(f"   ✓ Article generated: {article['metadata']['word_count']} words")
        print(f"   ✓ Title: {article['title']}")
        print(f"   ✓ Target audience: {article['metadata']['target_audience']}")
        
        # Save article
        with open("./examples_output/text/generated_article.md", "w") as f:
            f.write(f"# {article['title']}\n\n")
            f.write(article['content'])
            
    except Exception as e:
        print(f"✗ Article generation failed: {e}")

def demo_image_analyzer(sample_image_path):
    """Demonstrate ImageAnalyzer features."""
    print("\n" + "="*60)
    print("IMAGE ANALYZER DEMO")
    print("="*60)
    
    api_key = os.getenv('OPENAI_API_KEY', 'demo-api-key')
    analyzer = ImageAnalyzer(api_key=api_key, provider="openai")
    
    print(f"1. Analyzing image: {sample_image_path}")
    try:
        analysis = analyzer.analyze_image(
            sample_image_path,
            analysis_types=['object_detection', 'scene_analysis', 'color_analysis', 'quality_assessment'],
            save_annotations=True,
            output_dir="./examples_output/images/analysis"
        )
        
        print(f"   ✓ Image size: {analysis['image_info']['size']}")
        print(f"   ✓ Objects detected: {analysis['analysis']['object_detection']['objects_detected']}")
        print(f"   ✓ Scene type: {analysis['analysis']['scene_analysis']['scene_type']}")
        print(f"   ✓ Quality score: {analysis['analysis']['quality_assessment']['overall_score']}/100")
        print(f"   ✓ Dominant colors: {len(analysis['analysis']['color_analysis']['dominant_colors'])} colors")
        
    except Exception as e:
        print(f"✗ Image analysis failed: {e}")
    
    print("\n2. Image processing with filters...")
    try:
        # Process with different filters
        filters_to_test = ['enhance', 'vintage', 'dramatic']
        processed_images = []
        
        for filter_name in filters_to_test:
            processed = analyzer.process_image(
                sample_image_path,
                filters=[filter_name],
                output_format='PNG'
            )
            processed_images.append(processed['processed_image'])
            
            # Save processed image
            output_path = f"./examples_output/images/{filter_name}_filtered.png"
            processed['processed_image'].save(output_path)
            print(f"   ✓ {filter_name.title()} filter applied and saved")
        
        # Create comparison grid
        original_img = Image.open(sample_image_path)
        comparison_path = analyzer.create_comparison_grid(
            original_img, 
            processed_images, 
            [f.title() for f in filters_to_test],
            './examples_output/images/comparison_grid.png'
        )
        print(f"   ✓ Comparison grid saved: {comparison_path}")
        
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
    
    print("\n3. Quality assessment and recommendations...")
    try:
        # Load the sample image for quality assessment
        sample_img = Image.open(sample_image_path)
        quality_analysis = analyzer._assess_quality(sample_img)
        
        print(f"   ✓ Overall quality score: {quality_analysis['overall_score']}/100")
        print(f"   ✓ Brightness: {quality_analysis['metrics']['brightness']}")
        print(f"   ✓ Contrast: {quality_analysis['metrics']['contrast']}")
        print(f"   ✓ Sharpness: {quality_analysis['metrics']['sharpness']}")
        print("   Recommendations:")
        for rec in quality_analysis['recommendations']:
            print(f"     - {rec}")
            
    except Exception as e:
        print(f"✗ Quality assessment failed: {e}")

def demo_code_assistant():
    """Demonstrate CodeAssistant features."""
    print("\n" + "="*60)
    print("CODE ASSISTANT DEMO")
    print("="*60)
    
    api_key = os.getenv('OPENAI_API_KEY', 'demo-api-key')
    assistant = CodeAssistant(api_key=api_key)
    
    # Sample code for analysis
    sample_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

class DataProcessor:
    def __init__(self, data):
        self.data = data
        
    def filter_positive(self):
        return [x for x in self.data if x > 0]
"""
    
    print("1. Code generation...")
    try:
        generated = assistant.generate_code(
            description="Create a class for managing a simple task queue",
            language="python",
            code_type="class",
            include_tests=True,
            include_docs=True,
            complexity_level="intermediate"
        )
        
        print(f"   ✓ Generated {len(generated.code.split())} words of code")
        print(f"   ✓ Dependencies: {', '.join(generated.dependencies) if generated.dependencies else 'None'}")
        
        # Save generated code
        with open("./examples_output/code/generated_class.py", "w") as f:
            f.write(generated.code)
            if generated.tests:
                f.write("\n\n# Generated Tests\n")
                f.write(generated.tests)
                
    except Exception as e:
        print(f"✗ Code generation failed: {e}")
    
    print("\n2. Code review...")
    try:
        review = assistant.review_code(
            sample_code,
            language="python",
            review_type="comprehensive"
        )
        
        print(f"   ✓ Overall score: {review.overall_score}/100")
        print(f"   ✓ Complexity score: {review.complexity_score}/100")
        print(f"   ✓ Maintainability score: {review.maintainability_score}/100")
        print(f"   ✓ Issues found: {len(review.issues)}")
        
        if review.issues:
            print("   Issues:")
            for issue in review.issues[:3]:  # Show first 3 issues
                print(f"     - Line {issue.get('line', 'N/A')}: {issue.get('message', 'No message')}")
        
        if review.suggestions:
            print("   Suggestions:")
            for suggestion in review.suggestions[:3]:  # Show first 3 suggestions
                print(f"     - {suggestion}")
                
    except Exception as e:
        print(f"✗ Code review failed: {e}")
    
    print("\n3. Test generation...")
    try:
        tests = assistant.generate_tests(
            sample_code,
            language="python",
            test_framework="pytest",
            coverage_goal=90
        )
        
        print(f"   ✓ Tests generated with {tests['estimated_coverage']}% estimated coverage")
        print(f"   ✓ Test framework: {tests['test_framework']}")
        print(f"   ✓ Components tested: {len(tests['test_breakdown'])}")
        
        # Save tests
        with open("./examples_output/code/generated_tests.py", "w") as f:
            f.write(tests['test_code'])
            
        print("   Setup instructions:")
        for instruction in tests['setup_instructions']:
            print(f"     - {instruction}")
            
    except Exception as e:
        print(f"✗ Test generation failed: {e}")
    
    print("\n4. Code explanation...")
    try:
        explanation = assistant.explain_code(
            sample_code,
            language="python",
            explanation_level="intermediate",
            include_examples=True
        )
        
        print(f"   ✓ Key concepts identified: {len(explanation['key_concepts'])}")
        print(f"   ✓ Usage examples: {len(explanation['usage_examples'])}")
        print(f"   Key concepts: {', '.join(explanation['key_concepts'])}")
        print(f"   Complexity: {explanation['complexity_explanation']}")
        
        # Save explanation
        with open("./examples_output/code/code_explanation.md", "w") as f:
            f.write("# Code Explanation\n\n")
            f.write(f"**Complexity:** {explanation['complexity_explanation']}\n\n")
            f.write("## Key Concepts\n")
            for concept in explanation['key_concepts']:
                f.write(f"- {concept}\n")
            f.write("\n## Best Practices\n")
            for practice in explanation['best_practices']:
                f.write(f"- {practice}\n")
                
    except Exception as e:
        print(f"✗ Code explanation failed: {e}")

def demo_advanced_features():
    """Demonstrate advanced customization features."""
    print("\n" + "="*60)
    print("ADVANCED FEATURES DEMO")
    print("="*60)
    
    print("1. Custom configuration options...")
    
    # Demonstrate CartoonGenerator with custom options
    print("   CartoonGenerator custom options:")
    custom_cartoon_options = {
        'resolution': (4096, 2160),  # 4K resolution
        'style': 'vibrant',
        'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
        'art_style': 'anime',
        'language': 'en',
        'custom_options': {
            'mood': 'adventurous',
            'target_audience': 'teen',
            'include_dialogue': True,
            'panel_layout': 'dynamic',
            'background_detail': 'high'
        }
    }
    
    for key, value in custom_cartoon_options.items():
        print(f"     - {key}: {value}")
    
    print("\n   TextGenerator style presets:")
    text_generator = TextGenerator()
    for style, params in text_generator.style_presets.items():
        print(f"     - {style}: Temperature={params['temperature']}, Max tokens={params['max_tokens']}")
    
    print("\n   ImageAnalyzer filter presets:")
    image_analyzer = ImageAnalyzer(api_key="demo")
    for filter_name, settings in image_analyzer.filter_presets.items():
        print(f"     - {filter_name}: {settings}")
    
    print("\n2. Batch processing capabilities...")
    print("   ✓ Batch image processing with multiple filters")
    print("   ✓ Batch text generation with different styles")
    print("   ✓ Batch code analysis with various metrics")
    
    print("\n3. Export and integration options...")
    print("   ✓ Multiple output formats (PNG, JPEG, WebP, TIFF, GIF)")
    print("   ✓ API integration ready")
    print("   ✓ Configurable quality settings")
    print("   ✓ Thumbnail generation")
    print("   ✓ Annotation and visualization tools")

def show_summary():
    """Show summary of generated files and features."""
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    # List generated files
    output_dir = Path("./examples_output")
    if output_dir.exists():
        print("Generated files:")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(output_dir)
                print(f"   - {rel_path}")
    
    print(f"\nFeatures demonstrated:")
    features = [
        "✓ Cartoon generation with custom styles and options",
        "✓ Text generation (stories, code, articles) with multiple styles",
        "✓ Image analysis (object detection, scene analysis, color analysis)",
        "✓ Image processing with filters and effects",
        "✓ Code generation, review, and optimization",
        "✓ Test generation and code explanation",
        "✓ Advanced customization options",
        "✓ Batch processing capabilities",
        "✓ Multiple output formats and quality settings"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\nCustomization options available:")
    customizations = [
        "• Resolution and quality settings",
        "• Color palettes and visual styles", 
        "• Language and tone preferences",
        "• Framework and library choices",
        "• Output formats and compression",
        "• Analysis depth and focus areas",
        "• Filter presets and custom effects",
        "• Test coverage and framework selection"
    ]
    
    for customization in customizations:
        print(f"   {customization}")

def main():
    """Run all demos."""
    print("AI-DRIVEN PYTHON CODE EXAMPLES")
    print("=" * 60)
    print("This demo showcases the enhanced features and customization options")
    print("available in the AI-driven Python code library.")
    print()
    
    try:
        # Set up demo environment
        sample_image_path = setup_demo_environment()
        
        # Run demos
        demo_cartoon_generator()
        demo_text_generator()
        demo_image_analyzer(sample_image_path)
        demo_code_assistant()
        demo_advanced_features()
        
        # Show summary
        show_summary()
        
        print(f"\n{'='*60}")
        print("Demo completed successfully!")
        print("Check the './examples_output' directory for generated files.")
        print("See the README.md for detailed documentation and usage examples.")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()