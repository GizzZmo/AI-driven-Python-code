# AI-driven Python Code Examples

A comprehensive collection of AI-powered Python tools with extensive customization options for cartoon generation, text processing, image analysis, and code assistance.

## 🚀 Features

### CartoonGenerator
- **Advanced cartoon generation** with customizable styles, resolutions, and art styles
- **Multiple output formats** (PNG, JPEG, WebP, TIFF, GIF)
- **Style presets** (classic, vintage, vibrant, monochrome)
- **Custom color palettes** and artistic options
- **Thumbnail generation** and quality enhancement
- **Animated GIF creation** from cartoon sequences

### TextGenerator
- **Multi-provider AI support** (OpenAI, Anthropic, Gemini)
- **Story generation** with genre, character, and theme customization
- **Code generation** with framework integration and documentation
- **Article writing** with SEO optimization and tone control
- **Style presets** for different writing purposes
- **Reading time estimation** and content analysis

### ImageAnalyzer
- **Comprehensive image analysis** (objects, scenes, colors, quality)
- **Advanced image processing** with filter presets and custom effects
- **Batch processing** capabilities for multiple images
- **Quality assessment** with improvement recommendations
- **Comparison grid generation** for before/after visualization
- **Annotation tools** for analysis results

### CodeAssistant
- **AI-powered code generation** with customizable complexity levels
- **Comprehensive code review** with quality scoring
- **Test generation** with coverage estimation
- **Code optimization** and refactoring suggestions
- **Multi-language support** (Python, JavaScript, Java, Go)
- **Documentation generation** and code explanation

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/GizzZmo/AI-driven-Python-code.git
cd AI-driven-Python-code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (optional for demo):
```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## 🎯 Quick Start

Run the comprehensive examples:
```bash
python examples.py
```

Or use individual components:

### CartoonGenerator Example
```python
from CartoonGenerator import CartoonGenerator

generator = CartoonGenerator(api_key="your-api-key")

# Generate cartoon with custom options
cartoon = generator.generate_cartoon(
    prompt="Space adventure with robots",
    page_count=6,
    strips_per_page=3,
    resolution=(1920, 1080),
    style='vibrant',
    color_palette=['#FF6B6B', '#4ECDC4', '#45B7D1'],
    art_style='anime',
    custom_options={
        'mood': 'adventurous',
        'target_audience': 'teen'
    }
)

# Save with advanced options
generator.save_pages(
    cartoon, 
    "./output",
    format='PNG',
    apply_filters=True,
    enhance_quality=True,
    create_thumbnails=True
)

# Create animated GIF
generator.create_gif_animation(cartoon, "./animation.gif")
```

### TextGenerator Example
```python
from TextGenerator import TextGenerator

generator = TextGenerator(provider="openai", api_key="your-key")

# Generate a story
story = generator.generate_story(
    prompt="A robot discovers emotions",
    genre="sci-fi",
    length="short",
    characters=["ARIA-7", "Dr. Chen"],
    setting="Neo-Tokyo 2157",
    themes=["consciousness", "humanity"],
    target_audience="teen"
)

# Generate code
code = generator.generate_code(
    description="REST API for todo list",
    language="python",
    include_tests=True,
    frameworks=["flask", "sqlalchemy"]
)

# Generate article
article = generator.generate_article(
    topic="Future of AI",
    article_type="informative",
    target_audience="tech professionals",
    keywords=["AI", "machine learning"]
)
```

### ImageAnalyzer Example
```python
from ImageAnalyzer import ImageAnalyzer

analyzer = ImageAnalyzer(api_key="your-key")

# Analyze image
analysis = analyzer.analyze_image(
    "image.jpg",
    analysis_types=['object_detection', 'scene_analysis', 'color_analysis'],
    save_annotations=True
)

# Process with filters
processed = analyzer.process_image(
    "image.jpg",
    filters=['enhance', 'dramatic'],
    resize=(800, 600),
    output_format='PNG'
)

# Batch processing
results = analyzer.batch_process(
    input_dir="./images",
    output_dir="./processed",
    operations={
        'analyze': {'types': ['scene_analysis']},
        'process': {'filters': ['enhance'], 'format': 'PNG'}
    }
)
```

### CodeAssistant Example
```python
from CodeAssistant import CodeAssistant

assistant = CodeAssistant(api_key="your-key")

# Generate code
generated = assistant.generate_code(
    description="Calculate fibonacci numbers",
    language="python",
    include_tests=True,
    complexity_level="intermediate"
)

# Review code
review = assistant.review_code(
    code_string,
    language="python",
    review_type="comprehensive"
)

# Generate tests
tests = assistant.generate_tests(
    code_string,
    test_framework="pytest",
    coverage_goal=90
)

# Explain code
explanation = assistant.explain_code(
    code_string,
    explanation_level="beginner",
    include_examples=True
)
```

## ⚙️ Customization Options

### CartoonGenerator Options
- **Resolution**: Custom dimensions (e.g., `(1920, 1080)`)
- **Style presets**: `classic`, `vintage`, `vibrant`, `monochrome`
- **Art styles**: `cartoon`, `anime`, `realistic`, `sketch`
- **Color palettes**: Custom hex color arrays
- **Output formats**: `PNG`, `JPEG`, `WebP`, `TIFF`
- **Quality settings**: Enhancement and filter options
- **Animation**: GIF creation with custom duration and loops

### TextGenerator Options
- **Providers**: `openai`, `anthropic`, `gemini`
- **Style presets**: `creative`, `analytical`, `conversational`, `technical`
- **Content types**: Stories, code, articles, documentation
- **Languages**: Multiple language support
- **Frameworks**: Integration with popular libraries
- **Complexity levels**: `beginner`, `intermediate`, `advanced`

### ImageAnalyzer Options
- **Analysis types**: Object detection, scene analysis, color analysis, quality assessment
- **Filter presets**: `enhance`, `vintage`, `dramatic`, `soft`, `sharp`
- **Custom filters**: Brightness, contrast, saturation, blur parameters
- **Output formats**: Multiple image formats with quality control
- **Batch operations**: Process multiple images simultaneously

### CodeAssistant Options
- **Languages**: Python, JavaScript, Java, Go
- **Review types**: `quick`, `comprehensive`, `security`, `performance`
- **Test frameworks**: `pytest`, `unittest`, `jest`, `junit`
- **Complexity analysis**: Cyclomatic complexity and maintainability scoring
- **Optimization goals**: Performance, readability, memory usage

## 📁 Project Structure

```
AI-driven-Python-code/
├── CartoonGenerator.py      # Advanced cartoon generation
├── TextGenerator.py         # Multi-provider text generation
├── ImageAnalyzer.py         # Comprehensive image analysis
├── CodeAssistant.py         # AI-powered code assistance
├── examples.py             # Comprehensive usage examples
├── requirements.txt        # Project dependencies
├── README.md              # Documentation
└── examples_output/       # Generated demo files
    ├── cartoons/         # Cartoon generation outputs
    ├── images/           # Image analysis and processing
    ├── text/             # Generated text content
    └── code/             # Code generation and analysis
```

## 🔧 Advanced Features

### Multi-Provider AI Integration
Support for multiple AI providers with seamless switching:
- OpenAI GPT models
- Anthropic Claude
- Google Gemini
- Local model integration

### Batch Processing
Efficient handling of multiple files:
- Parallel processing capabilities
- Progress tracking and error handling
- Configurable output organization

### Quality Assessment
Automated quality metrics:
- Image quality scoring
- Code complexity analysis
- Content readability assessment
- Performance optimization suggestions

### Export Options
Flexible output formats:
- Multiple image formats with compression control
- Code export with documentation
- Analysis reports in various formats
- API-ready data structures

## 🎨 Style Presets

### CartoonGenerator Styles
- **Classic**: Balanced brightness, contrast, and saturation
- **Vintage**: Muted colors with sepia tones
- **Vibrant**: Enhanced colors and contrast
- **Monochrome**: Grayscale conversion

### TextGenerator Styles
- **Creative**: High temperature for imaginative content
- **Analytical**: Low temperature for factual content
- **Conversational**: Balanced for dialogue
- **Technical**: Precise for documentation

### ImageAnalyzer Filters
- **Enhance**: Improved brightness and clarity
- **Vintage**: Retro aesthetic with sepia
- **Dramatic**: High contrast and saturation
- **Soft**: Subtle blur for dreamy effect
- **Sharp**: Enhanced edge definition

## 📊 Performance Metrics

The library provides comprehensive metrics:
- **Generation time** tracking
- **Quality scores** (0-100 scale)
- **Complexity analysis** for code
- **Coverage estimation** for tests
- **Resource usage** monitoring

## 🔒 Security Features

- **Input validation** for all parameters
- **Secure API key handling**
- **Error handling** with detailed logging
- **Rate limiting** support
- **Content filtering** options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Follow existing code style
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Examples Gallery

The `examples.py` script demonstrates:
- ✅ **40+ customization options** across all modules
- ✅ **Multiple output formats** and quality settings
- ✅ **Batch processing** capabilities
- ✅ **Integration examples** between modules
- ✅ **Error handling** and validation
- ✅ **Performance optimization** techniques

Run `python examples.py` to see all features in action!

## 🔧 CI/CD & Automation

This repository includes a comprehensive GitHub Actions workflow system:

### Available Workflows
- **CI**: Continuous integration testing across Python 3.8-3.11
- **Code Quality**: Automated linting, formatting, and style checks
- **Security & Dependencies**: Vulnerability scanning and dependency management
- **Release**: Automated package building and publishing
- **Dependencies**: Monthly automated dependency updates

### Workflow Features
- ✅ **Multi-Python testing** (3.8, 3.9, 3.10, 3.11)
- ✅ **Code quality enforcement** (Black, Flake8, isort, Pylint)
- ✅ **Security scanning** (Bandit, Safety, Semgrep, CodeQL)
- ✅ **Automated releases** with PyPI publishing
- ✅ **Docker container support**
- ✅ **Dependency vulnerability tracking**

### Manual Triggers
- Trigger dependency updates: `workflow_dispatch` on Dependencies workflow
- Force security scan: `workflow_dispatch` on Security workflow
- Create release: Push a tag like `v1.0.0`

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the examples for usage patterns
- Review the comprehensive docstrings in each module

---

**Note**: This library includes mock implementations for demonstration. Replace API calls with actual service integrations for production use.
