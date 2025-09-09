import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """Advanced image analysis with AI-powered insights and customizable processing."""
    
    def __init__(self, api_key: str, provider: str = "openai", **kwargs):
        """Initialize the ImageAnalyzer with AI provider configuration.
        
        Args:
            api_key: API key for the AI service
            provider: AI provider ('openai', 'google', 'anthropic')
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.provider = provider.lower()
        self.config = kwargs
        
        # Analysis capabilities
        self.analysis_types = {
            'object_detection': {'confidence_threshold': 0.5, 'max_objects': 50},
            'scene_analysis': {'detail_level': 'comprehensive'},
            'text_extraction': {'languages': ['en', 'es', 'fr', 'de', 'zh']},
            'emotion_detection': {'include_confidence': True},
            'color_analysis': {'palette_size': 10, 'include_percentages': True},
            'quality_assessment': {'metrics': ['sharpness', 'brightness', 'contrast']},
            'style_analysis': {'categories': ['artistic', 'photographic', 'digital']}
        }
        
        # Filter presets
        self.filter_presets = {
            'enhance': {'brightness': 1.1, 'contrast': 1.2, 'saturation': 1.1},
            'vintage': {'brightness': 0.9, 'contrast': 1.3, 'saturation': 0.8, 'sepia': True},
            'dramatic': {'brightness': 0.8, 'contrast': 1.5, 'saturation': 1.2},
            'soft': {'brightness': 1.05, 'contrast': 0.9, 'blur': 1},
            'sharp': {'brightness': 1.0, 'contrast': 1.1, 'sharpen': 2}
        }
        
    def analyze_image(self, 
                     image_path: str,
                     analysis_types: List[str] = None,
                     custom_options: Dict = None,
                     save_annotations: bool = False,
                     output_dir: str = "./analysis_output") -> Dict:
        """Perform comprehensive image analysis.
        
        Args:
            image_path: Path to the image file
            analysis_types: List of analysis types to perform
            custom_options: Custom options for specific analysis types
            save_annotations: Whether to save annotated images
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary containing analysis results
        """
        if analysis_types is None:
            analysis_types = ['object_detection', 'scene_analysis', 'color_analysis']
            
        # Load and prepare image
        image = Image.open(image_path)
        image_data = self._prepare_image_data(image)
        
        # Initialize results
        results = {
            'image_info': {
                'filename': os.path.basename(image_path),
                'size': image.size,
                'mode': image.mode,
                'format': image.format
            },
            'analysis': {}
        }
        
        # Perform requested analyses
        for analysis_type in analysis_types:
            logger.info(f"Performing {analysis_type}...")
            
            if analysis_type == 'object_detection':
                results['analysis'][analysis_type] = self._detect_objects(image, custom_options)
            elif analysis_type == 'scene_analysis':
                results['analysis'][analysis_type] = self._analyze_scene(image, custom_options)
            elif analysis_type == 'text_extraction':
                results['analysis'][analysis_type] = self._extract_text(image, custom_options)
            elif analysis_type == 'emotion_detection':
                results['analysis'][analysis_type] = self._detect_emotions(image, custom_options)
            elif analysis_type == 'color_analysis':
                results['analysis'][analysis_type] = self._analyze_colors(image, custom_options)
            elif analysis_type == 'quality_assessment':
                results['analysis'][analysis_type] = self._assess_quality(image, custom_options)
            elif analysis_type == 'style_analysis':
                results['analysis'][analysis_type] = self._analyze_style(image, custom_options)
                
        # Save annotations if requested
        if save_annotations:
            os.makedirs(output_dir, exist_ok=True)
            self._save_annotations(image, results, output_dir)
            
        return results
        
    def process_image(self,
                     image_path: str,
                     filters: List[str] = None,
                     custom_filters: Dict = None,
                     resize: Optional[Tuple[int, int]] = None,
                     output_format: str = "PNG",
                     quality: int = 95) -> Dict:
        """Process image with various filters and transformations.
        
        Args:
            image_path: Path to the image file
            filters: List of filter presets to apply
            custom_filters: Custom filter parameters
            resize: New size as (width, height)
            output_format: Output image format
            quality: JPEG quality (1-100)
            
        Returns:
            Dictionary containing processed image info
        """
        image = Image.open(image_path)
        original_size = image.size
        
        # Apply resize if specified
        if resize:
            image = image.resize(resize, Image.Resampling.LANCZOS)
            
        # Apply filters
        if filters:
            for filter_name in filters:
                if filter_name in self.filter_presets:
                    image = self._apply_filter_preset(image, filter_name)
                    
        # Apply custom filters
        if custom_filters:
            image = self._apply_custom_filters(image, custom_filters)
            
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        filter_suffix = "_".join(filters) if filters else "processed"
        output_filename = f"{base_name}_{filter_suffix}.{output_format.lower()}"
        
        return {
            'processed_image': image,
            'original_size': original_size,
            'new_size': image.size,
            'filters_applied': filters or [],
            'output_filename': output_filename,
            'processing_info': {
                'resize_ratio': image.size[0] / original_size[0] if resize else 1.0,
                'format': output_format,
                'quality': quality if output_format.upper() == 'JPEG' else None
            }
        }
        
    def batch_process(self,
                     input_dir: str,
                     output_dir: str,
                     operations: Dict,
                     file_extensions: List[str] = None) -> Dict:
        """Process multiple images in batch.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for processed images
            operations: Dictionary defining operations to perform
            file_extensions: List of file extensions to process
            
        Returns:
            Dictionary containing batch processing results
        """
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            for file_path in Path(input_dir).glob(f"*{ext}"):
                image_files.append(str(file_path))
                
        results = {
            'total_files': len(image_files),
            'processed_files': 0,
            'failed_files': 0,
            'processing_details': []
        }
        
        for image_path in image_files:
            try:
                # Process image based on operations
                if 'analyze' in operations:
                    analysis = self.analyze_image(
                        image_path,
                        analysis_types=operations['analyze'].get('types', ['scene_analysis']),
                        save_annotations=operations['analyze'].get('save_annotations', False),
                        output_dir=output_dir
                    )
                    
                if 'process' in operations:
                    processed = self.process_image(
                        image_path,
                        filters=operations['process'].get('filters', []),
                        resize=operations['process'].get('resize'),
                        output_format=operations['process'].get('format', 'PNG')
                    )
                    
                    # Save processed image
                    output_path = os.path.join(output_dir, processed['output_filename'])
                    processed['processed_image'].save(output_path)
                    
                results['processed_files'] += 1
                results['processing_details'].append({
                    'file': os.path.basename(image_path),
                    'status': 'success'
                })
                
            except Exception as e:
                results['failed_files'] += 1
                results['processing_details'].append({
                    'file': os.path.basename(image_path),
                    'status': 'failed',
                    'error': str(e)
                })
                logger.error(f"Failed to process {image_path}: {e}")
                
        return results
        
    def create_comparison_grid(self,
                              original_image: Image.Image,
                              processed_images: List[Image.Image],
                              labels: List[str],
                              output_path: str) -> str:
        """Create a comparison grid showing original and processed versions.
        
        Args:
            original_image: Original image
            processed_images: List of processed image variants
            labels: Labels for each processed image
            output_path: Path to save the comparison grid
            
        Returns:
            Path to the saved comparison grid
        """
        # Calculate grid dimensions
        total_images = len(processed_images) + 1  # +1 for original
        cols = min(4, total_images)
        rows = (total_images + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if total_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        # Add original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Add processed images
        for i, (processed_img, label) in enumerate(zip(processed_images, labels)):
            axes[i+1].imshow(processed_img)
            axes[i+1].set_title(label, fontsize=12)
            axes[i+1].axis('off')
            
        # Hide unused subplots
        for i in range(total_images, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    def _prepare_image_data(self, image: Image.Image) -> Dict:
        """Prepare image data for API calls."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Encode to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'base64': img_str,
            'size': image.size,
            'mode': image.mode
        }
        
    def _detect_objects(self, image: Image.Image, options: Dict = None) -> Dict:
        """Simulate object detection (replace with actual API call)."""
        # Mock object detection results
        mock_objects = [
            {'label': 'person', 'confidence': 0.95, 'bbox': [100, 50, 200, 300]},
            {'label': 'car', 'confidence': 0.87, 'bbox': [300, 150, 500, 250]},
            {'label': 'tree', 'confidence': 0.76, 'bbox': [50, 10, 150, 200]}
        ]
        
        return {
            'objects_detected': len(mock_objects),
            'objects': mock_objects,
            'confidence_threshold': options.get('confidence_threshold', 0.5) if options else 0.5
        }
        
    def _analyze_scene(self, image: Image.Image, options: Dict = None) -> Dict:
        """Simulate scene analysis (replace with actual API call)."""
        return {
            'scene_type': 'urban street',
            'setting': 'outdoor',
            'time_of_day': 'afternoon',
            'weather': 'clear',
            'dominant_colors': ['blue', 'gray', 'green'],
            'objects_count': 15,
            'people_count': 3,
            'activity_level': 'moderate',
            'mood': 'neutral',
            'description': 'A busy urban street with pedestrians and vehicles during daytime.'
        }
        
    def _extract_text(self, image: Image.Image, options: Dict = None) -> Dict:
        """Simulate text extraction (replace with actual OCR)."""
        return {
            'text_found': True,
            'extracted_text': 'Sample text extracted from image',
            'text_regions': [
                {'text': 'STOP', 'bbox': [150, 100, 200, 130], 'confidence': 0.98},
                {'text': 'Main St', 'bbox': [300, 200, 380, 220], 'confidence': 0.92}
            ],
            'languages_detected': ['en']
        }
        
    def _detect_emotions(self, image: Image.Image, options: Dict = None) -> Dict:
        """Simulate emotion detection (replace with actual API call)."""
        return {
            'faces_detected': 2,
            'emotions': [
                {'face_id': 1, 'emotion': 'happy', 'confidence': 0.85, 'bbox': [120, 80, 180, 150]},
                {'face_id': 2, 'emotion': 'neutral', 'confidence': 0.92, 'bbox': [300, 60, 360, 140]}
            ],
            'dominant_emotion': 'happy'
        }
        
    def _analyze_colors(self, image: Image.Image, options: Dict = None) -> Dict:
        """Analyze color composition of the image."""
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Calculate color statistics
        pixels = img_array.reshape(-1, 3)
        
        # Get dominant colors (simplified)
        unique_colors = np.unique(pixels, axis=0)[:10]  # Top 10 unique colors
        
        # Calculate average color
        avg_color = np.mean(pixels, axis=0).astype(int)
        
        # Color temperature estimation (simplified)
        r, g, b = avg_color
        color_temp = 'warm' if r > b else 'cool'
        
        return {
            'dominant_colors': [
                {'color': [int(c[0]), int(c[1]), int(c[2])], 'hex': f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"}
                for c in unique_colors[:5]
            ],
            'average_color': {'rgb': avg_color.tolist(), 'hex': f"#{r:02x}{g:02x}{b:02x}"},
            'color_temperature': color_temp,
            'brightness': int(np.mean(pixels)),
            'contrast': int(np.std(pixels)),
            'color_diversity': len(unique_colors)
        }
        
    def _assess_quality(self, image: Image.Image, options: Dict = None) -> Dict:
        """Assess image quality metrics."""
        img_array = np.array(image.convert('L'))  # Convert to grayscale for analysis
        
        # Calculate sharpness (simplified)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        sharpness = np.sum(np.abs(np.convolve(img_array.flatten(), laplacian.flatten(), mode='valid')))
        
        # Normalize metrics to 0-100 scale
        brightness = int(np.mean(img_array))
        contrast = int(np.std(img_array))
        sharpness_score = min(100, int(sharpness / 1000))
        
        return {
            'overall_score': int((brightness + contrast + sharpness_score) / 3),
            'metrics': {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness_score
            },
            'recommendations': self._get_quality_recommendations(brightness, contrast, sharpness_score)
        }
        
    def _analyze_style(self, image: Image.Image, options: Dict = None) -> Dict:
        """Analyze artistic style of the image."""
        # Simplified style analysis
        img_array = np.array(image)
        
        # Basic style indicators
        edge_density = np.std(img_array)
        color_variation = len(np.unique(img_array.reshape(-1, 3), axis=0))
        
        # Determine style category
        if edge_density > 50 and color_variation > 1000:
            style_category = 'photographic'
        elif edge_density < 30:
            style_category = 'artistic'
        else:
            style_category = 'digital'
            
        return {
            'style_category': style_category,
            'artistic_elements': {
                'edge_density': float(edge_density),
                'color_complexity': color_variation,
                'texture_richness': 'high' if edge_density > 40 else 'medium' if edge_density > 20 else 'low'
            },
            'style_confidence': 0.75
        }
        
    def _apply_filter_preset(self, image: Image.Image, preset_name: str) -> Image.Image:
        """Apply a filter preset to an image."""
        preset = self.filter_presets[preset_name]
        
        # Apply brightness
        if 'brightness' in preset:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(preset['brightness'])
            
        # Apply contrast
        if 'contrast' in preset:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(preset['contrast'])
            
        # Apply saturation
        if 'saturation' in preset:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(preset['saturation'])
            
        # Apply blur
        if 'blur' in preset:
            image = image.filter(ImageFilter.GaussianBlur(radius=preset['blur']))
            
        # Apply sharpening
        if 'sharpen' in preset:
            image = image.filter(ImageFilter.UnsharpMask(radius=preset['sharpen']))
            
        # Apply sepia effect
        if preset.get('sepia', False):
            image = self._apply_sepia(image)
            
        return image
        
    def _apply_custom_filters(self, image: Image.Image, filters: Dict) -> Image.Image:
        """Apply custom filter parameters."""
        for filter_type, value in filters.items():
            if filter_type == 'brightness':
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(value)
            elif filter_type == 'contrast':
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(value)
            elif filter_type == 'saturation':
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(value)
            elif filter_type == 'blur':
                image = image.filter(ImageFilter.GaussianBlur(radius=value))
                
        return image
        
    def _apply_sepia(self, image: Image.Image) -> Image.Image:
        """Apply sepia tone effect."""
        pixels = np.array(image)
        
        # Sepia transformation matrix
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        sepia_img = pixels @ sepia_filter.T
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(sepia_img)
        
    def _get_quality_recommendations(self, brightness: int, contrast: int, sharpness: int) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if brightness < 80:
            recommendations.append("Increase brightness for better visibility")
        elif brightness > 200:
            recommendations.append("Reduce brightness to avoid overexposure")
            
        if contrast < 30:
            recommendations.append("Increase contrast for more definition")
        elif contrast > 80:
            recommendations.append("Reduce contrast for softer appearance")
            
        if sharpness < 30:
            recommendations.append("Apply sharpening filter to enhance details")
            
        if not recommendations:
            recommendations.append("Image quality is good - no adjustments needed")
            
        return recommendations
        
    def _save_annotations(self, image: Image.Image, results: Dict, output_dir: str):
        """Save annotated versions of the image."""
        # Create annotated image for object detection
        if 'object_detection' in results['analysis']:
            annotated = self._create_object_annotations(image, results['analysis']['object_detection'])
            annotated.save(os.path.join(output_dir, 'annotated_objects.png'))
            
        # Create color analysis visualization
        if 'color_analysis' in results['analysis']:
            color_viz = self._create_color_visualization(results['analysis']['color_analysis'])
            color_viz.save(os.path.join(output_dir, 'color_analysis.png'))
            
    def _create_object_annotations(self, image: Image.Image, detection_results: Dict) -> Image.Image:
        """Create annotated image with object detection results."""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        for obj in detection_results['objects']:
            bbox = obj['bbox']
            label = f"{obj['label']} ({obj['confidence']:.2f})"
            
            # Draw bounding box
            draw.rectangle(bbox, outline='red', width=2)
            
            # Draw label
            draw.text((bbox[0], bbox[1]-20), label, fill='red')
            
        return annotated
        
    def _create_color_visualization(self, color_analysis: Dict) -> Image.Image:
        """Create color palette visualization."""
        palette_img = Image.new('RGB', (500, 100), 'white')
        draw = ImageDraw.Draw(palette_img)
        
        colors = color_analysis['dominant_colors']
        width_per_color = 500 // len(colors)
        
        for i, color_info in enumerate(colors):
            color = tuple(color_info['color'])
            x1 = i * width_per_color
            x2 = (i + 1) * width_per_color
            draw.rectangle([x1, 0, x2, 100], fill=color)
            
        return palette_img

# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Initialize analyzer
    analyzer = ImageAnalyzer(api_key="your-api-key", provider="openai")
    
    # Create sample image for testing
    sample_img = Image.new('RGB', (400, 300), 'lightblue')
    draw = ImageDraw.Draw(sample_img)
    draw.rectangle([50, 50, 150, 150], fill='red')
    draw.rectangle([200, 100, 350, 200], fill='green')
    sample_path = "./sample_image.png"
    sample_img.save(sample_path)
    
    print("Analyzing sample image...")
    
    # Example 1: Comprehensive analysis
    analysis = analyzer.analyze_image(
        sample_path,
        analysis_types=['object_detection', 'scene_analysis', 'color_analysis', 'quality_assessment'],
        save_annotations=True
    )
    print(f"Analysis completed. Found {analysis['analysis']['object_detection']['objects_detected']} objects")
    
    # Example 2: Image processing
    processed = analyzer.process_image(
        sample_path,
        filters=['enhance', 'dramatic'],
        resize=(800, 600),
        output_format='PNG'
    )
    print(f"Image processed. New size: {processed['new_size']}")
    
    # Example 3: Create comparison grid
    processed_images = []
    labels = []
    
    for filter_name in ['enhance', 'vintage', 'dramatic']:
        filtered = analyzer.process_image(sample_path, filters=[filter_name])
        processed_images.append(filtered['processed_image'])
        labels.append(filter_name.title())
        
    comparison_path = analyzer.create_comparison_grid(
        sample_img, processed_images, labels, './comparison_grid.png'
    )
    print(f"Comparison grid saved: {comparison_path}")
    
    # Clean up
    os.remove(sample_path)