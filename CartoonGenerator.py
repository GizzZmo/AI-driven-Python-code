import requests
import json
import base64
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import logging
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CartoonGenerator:
    """Advanced cartoon generator with extensive customization options."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.gemini.com/v1/"):
        """Initialize the CartoonGenerator with API credentials and configuration.
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API endpoint
        """
        self.api_key = api_key
        self.base_url = base_url
        self.supported_formats = ['PNG', 'JPEG', 'WebP', 'TIFF']
        self.style_presets = {
            'classic': {'brightness': 1.1, 'contrast': 1.2, 'saturation': 1.3},
            'vintage': {'brightness': 0.9, 'contrast': 1.4, 'saturation': 0.8},
            'vibrant': {'brightness': 1.2, 'contrast': 1.3, 'saturation': 1.5},
            'monochrome': {'brightness': 1.0, 'contrast': 1.1, 'saturation': 0.0}
        }
        
    def generate_cartoon(self, 
                        prompt: str, 
                        page_count: int = 12, 
                        strips_per_page: int = 4,
                        resolution: Tuple[int, int] = (1024, 768),
                        style: str = 'classic',
                        color_palette: Optional[List[str]] = None,
                        art_style: str = 'cartoon',
                        language: str = 'en',
                        custom_options: Optional[Dict] = None) -> Dict:
        """Generate a cartoon with extensive customization options.
        
        Args:
            prompt: Description of the cartoon to generate
            page_count: Number of pages to generate
            strips_per_page: Number of comic strips per page
            resolution: Output resolution as (width, height)
            style: Style preset ('classic', 'vintage', 'vibrant', 'monochrome')
            color_palette: List of hex color codes for the palette
            art_style: Art style ('cartoon', 'anime', 'realistic', 'sketch')
            language: Language for text in cartoon
            custom_options: Additional custom parameters
            
        Returns:
            Dictionary containing cartoon data
            
        Raises:
            Exception: If generation fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Build request payload with custom options
        payload = {
            "prompt": prompt,
            "page_count": page_count,
            "strips_per_page": strips_per_page,
            "resolution": {"width": resolution[0], "height": resolution[1]},
            "style": style,
            "art_style": art_style,
            "language": language
        }
        
        if color_palette:
            payload["color_palette"] = color_palette
            
        if custom_options:
            payload.update(custom_options)
        
        try:
            logger.info(f"Generating cartoon with {page_count} pages...")
            response = requests.post(
                f"{self.base_url}generate",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Cartoon generated successfully!")
                return response.json()
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                raise Exception(f"Error generating cartoon: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Network error: {e}")
            
    def save_pages(self, 
                   cartoon_data: Dict, 
                   output_dir: str,
                   format: str = 'PNG',
                   apply_filters: bool = True,
                   enhance_quality: bool = True,
                   create_thumbnails: bool = False,
                   thumbnail_size: Tuple[int, int] = (200, 150)) -> List[str]:
        """Save cartoon pages with advanced processing options.
        
        Args:
            cartoon_data: Cartoon data from generate_cartoon()
            output_dir: Directory to save files
            format: Image format ('PNG', 'JPEG', 'WebP', 'TIFF')
            apply_filters: Whether to apply style filters
            enhance_quality: Whether to enhance image quality
            create_thumbnails: Whether to create thumbnail versions
            thumbnail_size: Size for thumbnails as (width, height)
            
        Returns:
            List of saved file paths
            
        Raises:
            Exception: If saving fails
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Use one of {self.supported_formats}")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        if create_thumbnails:
            os.makedirs(f"{output_dir}/thumbnails", exist_ok=True)
            
        saved_files = []
        
        for i, page in enumerate(cartoon_data.get("pages", [])):
            try:
                # Decode image data
                image_data = base64.b64decode(page["image"])
                image = Image.open(io.BytesIO(image_data))
                
                # Apply enhancements if requested
                if enhance_quality:
                    image = self._enhance_image(image)
                    
                if apply_filters:
                    image = self._apply_style_filters(image, cartoon_data.get("style", "classic"))
                
                # Save front and back sides
                front_side = image.crop((0, 0, image.width//2, image.height))
                back_side = image.crop((image.width//2, 0, image.width, image.height))
                
                # Save main images
                front_path = f"{output_dir}/page_{i+1}_front.{format.lower()}"
                back_path = f"{output_dir}/page_{i+1}_back.{format.lower()}"
                
                front_side.save(front_path, format=format, quality=95 if format == 'JPEG' else None)
                back_side.save(back_path, format=format, quality=95 if format == 'JPEG' else None)
                
                saved_files.extend([front_path, back_path])
                
                # Create thumbnails if requested
                if create_thumbnails:
                    front_thumb = front_side.copy()
                    back_thumb = back_side.copy()
                    front_thumb.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    back_thumb.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    
                    front_thumb_path = f"{output_dir}/thumbnails/page_{i+1}_front_thumb.{format.lower()}"
                    back_thumb_path = f"{output_dir}/thumbnails/page_{i+1}_back_thumb.{format.lower()}"
                    
                    front_thumb.save(front_thumb_path, format=format)
                    back_thumb.save(back_thumb_path, format=format)
                    
                    saved_files.extend([front_thumb_path, back_thumb_path])
                
                logger.info(f"Saved page {i+1}")
                
            except Exception as e:
                logger.error(f"Error saving page {i+1}: {e}")
                continue
                
        logger.info(f"Saved {len(saved_files)} files to {output_dir}")
        return saved_files
        
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply quality enhancements to an image."""
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        return image
        
    def _apply_style_filters(self, image: Image.Image, style: str) -> Image.Image:
        """Apply style-specific filters to an image."""
        if style not in self.style_presets:
            return image
            
        preset = self.style_presets[style]
        
        # Apply brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(preset['brightness'])
        
        # Apply contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(preset['contrast'])
        
        # Apply saturation (color)
        if preset['saturation'] == 0.0:
            # Convert to grayscale for monochrome
            image = image.convert('L').convert('RGB')
        else:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(preset['saturation'])
            
        return image
        
    def create_gif_animation(self, 
                           cartoon_data: Dict, 
                           output_path: str,
                           duration: int = 500,
                           loop: int = 0) -> str:
        """Create an animated GIF from cartoon pages.
        
        Args:
            cartoon_data: Cartoon data from generate_cartoon()
            output_path: Path for the output GIF file
            duration: Duration between frames in milliseconds
            loop: Number of loops (0 for infinite)
            
        Returns:
            Path to the created GIF file
        """
        frames = []
        
        for page in cartoon_data.get("pages", []):
            image_data = base64.b64decode(page["image"])
            image = Image.open(io.BytesIO(image_data))
            
            # Use front side for animation
            front_side = image.crop((0, 0, image.width//2, image.height))
            frames.append(front_side)
            
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=loop
            )
            logger.info(f"Created animated GIF: {output_path}")
            
        return output_path

# Example usage with enhanced features
if __name__ == "__main__":
    # Get API key from user input or environment variable
    import os
    
    api_key = os.getenv('GEMINI_API_KEY') or input("Enter your Gemini API key: ")
    generator = CartoonGenerator(api_key)
    
    # Example 1: Basic cartoon generation
    print("Generating basic cartoon...")
    try:
        cartoon_data = generator.generate_cartoon(
            prompt="A day in the life of a developer",
            page_count=8,
            strips_per_page=4
        )
        
        # Save with default settings
        saved_files = generator.save_pages(cartoon_data, "./output/basic")
        print(f"Basic cartoon saved: {len(saved_files)} files")
        
    except Exception as e:
        print(f"Basic generation failed: {e}")
    
    # Example 2: Advanced cartoon with custom options
    print("\nGenerating advanced cartoon with custom options...")
    try:
        advanced_cartoon = generator.generate_cartoon(
            prompt="Space adventure with robots and aliens",
            page_count=6,
            strips_per_page=3,
            resolution=(1920, 1080),
            style='vibrant',
            color_palette=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
            art_style='anime',
            language='en',
            custom_options={
                'mood': 'adventurous',
                'target_audience': 'teen',
                'include_dialogue': True
            }
        )
        
        # Save with advanced options
        saved_files = generator.save_pages(
            advanced_cartoon, 
            "./output/advanced",
            format='PNG',
            apply_filters=True,
            enhance_quality=True,
            create_thumbnails=True,
            thumbnail_size=(300, 200)
        )
        print(f"Advanced cartoon saved: {len(saved_files)} files")
        
        # Create animated GIF
        gif_path = generator.create_gif_animation(
            advanced_cartoon,
            "./output/advanced/animation.gif",
            duration=1000,
            loop=0
        )
        print(f"Animation created: {gif_path}")
        
    except Exception as e:
        print(f"Advanced generation failed: {e}")
    
    # Example 3: Monochrome vintage style
    print("\nGenerating vintage monochrome cartoon...")
    try:
        vintage_cartoon = generator.generate_cartoon(
            prompt="Detective story in 1940s noir style",
            page_count=4,
            strips_per_page=2,
            style='monochrome',
            art_style='realistic',
            custom_options={
                'era': '1940s',
                'atmosphere': 'noir'
            }
        )
        
        generator.save_pages(
            vintage_cartoon,
            "./output/vintage",
            format='JPEG',
            apply_filters=True
        )
        print("Vintage cartoon saved successfully!")
        
    except Exception as e:
        print(f"Vintage generation failed: {e}")
