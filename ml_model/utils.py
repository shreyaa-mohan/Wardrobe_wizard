# ml_model/utils.py
import numpy as np
from collections import Counter
from PIL import Image
from sklearn.cluster import KMeans

# --- Color Extraction ---
def extract_dominant_colors(image_path, num_colors=3):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((100, 100)) # Resize for faster processing
        img_array = np.array(img)
        img_array = img_array.reshape((-1, 3)) # Reshape to list of pixels

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto') # n_init='auto' or 10
        kmeans.fit(img_array)
        
        # Get colors and their frequencies
        counts = Counter(kmeans.labels_)
        dominant_colors_rgb = kmeans.cluster_centers_.astype(int)
        
        # Sort by frequency (most dominant first)
        sorted_indices = [i for i, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
        sorted_dominant_colors_rgb = dominant_colors_rgb[sorted_indices]
        
        return [tuple(color) for color in sorted_dominant_colors_rgb]
    except Exception as e:
        print(f"Error extracting colors from {image_path}: {e}")
        # Return a default color (e.g., black) if extraction fails
        return [(0,0,0)] * num_colors


# --- Basic Color Definitions (for simplicity, actual color theory is complex) ---
NEUTRAL_COLORS_RGB_RANGES = [
    # Black/Dark Gray (very dark colors)
    ((0, 60), (0, 60), (0, 60)),
    # White/Light Gray (very light colors)
    ((200, 255), (200, 255), (200, 255)),
    # Grays (where R, G, B are close to each other and not too dark/light)
    # ((80, 180), (80, 180), (80, 180)), # This is too broad, can lead to issues
    # More specific grays might check if R, G, B values are close to each other.
    # For simplicity, we'll focus on black/white/light gray as primary neutrals.
    # Browns/Beiges are harder to define simply with RGB ranges.
    # Example for a desaturated brownish/beige:
    # ((120, 200), (100, 180), (70, 150)) # Highly approximate
]

def is_neutral_rgb(rgb_color):
    """Checks if an RGB color is a neutral (black, white, gray). Simplified."""
    r, g, b = rgb_color
    # Check for black/dark gray or white/light gray
    for r_range, g_range, b_range in NEUTRAL_COLORS_RGB_RANGES:
        if r_range[0] <= r <= r_range[1] and \
           g_range[0] <= g <= g_range[1] and \
           b_range[0] <= b <= b_range[1]:
            return True
    
    # Check for grays (R, G, B values are close)
    # And not too saturated
    mean_val = (r + g + b) / 3
    if abs(r - mean_val) < 30 and abs(g - mean_val) < 30 and abs(b - mean_val) < 30:
        # Check saturation (max_rgb - min_rgb)
        if (max(r,g,b) - min(r,g,b)) < 50: # Low saturation threshold
             return True
    return False

def get_color_compatibility_score(colors1_rgb, colors2_rgb):
    """
    Simplified color compatibility score.
    colors1_rgb, colors2_rgb: lists of dominant RGB tuples for two items.
    Returns a score between 0 and 1.
    """
    if not colors1_rgb or not colors2_rgb:
        return 0.3 # Low score if color info is missing

    primary_color1 = colors1_rgb[0]
    primary_color2 = colors2_rgb[0]
    
    score = 0.5 # Base score

    # 1. Neutrals go well with most things
    if is_neutral_rgb(primary_color1) or is_neutral_rgb(primary_color2):
        score += 0.4
        if is_neutral_rgb(primary_color1) and is_neutral_rgb(primary_color2): # Two neutrals
            score += 0.1 # Slightly higher for two neutrals

    # 2. Similarity of primary colors (e.g., shades of blue)
    # Using Euclidean distance in RGB space (simple but not perceptually uniform)
    color_diff = np.linalg.norm(np.array(primary_color1) - np.array(primary_color2))
    if color_diff < 80: # Arbitrary threshold for "similar enough" (lower is more similar)
        score += 0.2
    elif color_diff > 200 and not (is_neutral_rgb(primary_color1) or is_neutral_rgb(primary_color2)): # Very different, non-neutral colors
        score -= 0.1 # Slight penalty for potentially clashing bright colors

    # This is still very basic. Advanced color harmony would use HSL/HSV,
    # consider complementary, analogous, triadic relationships, etc.
    # For example:
    # - A bright color + a neutral is good.
    # - Two very different bright colors might clash unless one is an accent.
    
    return np.clip(score, 0.0, 1.0)


# --- Pattern Compatibility (Heuristic-based) ---
def get_pattern_compatibility_score(pattern1, pattern2):
    pattern1 = pattern1.lower() if pattern1 else "solid"
    pattern2 = pattern2.lower() if pattern2 else "solid"

    if pattern1 == "solid" or pattern2 == "solid":
        if pattern1 == "solid" and pattern2 == "solid":
            return 1.0 # Two solids are perfectly compatible
        return 0.9  # One solid generally goes well with any pattern

    # Defining "busy" patterns (this is subjective)
    busy_patterns = ["floral", "animal_print", "geometric_bold", "abstract"]
    subtle_patterns = ["stripes_thin", "checks_small", "polka_dot_small", "heather"]

    if pattern1 == pattern2:
        if pattern1 in busy_patterns:
            return 0.3 # Two identical busy patterns can be overwhelming
        elif pattern1 in subtle_patterns:
            return 0.7 # Two identical subtle patterns can work
        else: # e.g. two "stripes" (could be bold or thin)
            return 0.5 # Default for same non-solid patterns

    # Different patterns
    p1_is_busy = pattern1 in busy_patterns
    p2_is_busy = pattern2 in busy_patterns
    p1_is_subtle = pattern1 in subtle_patterns
    p2_is_subtle = pattern2 in subtle_patterns

    if p1_is_busy and p2_is_busy:
        return 0.2 # Two different busy patterns are usually a clash
    
    if (p1_is_busy and p2_is_subtle) or (p1_is_subtle and p2_is_busy):
        return 0.8 # A busy pattern with a subtle one is often good
        
    if p1_is_subtle and p2_is_subtle:
        return 0.7 # Two different subtle patterns can work

    # Default for unclassified different patterns (e.g., stripes + checks)
    # This requires more nuanced rules or learning.
    return 0.6