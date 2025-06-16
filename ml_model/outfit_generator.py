# ml_model/outfit_generator.py
'''import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .utils import get_color_compatibility_score, get_pattern_compatibility_score
import itertools
import random

# Define core item types for basic outfit structures
# This can be expanded and made more flexible
OUTFIT_STRUCTURES = {
    'casual': {
        'required': ['top', 'bottom', 'shoes'],
        'optional': ['outerwear', 'accessory']
    },
    'work': {
        'required': ['top', 'bottom', 'shoes'],
        'optional': ['outerwear', 'accessory'] # e.g., a blazer or a subtle necklace
    },
    'formal': {
        'required': ['top', 'bottom', 'shoes'], # Or 'dress' and 'shoes'
        'optional': ['outerwear', 'accessory'] # e.g., a suit jacket, tie, elegant jewelry
    },
    'party': {
        'required': ['top', 'bottom', 'shoes'], # Or 'dress' and 'shoes'
        'optional': ['accessory', 'outerwear'] # e.g., statement jewelry, a stylish jacket
    },
    # Add more specific occasions if needed
}

# Alternative structure if a 'dress' is present
DRESS_OUTFIT_STRUCTURES = {
    'casual': {
        'required': ['dress', 'shoes'],
        'optional': ['outerwear', 'accessory']
    },
     'work': {
        'required': ['dress', 'shoes'],
        'optional': ['outerwear', 'accessory']
    },
    'formal': {
        'required': ['dress', 'shoes'],
        'optional': ['outerwear', 'accessory']
    },
    'party': {
        'required': ['dress', 'shoes'],
        'optional': ['accessory', 'outerwear']
    }
}


def calculate_outfit_score(outfit_items_list):
    """Calculates a score for a given list of items in an outfit."""
    if len(outfit_items_list) < 2:
        return 0 # Not enough items to compare

    total_score = 0
    num_pairs = 0

    # Score pairs within the outfit
    for item1, item2 in itertools.combinations(outfit_items_list, 2):
        # 1. Visual Similarity (using embeddings)
        # Ensure embeddings are 2D for cosine_similarity
        sim_score = 0
        if 'embedding' in item1 and 'embedding' in item2 and \
           item1['embedding'] is not None and item2['embedding'] is not None:
            try:
                sim_score = cosine_similarity(item1['embedding'].reshape(1, -1), item2['embedding'].reshape(1, -1))[0][0]
                 # Cosine similarity ranges from -1 to 1. Normalize to 0-1 for easier combination.
                sim_score_normalized = (sim_score + 1) / 2
            except ValueError as ve: # Catch issues if embeddings are not as expected
                print(f"ValueError during cosine similarity: {ve}. Items: {item1.get('name')}, {item2.get('name')}")
                sim_score_normalized = 0.3 # Default low score
        else:
            sim_score_normalized = 0.3 # Default if embeddings are missing

        # 2. Color Compatibility
        color_score = 0.3 # Default
        if item1.get('dominant_colors_rgb') and item2.get('dominant_colors_rgb'):
            color_score = get_color_compatibility_score(item1['dominant_colors_rgb'], item2['dominant_colors_rgb'])

        # 3. Pattern Compatibility
        pattern_score = 0.5 # Default
        if item1.get('pattern') and item2.get('pattern'):
             pattern_score = get_pattern_compatibility_score(item1['pattern'], item2['pattern'])

        # Weighted average for pair score
        # Weights can be tuned. Visual similarity and color often carry more weight.
        # These weights are critical and would ideally be learned or fine-tuned extensively.
        pair_score = (0.5 * sim_score_normalized) + (0.3 * color_score) + (0.2 * pattern_score)
        total_score += pair_score
        num_pairs += 1

    if num_pairs == 0:
        return 0
    
    average_outfit_score = total_score / num_pairs
    return round(average_outfit_score, 3)


def generate_combinations(wardrobe_items_raw, occasion, num_combinations=10, max_to_evaluate_raw=500):
    """
    Generates outfit combinations based on wardrobe items and occasion.
    wardrobe_items_raw: list of item dictionaries (from database to_dict())
    """
    occasion = occasion.lower()
    # Filter items by selected occasion and deserialize features
    occasion_items = []
    for item_data in wardrobe_items_raw:
        if occasion in [o.strip().lower() for o in item_data['occasion_tags'].split(',')]:
            # Deserialize embedding and dominant_colors
            # Ensure embedding_blob is not None before trying to process it
            if item_data['embedding_blob']:
                try:
                    item_data['embedding'] = np.frombuffer(item_data['embedding_blob'], dtype=np.float32)
                except Exception as e:
                    print(f"Error deserializing embedding for item {item_data.get('id', 'N/A')}: {e}")
                    item_data['embedding'] = None # Or skip item
            else:
                item_data['embedding'] = None

            if item_data['dominant_colors_rgb_str']:
                try:
                    item_data['dominant_colors_rgb'] = [
                        tuple(map(int, c.split(','))) for c in item_data['dominant_colors_rgb_str'].split(';') if c
                    ]
                except Exception as e:
                    print(f"Error deserializing colors for item {item_data.get('id', 'N/A')}: {e}")
                    item_data['dominant_colors_rgb'] = []
            else:
                item_data['dominant_colors_rgb'] = []
            
            if item_data['embedding'] is not None: # Only consider items with embeddings
                occasion_items.append(item_data)


    if not occasion_items:
        print(f"No items found for occasion: {occasion} with valid features.")
        return []

    items_by_type = {}
    for item in occasion_items:
        item_type = item['item_type'].lower()
        if item_type not in items_by_type:
            items_by_type[item_type] = []
        items_by_type[item_type].append(item)

    # Determine the outfit structure for the occasion
    # Check if 'dress' items exist for the occasion first
    has_dresses = 'dress' in items_by_type and items_by_type['dress']
    
    base_structure_rules = None
    if has_dresses and occasion in DRESS_OUTFIT_STRUCTURES:
        base_structure_rules = DRESS_OUTFIT_STRUCTURES.get(occasion)
    
    if not base_structure_rules: # Fallback to default or non-dress structure
        base_structure_rules = OUTFIT_STRUCTURES.get(occasion)

    if not base_structure_rules:
        print(f"No outfit structure defined for occasion: {occasion}")
        return []

    required_types = base_structure_rules['required']
    optional_types = base_structure_rules.get('optional', [])

    # Check if we have items for each *required* type
    for req_type in required_types:
        if req_type not in items_by_type or not items_by_type[req_type]:
            print(f"Warning: Missing required item type '{req_type}' for occasion '{occasion}'. Cannot generate complete outfits.")
            return [] # Cannot form base required outfit

    all_scored_outfits = []

    # --- Generate base required outfits ---
    required_item_lists = [items_by_type[rt] for rt in required_types]
    
    raw_combinations_iter = itertools.product(*required_item_lists)
    evaluated_count = 0

    for base_outfit_tuple in raw_combinations_iter:
        if evaluated_count >= max_to_evaluate_raw:
            print(f"Reached max_to_evaluate_raw ({max_to_evaluate_raw}) for base outfits.")
            break
        evaluated_count += 1

        # Ensure no duplicate item IDs in the base outfit (e.g., if a type allows multiple like 'accessory')
        # This check is more for when a type can have multiple instances in 'required'
        if len(set(item['id'] for item in base_outfit_tuple)) != len(base_outfit_tuple):
            continue

        current_outfit_items = list(base_outfit_tuple)
        
        # --- Attempt to add optional items ---
        # This part can become complex quickly. For simplicity:
        # Try adding one optional item at a time from each available optional category.
        # A more advanced system might try combinations of optional items.
        
        potential_outfits_with_optionals = [current_outfit_items[:]] # Start with the base outfit

        for opt_type in optional_types:
            if opt_type in items_by_type and items_by_type[opt_type]:
                new_potential_outfits = []
                for existing_outfit in potential_outfits_with_optionals:
                    for opt_item in items_by_type[opt_type]:
                        # Avoid adding the same item if it's already (somehow) in existing_outfit
                        if opt_item['id'] not in [eo['id'] for eo in existing_outfit]:
                            new_potential_outfits.append(existing_outfit + [opt_item])
                if new_potential_outfits: # Only update if we actually added options
                     # Limit explosion of combinations with optionals
                    potential_outfits_with_optionals.extend(new_potential_outfits)
                    if len(potential_outfits_with_optionals) > max_to_evaluate_raw * 2: # Heuristic limit
                        potential_outfits_with_optionals = random.sample(potential_outfits_with_optionals, max_to_evaluate_raw *2)


        for outfit_candidate in potential_outfits_with_optionals:
            # Deduplicate items in the final candidate list (e.g. if an optional was already required)
            # This shouldn't happen with strict required/optional types but good for safety
            unique_item_ids = set()
            final_outfit_items = []
            for item in outfit_candidate:
                if item['id'] not in unique_item_ids:
                    final_outfit_items.append(item)
                    unique_item_ids.add(item['id'])
            
            if len(final_outfit_items) < len(required_types): # Should not happen if initial check passed
                continue

            score = calculate_outfit_score(final_outfit_items)
            all_scored_outfits.append({'items': final_outfit_items, 'score': score})


    # Sort outfits by score in descending order
    sorted_outfits = sorted(all_scored_outfits, key=lambda x: x['score'], reverse=True)

    # Deduplicate outfits that might have the same items but different scores due to optional path
    # Or outfits that are identical.
    # A simple way: keep track of sets of item IDs.
    final_unique_outfits = []
    seen_outfit_item_ids_sets = set()

    for outfit in sorted_outfits:
        item_ids_tuple = tuple(sorted([item['id'] for item in outfit['items']]))
        if item_ids_tuple not in seen_outfit_item_ids_sets:
            final_unique_outfits.append(outfit)
            seen_outfit_item_ids_sets.add(item_ids_tuple)
            if len(final_unique_outfits) >= num_combinations:
                break
                
    return final_unique_outfits[:num_combinations]'''

# ml_model/outfit_generator.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .utils import get_color_compatibility_score, get_pattern_compatibility_score
import itertools
import random

# Define core item types for basic outfit structures
# This can be expanded and made more flexible
OUTFIT_STRUCTURES = {
    'casual': {
        'required': ['top', 'bottom', 'shoes'],
        'optional': ['outerwear', 'accessory']
    },
    'work': {
        'required': ['top', 'bottom', 'shoes'],
        'optional': ['outerwear', 'accessory'] # e.g., a blazer or a subtle necklace
    },
    'formal': {
        'required': ['top', 'bottom', 'shoes'], # Or 'dress' and 'shoes'
        'optional': ['outerwear', 'accessory'] # e.g., a suit jacket, tie, elegant jewelry
    },
    'party': {
        'required': ['top', 'bottom', 'shoes'], # Or 'dress' and 'shoes'
        'optional': ['accessory', 'outerwear'] # e.g., statement jewelry, a stylish jacket
    },
    # Add more specific occasions if needed
}

# Alternative structure if a 'dress' is present
DRESS_OUTFIT_STRUCTURES = {
    'casual': {
        'required': ['dress', 'shoes'],
        'optional': ['outerwear', 'accessory']
    },
     'work': {
        'required': ['dress', 'shoes'],
        'optional': ['outerwear', 'accessory']
    },
    'formal': {
        'required': ['dress', 'shoes'],
        'optional': ['outerwear', 'accessory']
    },
    'party': {
        'required': ['dress', 'shoes'],
        'optional': ['accessory', 'outerwear']
    }
}


def calculate_outfit_score(outfit_items_list):
    """Calculates a score for a given list of items in an outfit."""
    if len(outfit_items_list) < 2:
        return 0 # Not enough items to compare

    total_score = 0
    num_pairs = 0

    # Score pairs within the outfit
    for item1, item2 in itertools.combinations(outfit_items_list, 2):
        # 1. Visual Similarity (using embeddings)
        # Ensure embeddings are 2D for cosine_similarity
        sim_score = 0
        if 'embedding' in item1 and 'embedding' in item2 and \
           item1['embedding'] is not None and item2['embedding'] is not None:
            try:
                # Ensure embeddings are numpy arrays and correctly shaped
                emb1 = np.asarray(item1['embedding']).reshape(1, -1)
                emb2 = np.asarray(item2['embedding']).reshape(1, -1)
                sim_score = cosine_similarity(emb1, emb2)[0][0]
                 # Cosine similarity ranges from -1 to 1. Normalize to 0-1 for easier combination.
                sim_score_normalized = (sim_score + 1) / 2
            except ValueError as ve: # Catch issues if embeddings are not as expected
                print(f"ValueError during cosine similarity: {ve}. Items: {item1.get('name')}, {item2.get('name')}")
                sim_score_normalized = 0.3 # Default low score
        else:
            sim_score_normalized = 0.3 # Default if embeddings are missing

        # 2. Color Compatibility
        color_score = 0.3 # Default
        if item1.get('dominant_colors_rgb') and item2.get('dominant_colors_rgb'):
            color_score = get_color_compatibility_score(item1['dominant_colors_rgb'], item2['dominant_colors_rgb'])

        # 3. Pattern Compatibility
        pattern_score = 0.5 # Default
        if item1.get('pattern') and item2.get('pattern'):
             pattern_score = get_pattern_compatibility_score(item1.get('pattern', 'solid'), item2.get('pattern', 'solid')) # Added defaults here too

        # Weighted average for pair score
        # Weights can be tuned. Visual similarity and color often carry more weight.
        # These weights are critical and would ideally be learned or fine-tuned extensively.
        pair_score = (0.5 * sim_score_normalized) + (0.3 * color_score) + (0.2 * pattern_score)
        total_score += pair_score
        num_pairs += 1

    if num_pairs == 0:
        return 0
    
    average_outfit_score = total_score / num_pairs
    return round(average_outfit_score, 3)


def generate_combinations(wardrobe_items_raw, occasion, num_combinations=10, max_to_evaluate_raw=500):
    """
    Generates outfit combinations based on wardrobe items and occasion.
    wardrobe_items_raw: list of item dictionaries (from database to_dict())
    """
    occasion = occasion.lower()
    # Filter items by selected occasion and deserialize features
    occasion_items = []
    print(f"\n[generate_combinations] Processing occasion: {occasion}")
    print(f"[generate_combinations] Received {len(wardrobe_items_raw)} raw items.")

    for item_data in wardrobe_items_raw:
        item_name_for_log = item_data.get('name', f"ID: {item_data.get('id', 'N/A')}")
        # Check occasion match first
        item_occasions = [o.strip().lower() for o in item_data.get('occasion_tags', '').split(',')]
        if occasion not in item_occasions:
            # print(f"[DEBUG] Skipping item '{item_name_for_log}' - does not match occasion '{occasion}' (Tags: {item_occasions})")
            continue

        # Now deserialize features for items matching the occasion
        temp_item_data = item_data.copy() # Work on a copy to avoid modifying original raw list if reused
        valid_features = True

        # Deserialize embedding
        if temp_item_data.get('embedding_blob'):
            try:
                temp_item_data['embedding'] = np.frombuffer(temp_item_data['embedding_blob'], dtype=np.float32)
                if temp_item_data['embedding'].size == 0 or temp_item_data['embedding'].shape[0] != 2048: # Check shape/size
                    print(f"[WARN] Invalid embedding size for item '{item_name_for_log}'. Size: {temp_item_data['embedding'].size}")
                    temp_item_data['embedding'] = None
                    valid_features = False
            except Exception as e:
                print(f"[ERROR] Deserializing embedding for item '{item_name_for_log}': {e}")
                temp_item_data['embedding'] = None
                valid_features = False
        else:
            # print(f"[DEBUG] Missing embedding_blob for item '{item_name_for_log}'")
            temp_item_data['embedding'] = None
            valid_features = False # Require embedding for ML combination generation

        # Deserialize dominant colors
        if temp_item_data.get('dominant_colors_rgb_str'):
            try:
                colors_list = []
                for c in temp_item_data['dominant_colors_rgb_str'].split(';'):
                    if c: # Ensure string is not empty
                        rgb_parts = list(map(int, c.split(',')))
                        if len(rgb_parts) == 3:
                            colors_list.append(tuple(rgb_parts))
                        else:
                            print(f"[WARN] Invalid color format '{c}' for item '{item_name_for_log}'")
                temp_item_data['dominant_colors_rgb'] = colors_list
                if not temp_item_data['dominant_colors_rgb']: # If parsing failed for all parts
                     print(f"[WARN] Could not parse any valid colors for item '{item_name_for_log}' from string: {temp_item_data['dominant_colors_rgb_str']}")
            except Exception as e:
                print(f"[ERROR] Deserializing colors for item '{item_name_for_log}': {e}")
                temp_item_data['dominant_colors_rgb'] = []
        else:
            # print(f"[DEBUG] Missing dominant_colors_rgb_str for item '{item_name_for_log}'")
            temp_item_data['dominant_colors_rgb'] = []

        # Ensure pattern exists, default to solid if missing or None
        if 'pattern' not in temp_item_data or temp_item_data['pattern'] is None:
            temp_item_data['pattern'] = 'solid'

        # Only add items that match the occasion AND have valid features (specifically embedding)
        if valid_features:
            occasion_items.append(temp_item_data)
        else:
            print(f"[INFO] Skipping item '{item_name_for_log}' for combination generation due to missing/invalid features.")


    print(f"[generate_combinations] Found {len(occasion_items)} items with valid features for occasion '{occasion}'.")
    if not occasion_items:
        print(f"[generate_combinations] No items with features found for occasion: {occasion}. Cannot generate combinations.")
        return []

    items_by_type = {}
    for item in occasion_items:
        item_type = item.get('item_type', 'unknown').lower()
        if item_type not in items_by_type:
            items_by_type[item_type] = []
        items_by_type[item_type].append(item)

    # Determine the outfit structure for the occasion
    has_dresses = 'dress' in items_by_type and items_by_type['dress']

    base_structure_rules = None
    structure_type_used = "Default" # For logging
    if has_dresses and occasion in DRESS_OUTFIT_STRUCTURES:
        base_structure_rules = DRESS_OUTFIT_STRUCTURES.get(occasion)
        structure_type_used = "Dress"
        print(f"[generate_combinations] Using 'Dress' structure rules for {occasion}.")

    if not base_structure_rules: # Fallback to default or non-dress structure
        base_structure_rules = OUTFIT_STRUCTURES.get(occasion)
        structure_type_used = "Standard"
        print(f"[generate_combinations] Using 'Standard' ({'found' if base_structure_rules else 'not found, default needed'}) structure rules for {occasion}.")


    if not base_structure_rules:
        print(f"[ERROR] No outfit structure rule found for occasion: {occasion}. Cannot proceed.")
        # Fallback to a very basic structure if none defined?
        # base_structure_rules = {'required': ['top', 'bottom', 'shoes'], 'optional': []}
        # print("[WARN] Using basic fallback structure: top, bottom, shoes.")
        return [] # Or return empty if structure is critical

    required_types = base_structure_rules.get('required', [])
    optional_types = base_structure_rules.get('optional', [])
    print(f"[generate_combinations] Structure ({structure_type_used}): Required={required_types}, Optional={optional_types}")

    # Check if we have items for each *required* type in our filtered list
    missing_required = []
    for req_type in required_types:
        if req_type not in items_by_type or not items_by_type[req_type]:
            missing_required.append(req_type)

    if missing_required:
        print(f"[WARN] Missing required item types for occasion '{occasion}': {missing_required}. Cannot generate complete '{structure_type_used}' outfits.")
        return [] # Cannot form base required outfit

    all_scored_outfits = []

    # --- Generate base required outfits ---
    try:
        required_item_lists = [items_by_type[rt] for rt in required_types]
    except KeyError as e:
         print(f"[ERROR] KeyError while building required_item_lists. Missing type: {e}. This shouldn't happen after check.")
         return []

    print(f"[generate_combinations] Generating combinations from required types: {required_types}")
    # Estimate number of raw combos
    num_raw_combos = 1
    for L in required_item_lists: num_raw_combos *= len(L)
    print(f"[generate_combinations] Estimated raw combinations from required items: {num_raw_combos}")

    if num_raw_combos == 0:
        print("[WARN] No combinations possible from required items (one list is empty).")
        return []

    raw_combinations_iter = itertools.product(*required_item_lists)
    evaluated_count = 0

    # Limit the initial evaluation if too many raw combinations
    effective_max_eval = min(max_to_evaluate_raw, num_raw_combos) if num_raw_combos > 0 else 0
    print(f"[generate_combinations] Will evaluate up to {effective_max_eval} base combinations.")

    for base_outfit_tuple in raw_combinations_iter:
        if evaluated_count >= effective_max_eval:
            print(f"[INFO] Reached max_to_evaluate_raw ({effective_max_eval}) for base outfits.")
            break
        evaluated_count += 1

        # Ensure no duplicate item IDs in the base outfit (mainly for safety)
        base_ids = set(item['id'] for item in base_outfit_tuple)
        if len(base_ids) != len(base_outfit_tuple):
            # print(f"[DEBUG] Skipping base combo with duplicate item IDs (should be rare with type separation): {[item['id'] for item in base_outfit_tuple]}")
            continue

        current_outfit_items = list(base_outfit_tuple)

        # --- Attempt to add optional items ---
        potential_outfits_with_optionals = [current_outfit_items[:]] # Start with the base outfit

        for opt_type in optional_types:
            if opt_type in items_by_type and items_by_type[opt_type]:
                # Temporarily store new outfits generated in this optional step
                outfits_generated_this_step = []
                # Iterate through outfits generated *so far* (including base and previous optionals)
                for existing_outfit in potential_outfits_with_optionals:
                    existing_ids = set(eo['id'] for eo in existing_outfit)
                    for opt_item in items_by_type[opt_type]:
                        # Avoid adding the same item ID again
                        if opt_item['id'] not in existing_ids:
                            # Create a new list for the new outfit
                            new_outfit_with_optional = existing_outfit + [opt_item]
                            outfits_generated_this_step.append(new_outfit_with_optional)

                # Add the newly generated outfits (with the current optional type) to the main list
                if outfits_generated_this_step:
                    potential_outfits_with_optionals.extend(outfits_generated_this_step)
                    # Optional: Add pruning/sampling if list grows too large
                    optional_limit = effective_max_eval * 3 # Heuristic limit for total outfits with optionals
                    if len(potential_outfits_with_optionals) > optional_limit:
                        print(f"[INFO] Pruning potential outfits with optionals from {len(potential_outfits_with_optionals)} to {optional_limit}")
                        potential_outfits_with_optionals = random.sample(potential_outfits_with_optionals, optional_limit)

        # Score all potential outfits (base + those with optionals)
        # print(f"[DEBUG] Scoring {len(potential_outfits_with_optionals)} potential outfits (incl. optionals) derived from base combo {evaluated_count}")
        for outfit_candidate in potential_outfits_with_optionals:
            # Final check for duplicates and required types before scoring
            unique_item_ids = set()
            final_outfit_items = []
            final_types = set()
            for item in outfit_candidate:
                if item['id'] not in unique_item_ids:
                    final_outfit_items.append(item)
                    unique_item_ids.add(item['id'])
                    final_types.add(item.get('item_type', 'unknown').lower())

            # Ensure all required types are still present (should be, but check)
            if not all(req in final_types for req in required_types):
                 # print(f"[DEBUG] Skipping candidate missing required type after optional add: {[item['name'] for item in final_outfit_items]}")
                 continue

            if len(final_outfit_items) < 2 : # Need at least 2 items to score
                # print(f"[DEBUG] Skipping candidate with < 2 items: {[item['name'] for item in final_outfit_items]}")
                continue

            score = calculate_outfit_score(final_outfit_items)
            # Store the clean list of items
            all_scored_outfits.append({'items': final_outfit_items, 'score': score})

    print(f"[generate_combinations] Total scored outfits generated (before sorting/deduplication): {len(all_scored_outfits)}")

    # Sort outfits by score in descending order
    # Handle potential None scores? Should not happen with current logic, but maybe add default
    sorted_outfits = sorted(all_scored_outfits, key=lambda x: x.get('score', 0), reverse=True)

    # Deduplicate outfits based on the set of item IDs they contain
    final_unique_outfits = []
    seen_outfit_item_ids_sets = set()

    for outfit in sorted_outfits:
        # Ensure 'items' key exists and is a list before proceeding
        if 'items' not in outfit or not isinstance(outfit['items'], list) or not outfit['items']:
            print(f"[WARN] Skipping malformed outfit during deduplication: {outfit}")
            continue

        # Create a frozenset of item IDs for checking uniqueness (tuples work too but sets ignore order)
        item_ids_set = frozenset(item['id'] for item in outfit['items'])
        if not item_ids_set: # Skip if somehow empty
             continue

        if item_ids_set not in seen_outfit_item_ids_sets:
            # Make sure the structure is correct before adding
            if 'score' in outfit:
                final_unique_outfits.append(outfit)
                seen_outfit_item_ids_sets.add(item_ids_set)
            else:
                print(f"[WARN] Skipping outfit missing 'score' during final selection: {outfit}")

            if len(final_unique_outfits) >= num_combinations:
                break

    print(f"[generate_combinations] Final unique outfits count: {len(final_unique_outfits)}")

    # <<< START OF ADDED DEBUGGING BLOCK >>>
    print("\n--- Final Outfits Before Return ---")
    if not final_unique_outfits:
        print("  No outfits generated.")
    for i, outfit in enumerate(final_unique_outfits[:num_combinations]):
        print(f"Outfit {i}:")
        # Check main structure
        if not isinstance(outfit, dict):
            print(f"  ERROR: Outfit is not a dictionary! Type: {type(outfit)}")
            continue

        # Check 'items' key
        if 'items' not in outfit:
            print(f"  ERROR: Missing 'items' key!")
            items_ok = False
        elif not isinstance(outfit['items'], list):
             print(f"  ERROR: 'items' is not a list! Type: {type(outfit.get('items'))}")
             items_ok = False
        else:
             print(f"  Items count: {len(outfit.get('items', []))}") # Safely get length
             items_ok = True
             # Optional: Check individual items within the list
             for idx, item in enumerate(outfit['items']):
                 if not isinstance(item, dict):
                     print(f"    ERROR: Item at index {idx} is not a dict! Type: {type(item)}")


        # Check 'score' key
        if 'score' not in outfit:
            print(f"  ERROR: Missing 'score' key!")
        else:
            print(f"  Score: {outfit.get('score', 'N/A')}")

        # Optional: Print item names for more detail if items structure is okay
        if items_ok:
            item_names = [item.get('name', 'Unknown') for item in outfit.get('items', [])]
            print(f"  Item Names: {item_names}")
        print("-" * 10) # Separator for clarity
    print("---------------------------------\n")
    # <<< END OF ADDED DEBUGGING BLOCK >>>

    return final_unique_outfits[:num_combinations]