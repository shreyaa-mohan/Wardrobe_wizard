


import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import numpy as np # Ensure NumPy is imported
import logging # Import logging
from datetime import datetime # MAKE SURE THIS IMPORT IS PRESENT
from datetime import datetime

# Import ML functions AFTER defining app and db to avoid circular imports if they needed app context
# Assuming ml_model is correctly structured as a package
# Make sure these imports point to your actual ML files
try:
    from ml_model.feature_extractor import extract_features
    from ml_model.outfit_generator import generate_combinations
except ImportError as e:
    logging.error(f"Failed to import ML modules: {e}. ML features will be disabled.")
    # Define dummy functions if imports fail, to prevent crashes later
    def extract_features(filepath):
        logging.warning("Using dummy extract_features due to import error.")
        return [], None # Return empty colors and None embedding
    def generate_combinations(wardrobe_data, selected_occasion, num_combinations=10):
        logging.warning("Using dummy generate_combinations due to import error.")
        return [] # Return empty list

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_CHANGE_ME' # Change this in production!
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Use SQLite for simplicity. Create 'instance' folder if it doesn't exist.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'wardrobe.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Keep console logging active when running flask run or python app.py
# Optionally add file logging if needed:
# handler = logging.FileHandler('app.log') # Log to a file named app.log
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# app.logger.addHandler(handler) # Add the file handler to Flask's logger


db = SQLAlchemy(app)
@app.context_processor
def inject_now():
    return {'SCRIPT_BEGIN_TIME': datetime.utcnow()} # or datetime.now()

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Create instance folder for db if it doesn't exist (Flask usually does this, but good practice)
os.makedirs(os.path.join(BASE_DIR, 'instance'), exist_ok=True)


# --- Database Model ---
class WardrobeItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    item_type = db.Column(db.String(50), nullable=False) # e.g., top, bottom, shoes, accessory, outerwear, dress
    pattern = db.Column(db.String(50), nullable=True, default='solid') # e.g., solid, striped, floral, geometric, animal_print
    occasion_tags = db.Column(db.String(200), nullable=False) # Comma-separated: e.g., "casual, work, party"
    image_filename = db.Column(db.String(100), nullable=False, unique=True)

    # ML Features - Stored efficiently
    dominant_colors_rgb_str = db.Column(db.String(100), nullable=True) # Store as "R,G,B;R,G,B;..."
    embedding_blob = db.Column(db.LargeBinary, nullable=True) # Store numpy array as blob

    def __repr__(self):
        return f'<WardrobeItem {self.name}>'

    # Helper to get image URL
    @property
    def image_url(self):
        # Use try-except block in case url_for fails (e.g., outside request context)
        try:
            # Use _external=False for relative paths suitable for templates
            return url_for('uploaded_file', filename=self.image_filename, _external=False)
        except RuntimeError:
            # Fallback or return None if url generation fails (e.g., during background tasks)
            app.logger.warning(f"url_for failed for {self.image_filename}, using basic fallback.")
            return f"/uploads/{self.image_filename}" # Basic fallback path

    # Helper to represent item for the ML model input
    # Ensure ALL keys needed by the template/ML model are here and spelled correctly
    def to_dict(self):
         # Get image_url safely using the property
        img_url = self.image_url

        return {
             'id': self.id,
             'name': self.name,
             'item_type': self.item_type,
             'pattern': self.pattern,
             'occasion_tags': self.occasion_tags,
             'image_filename': self.image_filename, # Crucial for template image display
             'dominant_colors_rgb_str': self.dominant_colors_rgb_str,
             'embedding_blob': self.embedding_blob, # Pass blob to ML func
             'image_url': img_url # Include for convenience in templates
             # Note: 'embedding' (actual numpy array) and 'dominant_colors_rgb' (parsed list)
             # are typically reconstructed/added IN the generator function after deserialization
         }


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Routes ---
@app.route('/')
def index():
    """Homepage: Display wardrobe items and occasion selection form."""
    app.logger.info("Accessing homepage route '/'")
    try:
        items = WardrobeItem.query.order_by(WardrobeItem.id.desc()).all()
        # Get unique occasions for the dropdown
        all_tags = set()
        for item in items:
            # Split tags, strip whitespace, convert to lowercase, filter out empty strings
            tags = [tag.strip().lower() for tag in item.occasion_tags.split(',') if tag.strip()]
            all_tags.update(tags)
        # Sort the unique tags for consistent dropdown order
        occasions = sorted(list(all_tags))
        app.logger.info(f"Found {len(items)} items and {len(occasions)} unique occasions.")

    except Exception as e:
        app.logger.error(f"Error loading wardrobe items or occasions: {e}", exc_info=True)
        flash(f"Error loading wardrobe data. Please check the logs.", 'danger')
        items = []
        occasions = []

    return render_template('index.html', items=items, occasions=occasions)


@app.route('/add', methods=['GET', 'POST'])
def add_item():
    """Add a new wardrobe item."""
    if request.method == 'POST':
        app.logger.info("Received POST request to /add")
        # --- Form Data ---
        name = request.form.get('name')
        item_type = request.form.get('item_type')
        pattern = request.form.get('pattern', 'solid') # Default to solid if not provided
        occasion_tags = request.form.get('occasion_tags')
        image_file = request.files.get('image_file')

        app.logger.info(f"Form data: name='{name}', type='{item_type}', pattern='{pattern}', tags='{occasion_tags}', file provided={bool(image_file)}")

        # --- Validation ---
        errors = []
        if not name: errors.append("Item name is required.")
        if not item_type: errors.append("Item type is required.")
        if not occasion_tags: errors.append("Occasion tags are required.")
        if not image_file: errors.append("Image file is required.")
        elif not allowed_file(image_file.filename):
             errors.append(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

        if errors:
            for error in errors:
                flash(error, 'warning')
            app.logger.warning(f"Add item failed validation: {errors}")
            # Return to form, preserving choices if possible (requires passing choices back)
            item_type_choices = ['Top', 'Bottom', 'Dress', 'Outerwear', 'Shoes', 'Accessory']
            pattern_choices = ['Solid', 'Stripes', 'Checks', 'Floral', 'Geometric', 'Animal Print', 'Polka Dot', 'Other']
            occasion_choices = ['Casual', 'Work', 'Formal', 'Party', 'Sport', 'Loungewear']
            return render_template('add_item.html', item_type_choices=item_type_choices, pattern_choices=pattern_choices, occasion_choices=occasion_choices, form_data=request.form), 400 # Bad Request

        # --- File Handling ---
        try:
            # Create a more unique and safe filename
            base, ext = os.path.splitext(image_file.filename)
            safe_base = secure_filename(base)
            # Ensure filename isn't excessively long if name/type/original filename are long
            filename_core = f"{name[:20].replace(' ', '_')}_{item_type[:10]}_{safe_base[:20]}"
            filename = secure_filename(f"{filename_core}_{os.urandom(4).hex()}{ext}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            app.logger.info(f"Saving uploaded file to: {filepath}")
            image_file.save(filepath)
        except Exception as e:
            app.logger.error(f'Error saving image: {e}', exc_info=True)
            flash(f'Error saving image. Please check the logs.', 'danger')
            return redirect(request.url) # Redirect back to add form

        # --- Feature Extraction (ML) ---
        dominant_colors_str = None
        embedding_bytes = None
        try:
            app.logger.info(f"Extracting features for: {filepath}")
            # Make sure extract_features is correctly imported or handled if missing
            dominant_colors_rgb, embedding = extract_features(filepath)

            if dominant_colors_rgb:
                 # Convert RGB tuples/lists to strings: "R,G,B;R,G,B;..."
                try:
                    dominant_colors_str = ";".join([",".join(map(str, color)) for color in dominant_colors_rgb])
                    app.logger.info(f"Extracted dominant colors: {dominant_colors_str}")
                except Exception as color_err:
                    app.logger.error(f"Error converting dominant colors to string: {color_err}", exc_info=True)
                    dominant_colors_str = None # Reset on error
            else:
                 app.logger.warning(f"No dominant colors extracted for {filename}")


            if embedding is not None and isinstance(embedding, np.ndarray) and embedding.size > 0:
                # Convert numpy embedding to bytes (blob) using float32 for consistency
                try:
                    embedding_bytes = embedding.astype(np.float32).tobytes()
                    app.logger.info(f"Generated embedding of size: {len(embedding_bytes)} bytes")
                except Exception as emb_err:
                    app.logger.error(f"Error converting embedding to bytes: {emb_err}", exc_info=True)
                    embedding_bytes = None # Reset on error
            else:
                 app.logger.warning(f"No valid embedding generated for {filename}. Type received: {type(embedding)}")


        except Exception as e:
            # Catch errors during the call to extract_features itself
            app.logger.error(f'Error during feature extraction call: {e}. Item will be added without ML data.', exc_info=True)
            flash(f'Feature extraction failed. Item added without ML data. Check logs.', 'warning') # Use warning, not danger
            dominant_colors_str = None # Ensure these are None if extraction failed
            embedding_bytes = None
            # Allow adding without features

        # --- Database Insertion ---
        try:
            new_item = WardrobeItem(
                name=name,
                item_type=item_type.lower(), # Store lowercase for consistency
                pattern=pattern.lower().replace(' ', '_'), # Store lowercase and replace spaces
                # Ensure tags are stored reasonably, maybe strip extra whitespace
                occasion_tags=",".join([tag.strip() for tag in occasion_tags.split(',') if tag.strip()]),
                image_filename=filename,
                dominant_colors_rgb_str=dominant_colors_str,
                embedding_blob=embedding_bytes
            )
            db.session.add(new_item)
            db.session.commit()
            app.logger.info(f'Item "{name}" (ID: {new_item.id}) added successfully to DB.')
            flash(f'Item "{name}" added successfully!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            db.session.rollback() # Rollback in case of DB error
            app.logger.error(f'Database error adding item: {e}', exc_info=True)
            flash(f'Database error adding item. Please check the logs.', 'danger')
             # Clean up the uploaded file if DB insertion fails
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    app.logger.info(f"Deleted orphaned file {filepath} after DB error.")
                except OSError as rm_err:
                    app.logger.error(f"Error deleting orphaned file {filepath} after DB error: {rm_err}")
            return redirect(request.url) # Redirect back to add form

    # --- GET Request ---
    # Provide choices for dropdowns
    item_type_choices = ['Top', 'Bottom', 'Dress', 'Outerwear', 'Shoes', 'Accessory']
    pattern_choices = ['Solid', 'Stripes', 'Checks', 'Floral', 'Geometric', 'Animal Print', 'Polka Dot', 'Other']
    occasion_choices = ['Casual', 'Work', 'Formal', 'Party', 'Sport', 'Loungewear'] # Suggest common tags
    return render_template('add_item.html',
                           item_type_choices=item_type_choices,
                           pattern_choices=pattern_choices,
                           occasion_choices=occasion_choices)


@app.route('/combinations', methods=['POST'])
def view_combinations():
    """Generate and display outfit combinations based on selected occasion."""
    selected_occasion = request.form.get('occasion')
    app.logger.info(f"Received POST request to /combinations for occasion: '{selected_occasion}'")

    if not selected_occasion:
        flash('Please select an occasion.', 'warning')
        app.logger.warning("Combination request failed: No occasion selected.")
        return redirect(url_for('index'))

    try:
        app.logger.info("Fetching all wardrobe items from DB...")
        all_items_raw = WardrobeItem.query.all()
        app.logger.info(f"Fetched {len(all_items_raw)} total items.")

        # Convert DB objects to dictionaries for the ML function
        # The ML function should handle filtering items that don't have necessary features (like embedding_blob)
        wardrobe_data = [item.to_dict() for item in all_items_raw]

        if not wardrobe_data:
             flash('Wardrobe is empty. Add items first.', 'warning')
             app.logger.warning("Combination generation skipped: Wardrobe is empty.")
             return redirect(url_for('index'))

        # --- Generate Combinations (ML) ---
        app.logger.info(f"Calling generate_combinations for occasion: {selected_occasion}")
        # Assume generate_combinations might return complex objects (like numpy types) initially
        # Make sure generate_combinations is correctly imported or handled if missing
        combinations_result = generate_combinations(wardrobe_data, selected_occasion, num_combinations=10)

        # --- Check if combinations were generated ---
        if not combinations_result:
             app.logger.info(f"ML model returned no combinations for '{selected_occasion}'.")
             flash(f'Could not generate combinations for "{selected_occasion}". Do you have enough compatible items (e.g., top, bottom, shoes) tagged for this occasion with features extracted?', 'info')
             return redirect(url_for('index'))

        # --- Clean up non-serializable data before sending to template ---
        app.logger.info("Cleaning combinations data for template rendering...")
        cleaned_combinations = []
        if isinstance(combinations_result, list):
            for i, combo in enumerate(combinations_result):
                # Check if combo is a dictionary
                if not isinstance(combo, dict):
                    app.logger.warning(f"Skipping non-dict item at index {i} in combinations_result: {type(combo)}")
                    continue

                # Clean the score (convert numpy floats/generics to python floats)
                score = combo.get('score')
                cleaned_score = None
                if isinstance(score, np.generic): # Checks for any numpy number type (int, float, etc.)
                    cleaned_score = score.item() # Convert to standard Python type
                elif isinstance(score, (int, float)):
                    cleaned_score = score # Already a standard Python type
                else:
                     # Log warning and provide a default if score is missing or wrong type
                     app.logger.warning(f"Unexpected or missing score type in combo {i}: {type(score)}. Setting to 0.0.")
                     cleaned_score = 0.0

                cleaned_combo = {'score': cleaned_score}
                cleaned_items = []
                items_list = combo.get('items')

                if isinstance(items_list, list):
                    for j, item in enumerate(items_list):
                        if not isinstance(item, dict):
                            app.logger.warning(f"Skipping non-dict item at index {j} within combo {i}: {type(item)}")
                            continue

                        # Create a clean copy of the item dict, excluding problematic keys/types
                        # Ensure all needed keys ARE included and problematic ones ARE excluded
                        cleaned_item = {}
                        for k, v in item.items():
                            # Exclude specific keys known to cause issues and NumPy arrays
                            if k not in ['embedding', 'embedding_blob'] and not isinstance(v, np.ndarray):
                                cleaned_item[k] = v
                            # Optionally log if other unexpected types are found, though usually exclusion is enough

                        # Ensure required keys for the template are present in the cleaned item
                        required_keys = ['id', 'name', 'item_type', 'image_filename', 'image_url'] # Added image_url
                        if all(key in cleaned_item for key in required_keys):
                            cleaned_items.append(cleaned_item)
                        else:
                             missing_keys = [key for key in required_keys if key not in cleaned_item]
                             app.logger.warning(f"Cleaned item (ID: {cleaned_item.get('id', 'N/A')}) in combo {i} is missing required keys for template: {missing_keys}. Skipping this item.")
                else:
                    app.logger.warning(f"Combo {i} (Score: {cleaned_score}) has non-list or missing 'items': {type(items_list)}")

                # Only add the combo to the final list if it has items after cleaning
                if cleaned_items:
                    cleaned_combo['items'] = cleaned_items
                    cleaned_combinations.append(cleaned_combo)
                else:
                    app.logger.warning(f"Combo {i} (Score: {cleaned_score}) resulted in an empty 'items' list after cleaning. Skipping this combo.")
        else:
            # Handle case where combinations_result is not a list
            app.logger.error(f"Combinations result from ML was not a list! Type: {type(combinations_result)}")
            cleaned_combinations = [] # Send empty list to template

        # --- Log cleaned data sample ---
        app.logger.info("\n--- Data AFTER Cleaning (Sample): ---")
        if cleaned_combinations:
            first_cleaned_combo = cleaned_combinations[0]
            log_detail = {
                     'score': first_cleaned_combo.get('score'),
                     'type_score': type(first_cleaned_combo.get('score')).__name__,
                     'items_count': len(first_cleaned_combo.get('items', [])),
                     'first_item_keys': list(first_cleaned_combo.get('items', [{}])[0].keys()) if first_cleaned_combo.get('items') else []
                 }
            app.logger.info(f"First cleaned combo structure sample: {log_detail}")
        else:
             app.logger.info("Cleaned combinations list is empty.")
        app.logger.info("------------------------------------\n")

        # Check again if cleaning resulted in no combinations left
        if not cleaned_combinations:
             app.logger.info(f"No valid combinations remained after cleaning for '{selected_occasion}'.")
             # Provide a more user-friendly message
             flash(f'Could not generate displayable combinations for "{selected_occasion}". This might be due to data issues with the generated items. Please check logs.', 'warning')
             return redirect(url_for('index'))

        # --- Render Template ---
        app.logger.info(f"Rendering view_combinations.html with {len(cleaned_combinations)} cleaned combinations.")
        return render_template('view_combinations.html',
                               combinations=cleaned_combinations, # Pass the CLEANED data
                               occasion=selected_occasion)

    except Exception as e:
        # Catch any unexpected errors during the process
        app.logger.error(f"Error in view_combinations route for occasion '{selected_occasion}': {e}", exc_info=True)
        flash(f"An unexpected error occurred while generating combinations. Please check the application logs.", 'danger')
        return redirect(url_for('index'))


@app.route('/uploads/<path:filename>') # Use path converter for flexibility if subdirs were ever used
def uploaded_file(filename):
    """Serve uploaded files safely."""
    # Basic security check: Use secure_filename to prevent directory traversal vulnerability
    # Although send_from_directory has some built-in protection, this adds an explicit layer.
    # Note: This might alter filenames with special characters, ensure consistency with DB.
    # If secure_filename transformation causes issues matching DB, consider storing BOTH
    # original and secured filename, or adjusting generation logic. For now, let's trust send_from_directory.

    # app.logger.debug(f"Serving file: {filename} from {app.config['UPLOAD_FOLDER']}") # Debug logging if needed
    try:
        # send_from_directory is generally safe against path traversal IF the directory is trusted
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)
    except FileNotFoundError:
        app.logger.error(f"File not found in uploads directory: {filename}")
        return "File not found", 404
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {e}", exc_info=True)
        return "Error serving file", 500


# --- Initialize Database ---
def init_db():
    """Creates database tables if they don't exist."""
    with app.app_context():
        try:
            app.logger.info("Initializing database...")
            db.create_all()
            app.logger.info("Database tables created or already exist.")
        except Exception as e:
            app.logger.error(f"Error creating database tables: {e}", exc_info=True)
            # Depending on severity, you might want to exit or just log
            # raise # Re-raise if you want the app to stop on DB init failure

# --- Run Application ---
if __name__ == '__main__':
    init_db() # Ensure DB is created before running the app server
    # Use debug=True ONLY for development! It auto-reloads and provides detailed errors in browser.
    # Set host='0.0.0.0' to make it accessible on your local network (use with caution).
    # Use host='127.0.0.1' (default) for access only on your machine.
    app.logger.info("Starting Flask application server...")
    app.run(debug=True, host='0.0.0.0') # Set debug=False for production!
