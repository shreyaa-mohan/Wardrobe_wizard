# 👕 Smart Wardrobe Outfit Recommender

A personalized outfit suggestion app built with **Flask**, **SQLAlchemy**, and **machine learning**.

This intelligent wardrobe assistant lets users:
- Upload clothing items with images and metadata
- Automatically extract **color features** and **visual embeddings**
- Generate **outfit combinations** based on occasion
- View results with image previews and compatibility scores

---

## 🎯 Features

- 🧠 ML-powered outfit matching
- 🎨 Extracts **dominant colors** and **embedding vectors**
- 🧥 Handles tops, bottoms, shoes, accessories, dresses, outerwear
- 📷 Upload and preview clothing item images
- 📅 Tag items for specific occasions (e.g. casual, formal, party)
- ✅ Clean, responsive web UI with Flask + Jinja templates
- 📦 Robust backend with **SQLite + SQLAlchemy**
- 🧾 Smart error handling and logging

---

## 🛠 Tech Stack

| Layer         | Tools/Frameworks |    
|-------------- |------------------|  
| Web Framework | Flask            |  
| Database      | SQLite + SQLAlchemy |  
| Templating    | Jinja2 (HTML)     |  
| ML Backend    | NumPy, Custom feature extractor & recommender modules |  
| Image Uploads | `werkzeug.utils.secure_filename` |  
| Logging       | Python logging module (file + console ready) |  

---

### 1. Clone the repository
git clone https://github.com/shreyaa-mohan/Wardrobe_wizard.git  
cd wardrobe_wizard  

### 2. Install dependencies
pip install -r requirements.txt 
### 3. Create upload and instance folders
mkdir uploads  
mkdir instance  
### 4. Run the Flask app
python app.py  
### 📦 Example Use Case
Upload:  

Red floral top (party)  

White trousers (casual, party)  

Nude heels (party)  

Generate:  

✅ Top + Bottom + Shoes combination for "party"  

With compatibility score and image preview  


