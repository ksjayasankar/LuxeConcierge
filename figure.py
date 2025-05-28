import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
import logging

# --- Configuration ---
# Option 1: Database loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in project root
DB_NAME = "luxury_leather.db"
DB_URL = f"sqlite:///{os.path.join(BASE_DIR, DB_NAME)}"
LOAD_FROM_DB = True # Set to False if using Option 2

# Option 2: Manual DataFrame creation (if DB is not accessible or for testing)
# Ensure column names match your actual data:
# 'id', 'name', 'description', 'price', 'leather_type', 'color',
# 'style', 'hardware', 'occasion', 'craftsmanship_score', 'sustainability_score'
MANUAL_DATA = {
    'style': ['Briefcase', 'Tote', 'Briefcase', 'Handbag', 'Messenger', 'Tote', 'Sling', 'Briefcase', 'Handbag', 'Duffel'],
    'price': [450, 320, 680, 550, 390, 280, 210, 750, 480, 600],
    'craftsmanship_score': [8.5, 7.8, 9.2, 8.8, 8.0, 7.5, 7.2, 9.5, 8.7, 8.9],
    'sustainability_score': [7.0, 8.0, 6.5, 7.5, 6.8, 8.5, 7.8, 6.0, 7.2, 7.0]
    # Add other columns if needed, or just the ones for plotting
}

OUTPUT_DIR = "charts" # Directory to save charts
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Data ---
df = None
if LOAD_FROM_DB:
    logging.info(f"Attempting to load data from database: {DB_URL}")
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as connection:
            df = pd.read_sql_table("products", con=connection)
        logging.info(f"Successfully loaded {len(df)} records from the database.")
    except Exception as e:
        logging.error(f"Failed to load data from database: {e}")
        logging.info("Falling back to manual data for demonstration.")
        df = pd.DataFrame(MANUAL_DATA) # Fallback to manual data
else:
    logging.info("Using manually defined DataFrame for demonstration.")
    df = pd.DataFrame(MANUAL_DATA)

# --- Data Validation ---
if df is None or df.empty:
    logging.critical("DataFrame is empty. Cannot generate plots.")
    exit()

required_cols = ['style', 'price', 'craftsmanship_score', 'sustainability_score']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    logging.critical(f"DataFrame is missing required columns: {missing_cols}. Cannot generate plots.")
    exit()

# Convert scores to numeric if they aren't already, handling potential errors
df['craftsmanship_score'] = pd.to_numeric(df['craftsmanship_score'], errors='coerce')
df['sustainability_score'] = pd.to_numeric(df['sustainability_score'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Drop rows where essential numeric data is missing after conversion
df.dropna(subset=['price', 'craftsmanship_score', 'sustainability_score'], inplace=True)

if df.empty:
    logging.critical("DataFrame is empty after handling missing numeric values. Cannot generate plots.")
    exit()


# --- Generate Plots (Combined Figure 3.1) ---
logging.info("Generating combined distribution plots...")
sns.set_theme(style="whitegrid") # Set a nice theme

fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # Create a 2x2 grid of subplots
fig.suptitle('Figure 3.1: Distribution of Key Product Attributes', fontsize=16, y=1.02)

# Plot 1: Style Distribution (Bar Chart)
style_ax = axes[0, 0]
sns.countplot(y=df['style'], ax=style_ax, order=df['style'].value_counts().index, palette="viridis")
style_ax.set_title('Distribution of Product Styles')
style_ax.set_xlabel('Number of Products')
style_ax.set_ylabel('Style Category')

# Plot 2: Price Distribution (Histogram)
price_ax = axes[0, 1]
sns.histplot(df['price'], kde=True, ax=price_ax, color='skyblue')
price_ax.set_title('Distribution of Product Prices')
price_ax.set_xlabel('Price')
price_ax.set_ylabel('Frequency')

# Plot 3: Craftsmanship Score Distribution (Histogram)
craft_ax = axes[1, 0]
sns.histplot(df['craftsmanship_score'], kde=True, ax=craft_ax, color='lightcoral', binwidth=0.5) # Adjust binwidth as needed
craft_ax.set_title('Distribution of Craftsmanship Scores')
craft_ax.set_xlabel('Craftsmanship Score (1-10)')
craft_ax.set_ylabel('Frequency')
craft_ax.set_xlim(min(5, df['craftsmanship_score'].min()-0.5), max(10, df['craftsmanship_score'].max()+0.5)) # Adjust x-limits


# Plot 4: Sustainability Score Distribution (Histogram)
sustain_ax = axes[1, 1]
sns.histplot(df['sustainability_score'], kde=True, ax=sustain_ax, color='mediumseagreen', binwidth=0.5) # Adjust binwidth as needed
sustain_ax.set_title('Distribution of Sustainability Scores')
sustain_ax.set_xlabel('Sustainability Score (1-10)')
sustain_ax.set_ylabel('Frequency')
sustain_ax.set_xlim(min(5, df['sustainability_score'].min()-0.5), max(10, df['sustainability_score'].max()+0.5)) # Adjust x-limits


# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to prevent overlap, leave space for suptitle
figure_path = os.path.join(OUTPUT_DIR, "figure_3_1_dataset_distributions.png")
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
logging.info(f"Combined distribution chart saved to: {figure_path}")

# --- Display Plots (Optional) ---
# plt.show()

logging.info("Script finished.")