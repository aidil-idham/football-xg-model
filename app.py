# ---- IMPORT THE LIBRARIES ----
import streamlit as st                # streamlit: turns Python into a web app
import numpy as np                    # numpy: for the distance/angle math
import pandas as pd                   # pandas: to load the player stats CSV
import joblib                         # joblib: to load our saved trained model
import matplotlib.pyplot as plt       # matplotlib: to draw the pitch and charts
from matplotlib.patches import Circle, Rectangle, Arc  # shapes for drawing the pitch
import io                             # io: to turn our pitch drawing into an image in memory
from PIL import Image                 # PIL: to open that image so the click component can use it
from streamlit_image_coordinates import streamlit_image_coordinates  # the click-to-get-coordinates tool

# ---- PAGE CONFIG (must be first Streamlit command) ----
st.set_page_config(
    page_title="xG Model — Football Analytics",  # browser tab title
    page_icon="⚽",                                # browser tab icon
    layout="wide"                                  # use full screen width
)

# ---- CUSTOM STYLING (navy + white sporty theme) ----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp { background-color: #0a2540; }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li { color: #ffffff; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; }
    section[data-testid="stSidebar"] { background-color: #061a2e; }
    .card {
        background-color: #ffffff; color: #0a2540;
        padding: 25px; border-radius: 12px; margin-bottom: 20px;
    }
    .card h1, .card h2, .card h3, .card p { color: #0a2540 !important; }
    </style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL AND DATA (cached for speed) ----
@st.cache_resource
def load_model():
    return joblib.load('xg_model.pkl')       # load trained model

@st.cache_data
def load_player_stats():
    return pd.read_csv('player_stats.csv')   # load player stats

model = load_model()
player_stats = load_player_stats()

# ---- PITCH DIMENSIONS ----
# StatsBomb pitch is 120 long x 80 wide. We draw the whole pitch as an image.
PITCH_LENGTH = 120   # along the pitch (goals at 0 and 120)
PITCH_WIDTH = 80     # across the pitch

# We'll draw the image at this many pixels per pitch-unit, so we can convert clicks back.
SCALE = 7            # 7 pixels per unit -> image is 840 x 560 pixels
IMG_W = PITCH_LENGTH * SCALE   # image width in pixels
IMG_H = PITCH_WIDTH * SCALE    # image height in pixels

# ---- FUNCTION: draw the football pitch and return it as an image ----
def draw_pitch(shot_x=None, shot_y=None):
    # Create a figure sized exactly to our pixel dimensions
    fig, ax = plt.subplots(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax.set_facecolor('#1a7a3c')          # green pitch colour
    fig.patch.set_facecolor('#1a7a3c')

    # Draw the outer boundary of the pitch
    ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], color='white', linewidth=2)
    # Halfway line down the middle
    ax.plot([60, 60], [0, 80], color='white', linewidth=2)
    # Center circle and center spot
    ax.add_patch(Circle((60, 40), 10, color='white', fill=False, linewidth=2))
    ax.add_patch(Circle((60, 40), 0.5, color='white'))

    # --- Right-side penalty area (the goal we attack, at x=120) ---
    ax.add_patch(Rectangle((102, 22), 18, 36, color='white', fill=False, linewidth=2))  # big box
    ax.add_patch(Rectangle((114, 30), 6, 20, color='white', fill=False, linewidth=2))   # 6-yard box
    ax.add_patch(Circle((108, 40), 0.5, color='white'))                                 # penalty spot
    # --- Left-side penalty area (for visual completeness) ---
    ax.add_patch(Rectangle((0, 22), 18, 36, color='white', fill=False, linewidth=2))
    ax.add_patch(Rectangle((0, 30), 6, 20, color='white', fill=False, linewidth=2))
    ax.add_patch(Circle((12, 40), 0.5, color='white'))

    # --- The goals ---
    ax.plot([120, 120], [36, 44], color='yellow', linewidth=4)   # attacking goal (yellow, stands out)
    ax.plot([0, 0], [36, 44], color='white', linewidth=4)         # other goal

    # If a shot location is given, draw a red marker there
    if shot_x is not None and shot_y is not None:
        ax.add_patch(Circle((shot_x, shot_y), 1.5, color='red', zorder=5))

    # Add a small hint of which way to attack
    ax.annotate('ATTACK →', (95, 76), color='yellow', fontsize=9, fontweight='bold')

    # Clean up the axes (no ticks, exact limits)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.axis('off')                        # hide the axis lines/numbers
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # no white margin

    # Save the drawing into memory as a PNG, then open it as an image
    buf = io.BytesIO()                    # a place in memory to hold the image
    fig.savefig(buf, format='png', facecolor='#1a7a3c')
    plt.close(fig)                        # close the figure to free memory
    buf.seek(0)                           # rewind to the start of the buffer
    return Image.open(buf)                # return the image

# ---- SIDEBAR NAVIGATION ----
st.sidebar.markdown("## ⚽ xG Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["xG Calculator", "Player Analysis", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Built with StatsBomb data • La Liga 2015/16")

# ============================================================
# PAGE 1: xG CALCULATOR (click the pitch!)
# ============================================================
if page == "xG Calculator":

    st.markdown("# ⚽ Expected Goals Calculator")
    st.markdown("**Click anywhere on the pitch** to place a shot — the model predicts its chance of scoring. "
                "You're attacking the **right-hand goal** (yellow).")
    st.markdown("")

    # We remember the clicked spot using session_state so it survives reruns.
    # If no click yet, start with a default position near the box.
    if 'shot_x' not in st.session_state:
        st.session_state.shot_x = 105    # default x
        st.session_state.shot_y = 40     # default y

    # Two columns: the pitch on the left, the result on the right
    col_pitch, col_result = st.columns([2, 1])

    with col_pitch:
        # Draw the pitch showing the current shot marker
        pitch_img = draw_pitch(st.session_state.shot_x, st.session_state.shot_y)

        # Show the pitch and capture where the user clicks.
        # It returns the pixel coordinates of the click.
        coords = streamlit_image_coordinates(pitch_img, key="pitch")

        # If the user clicked, convert pixel coordinates -> pitch coordinates
        if coords is not None:
            # coords['x'] and coords['y'] are in pixels; divide by SCALE to get pitch units
            clicked_x = coords['x'] / SCALE
            # FLIP the y-axis: images count y from the TOP, but our pitch counts from the BOTTOM.
            # So we subtract from PITCH_WIDTH (80) to flip it right-side up.
            clicked_y = PITCH_WIDTH - (coords['y'] / SCALE)
            # Only update if the click actually moved (avoids needless reruns)
            if (abs(clicked_x - st.session_state.shot_x) > 0.1 or
                    abs(clicked_y - st.session_state.shot_y) > 0.1):
                st.session_state.shot_x = clicked_x
                st.session_state.shot_y = clicked_y
                st.rerun()   # redraw with the new marker

    # Read the current shot position
    shot_x = st.session_state.shot_x
    shot_y = st.session_state.shot_y

    with col_result:
        st.markdown("### Shot details")
        # Checkboxes for shot type
        is_header = st.checkbox("Header?")
        is_penalty = st.checkbox("Penalty?")
        is_open_play = st.checkbox("Open play?", value=True)

        # --- Calculate features (same formulas as Phase 2) ---
        x_dist = 120 - shot_x
        y_dist = abs(shot_y - 40)
        distance = np.sqrt(x_dist**2 + y_dist**2)
        goal_width = 8
        angle = np.arctan2(goal_width * x_dist,
                           x_dist**2 + y_dist**2 - (goal_width / 2)**2)
        if angle < 0:
            angle = angle + np.pi

        # --- Predict ---
        features = [[distance, angle, int(is_header), int(is_penalty), int(is_open_play)]]
        xg_value = model.predict_proba(features)[0][1]

        # Choose colour/message by chance quality
        if xg_value > 0.3:
            verdict, colour = "Great chance 🔥", "#1e7a46"
        elif xg_value > 0.1:
            verdict, colour = "Decent opportunity ⚽", "#2e6e8e"
        else:
            verdict, colour = "Tough shot 😬", "#b5651d"

        # Result card
        st.markdown(f"""
            <div class="card" style="text-align:center;">
                <p style="font-size:15px; margin-bottom:5px;">Chance of scoring</p>
                <p style="font-size:48px; font-weight:800; color:{colour} !important; margin:0;">
                    {xg_value:.1%}
                </p>
                <p style="font-size:18px; font-weight:600; color:{colour} !important;">{verdict}</p>
                <hr>
                <p style="font-size:13px;">Distance: {distance:.1f} • Angle: {angle:.2f} rad</p>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 2: PLAYER ANALYSIS
# ============================================================
elif page == "Player Analysis":

    st.markdown("# 📊 Player Analysis")
    st.markdown("Actual goals vs expected goals (xG). Players **above** the line are clinical finishers; "
                "**below** means they under-performed their chances.")
    st.markdown("")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a2540')
    ax.set_facecolor('#0a2540')
    ax.scatter(player_stats['xg'], player_stats['goals'],
               alpha=0.6, s=45, color='#4da6ff', edgecolor='white', linewidth=0.3)
    max_val = player_stats['goals'].max() + 5
    ax.plot([0, max_val], [0, max_val], '--', color='white',
            label='Average finishing (goals = xG)')
    players_to_label = pd.concat([
        player_stats.sort_values('goals_minus_xg', ascending=False).head(5),
        player_stats.sort_values('goals_minus_xg', ascending=True).head(3)
    ])
    for _, row in players_to_label.iterrows():
        ax.annotate(row['player'].split()[0], (row['xg'], row['goals']),
                    fontsize=9, color='white', xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Expected Goals (xG)', color='white')
    ax.set_ylabel('Actual Goals', color='white')
    ax.set_title('Which players beat their xG? (La Liga 2015/16)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    legend = ax.legend(facecolor='#0a2540', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    ax.grid(True, alpha=0.15)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔥 Top over-performers")
        top_over = player_stats.sort_values('goals_minus_xg', ascending=False).head(10)
        st.dataframe(top_over[['player', 'goals', 'xg', 'goals_minus_xg']],
                     use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### 😬 Top under-performers")
        top_under = player_stats.sort_values('goals_minus_xg', ascending=True).head(10)
        st.dataframe(top_under[['player', 'goals', 'xg', 'goals_minus_xg']],
                     use_container_width=True, hide_index=True)

# ============================================================
# PAGE 3: ABOUT
# ============================================================
elif page == "About":
    st.markdown("# About this project")
    st.markdown("""
        <div class="card">
            <h3>Football Expected Goals (xG) Model</h3>
            <p>A machine-learning model predicting the probability any shot becomes a goal,
            trained on StatsBomb open data (La Liga 2015/16 — 9,168 shots).</p>
            <p><b>What's inside:</b></p>
            <ul>
                <li>An xG model built from scratch (~0.81 AUC, close to StatsBomb's 0.85 using just 5 features).</li>
                <li>A cross-league test of how xG transfers between leagues.</li>
                <li>Player analysis identifying clinical finishers vs under-performers.</li>
            </ul>
            <p><b>Tools:</b> Python, pandas, scikit-learn, SQLite, SHAP, Streamlit.</p>
        </div>
    """, unsafe_allow_html=True)