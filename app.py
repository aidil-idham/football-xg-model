# ---- IMPORT THE LIBRARIES ----
import streamlit as st                # streamlit: turns Python into a web app
import numpy as np                    # numpy: for the distance/angle math
import pandas as pd                   # pandas: to load the player stats CSV
import joblib                         # joblib: to load our saved trained model
import matplotlib.pyplot as plt       # matplotlib: to draw the pitch and charts
from matplotlib.patches import Circle, Rectangle, Arc  # shapes for the pitch
import io                             # io: to hold the pitch image in memory
from PIL import Image                 # PIL: to open the image for the click component
from streamlit_image_coordinates import streamlit_image_coordinates  # click-to-get-coordinates

# ---- PAGE CONFIG (must be first Streamlit command) ----
st.set_page_config(
    page_title="xG Model — Football Analytics",  # browser tab title
    page_icon="⚽",                                # browser tab icon
    layout="wide"                                  # use full screen width
)

# ---- CUSTOM STYLING: a polished navy + green football theme ----
# All the CSS that makes the app look designed rather than default.
st.markdown("""
    <style>
    /* Bold sporty font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* Deep navy gradient background */
    .stApp {
        background: linear-gradient(160deg, #0a2540 0%, #061a2e 100%);
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li { color: #e8eef5; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #061a2e;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Hero banner at the top of the calculator */
    .hero {
        background: linear-gradient(120deg, #1e88e5 0%, #0a2540 100%);
        padding: 30px 35px; border-radius: 16px; margin-bottom: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .hero h1 { margin: 0; font-size: 38px; font-weight: 900 !important; }
    .hero p { margin: 8px 0 0 0; font-size: 16px; color: #cfe0f0; }

    /* White result card */
    .card {
        background: #ffffff; color: #0a2540;
        padding: 28px; border-radius: 16px; margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }
    .card p, .card h1, .card h2, .card h3 { color: #0a2540 !important; }

    /* Stat badge row (like the trophy row in football sites) */
    .badge-row { display: flex; gap: 14px; margin-bottom: 25px; }
    .badge {
        flex: 1; background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 16px; text-align: center;
    }
    .badge .num { font-size: 26px; font-weight: 800; color: #4da6ff; }
    .badge .lbl { font-size: 12px; color: #9fb3c8; text-transform: uppercase; letter-spacing: 1px; }

    /* Make Streamlit checkboxes/text a bit cleaner */
    .stCheckbox { padding: 4px 0; }
    </style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL AND DATA (cached) ----
@st.cache_resource
def load_model():
    return joblib.load('xg_model.pkl')

@st.cache_data
def load_player_stats():
    return pd.read_csv('player_stats.csv')

model = load_model()
player_stats = load_player_stats()

# ---- PITCH DIMENSIONS ----
PITCH_LENGTH = 120
PITCH_WIDTH = 80
SCALE = 7
IMG_W = PITCH_LENGTH * SCALE
IMG_H = PITCH_WIDTH * SCALE

# ---- FUNCTION: draw a nice-looking pitch and return it as an image ----
def draw_pitch(shot_x=None, shot_y=None):
    fig, ax = plt.subplots(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)

    # Rich dark green base
    base_green = '#0f5c2e'
    stripe_green = '#126633'
    ax.set_facecolor(base_green)
    fig.patch.set_facecolor(base_green)

    # --- Mowing stripes (alternating vertical bands) for a real-pitch look ---
    stripe_width = 12
    for i, x0 in enumerate(range(0, 120, stripe_width)):
        if i % 2 == 0:
            ax.add_patch(Rectangle((x0, 0), stripe_width, 80,
                                   color=stripe_green, zorder=0))

    line = 'white'
    lw = 2
    # Outer boundary
    ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], color=line, linewidth=lw, zorder=2)
    # Halfway line
    ax.plot([60, 60], [0, 80], color=line, linewidth=lw, zorder=2)
    # Center circle + spot
    ax.add_patch(Circle((60, 40), 10, color=line, fill=False, linewidth=lw, zorder=2))
    ax.add_patch(Circle((60, 40), 0.6, color=line, zorder=2))

    # Right penalty area (attacking, x=120)
    ax.add_patch(Rectangle((102, 22), 18, 36, color=line, fill=False, linewidth=lw, zorder=2))
    ax.add_patch(Rectangle((114, 30), 6, 20, color=line, fill=False, linewidth=lw, zorder=2))
    ax.add_patch(Circle((108, 40), 0.6, color=line, zorder=2))
    ax.add_patch(Arc((108, 40), 20, 20, theta1=308, theta2=52, color=line, linewidth=lw, zorder=2))  # D arc
    # Left penalty area
    ax.add_patch(Rectangle((0, 22), 18, 36, color=line, fill=False, linewidth=lw, zorder=2))
    ax.add_patch(Rectangle((0, 30), 6, 20, color=line, fill=False, linewidth=lw, zorder=2))
    ax.add_patch(Circle((12, 40), 0.6, color=line, zorder=2))
    ax.add_patch(Arc((12, 40), 20, 20, theta1=128, theta2=232, color=line, linewidth=lw, zorder=2))

    # Goals
    ax.plot([120, 120], [36, 44], color='#ffd400', linewidth=5, zorder=3)  # attacking goal (gold)
    ax.plot([0, 0], [36, 44], color=line, linewidth=4, zorder=3)

    # Shot marker with a glow effect
    if shot_x is not None and shot_y is not None:
        ax.add_patch(Circle((shot_x, shot_y), 3.2, color='#ff5252', alpha=0.25, zorder=4))  # glow
        ax.add_patch(Circle((shot_x, shot_y), 1.6, color='#ff1744', zorder=5))              # dot
        ax.add_patch(Circle((shot_x, shot_y), 1.6, color='white', fill=False, linewidth=1.2, zorder=6))

    # Attack direction hint
    ax.annotate('ATTACK →', (92, 76), color='#ffd400', fontsize=11, fontweight='bold', zorder=3)

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=base_green)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---- SIDEBAR ----
st.sidebar.markdown("## ⚽ xG Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["xG Calculator", "Player Analysis", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Built with StatsBomb data\nLa Liga 2015/16 • 9,168 shots")

# ============================================================
# PAGE 1: xG CALCULATOR
# ============================================================
if page == "xG Calculator":

    # Hero banner
    st.markdown("""
        <div class="hero">
            <h1>⚽ Expected Goals Calculator</h1>
            <p>Click anywhere on the pitch to place a shot — the model predicts its chance of becoming a goal. You're attacking the gold goal on the right.</p>
        </div>
    """, unsafe_allow_html=True)

    # Stat badges row (project highlights, like a football site's trophy row)
    st.markdown("""
        <div class="badge-row">
            <div class="badge"><div class="num">9,168</div><div class="lbl">Shots analysed</div></div>
            <div class="badge"><div class="num">0.81</div><div class="lbl">Model AUC</div></div>
            <div class="badge"><div class="num">5</div><div class="lbl">Features</div></div>
            <div class="badge"><div class="num">151</div><div class="lbl">Players ranked</div></div>
        </div>
    """, unsafe_allow_html=True)

    if 'shot_x' not in st.session_state:
        st.session_state.shot_x = 105
        st.session_state.shot_y = 40

    col_pitch, col_result = st.columns([2, 1])

    with col_pitch:
        pitch_img = draw_pitch(st.session_state.shot_x, st.session_state.shot_y)
        coords = streamlit_image_coordinates(pitch_img, key="pitch")
        if coords is not None:
            clicked_x = coords['x'] / SCALE
            clicked_y = PITCH_WIDTH - (coords['y'] / SCALE)   # flip y so top=top
            if (abs(clicked_x - st.session_state.shot_x) > 0.1 or
                    abs(clicked_y - st.session_state.shot_y) > 0.1):
                st.session_state.shot_x = clicked_x
                st.session_state.shot_y = clicked_y
                st.rerun()

    shot_x = st.session_state.shot_x
    shot_y = st.session_state.shot_y

    with col_result:
        st.markdown("### Shot details")
        is_header = st.checkbox("Header?")
        is_penalty = st.checkbox("Penalty?")
        is_open_play = st.checkbox("Open play?", value=True)

        # Features (same formulas as Phase 2)
        x_dist = 120 - shot_x
        y_dist = abs(shot_y - 40)
        distance = np.sqrt(x_dist**2 + y_dist**2)
        goal_width = 8
        angle = np.arctan2(goal_width * x_dist,
                           x_dist**2 + y_dist**2 - (goal_width / 2)**2)
        if angle < 0:
            angle = angle + np.pi

        features = [[distance, angle, int(is_header), int(is_penalty), int(is_open_play)]]
        xg_value = model.predict_proba(features)[0][1]

        if xg_value > 0.3:
            verdict, colour = "Great chance 🔥", "#1e7a46"
        elif xg_value > 0.1:
            verdict, colour = "Decent opportunity ⚽", "#1565c0"
        else:
            verdict, colour = "Tough shot 😬", "#c05621"

        # Progress-bar style visual for the xG
        bar_pct = int(xg_value * 100)
        st.markdown(f"""
            <div class="card" style="text-align:center;">
                <p style="font-size:14px; letter-spacing:1px; text-transform:uppercase; color:#66788a !important;">Chance of scoring</p>
                <p style="font-size:60px; font-weight:900; color:{colour} !important; margin:5px 0;">{xg_value:.1%}</p>
                <div style="background:#e6ecf2; border-radius:20px; height:14px; width:100%; margin:12px 0;">
                    <div style="background:{colour}; width:{bar_pct}%; height:14px; border-radius:20px;"></div>
                </div>
                <p style="font-size:19px; font-weight:700; color:{colour} !important;">{verdict}</p>
                <hr style="border-color:#e6ecf2;">
                <p style="font-size:13px; color:#66788a !important;">Distance: {distance:.1f} units &nbsp;•&nbsp; Angle: {angle:.2f} rad</p>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 2: PLAYER ANALYSIS
# ============================================================
elif page == "Player Analysis":

    st.markdown("""
        <div class="hero">
            <h1>📊 Player Analysis</h1>
            <p>Actual goals vs expected goals. Players above the line are clinical finishers; below means they under-performed their chances.</p>
        </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a2540')
    ax.set_facecolor('#0a2540')
    ax.scatter(player_stats['xg'], player_stats['goals'],
               alpha=0.6, s=45, color='#4da6ff', edgecolor='white', linewidth=0.3)
    max_val = player_stats['goals'].max() + 5
    ax.plot([0, max_val], [0, max_val], '--', color='#ffd400',
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
    ax.set_title('Which players beat their xG? (La Liga 2015/16)', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#33506e')
    legend = ax.legend(facecolor='#0a2540', edgecolor='#33506e')
    for text in legend.get_texts():
        text.set_color('white')
    ax.grid(True, alpha=0.12)
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
    st.markdown("""
        <div class="hero">
            <h1>About this project</h1>
            <p>An end-to-end expected goals (xG) model — from raw data to a live app.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
            <h3>Football Expected Goals (xG) Model</h3>
            <p>A machine-learning model predicting the probability any shot becomes a goal,
            trained on StatsBomb open data (La Liga 2015/16 — 9,168 shots).</p>
            <p><b>What's inside:</b></p>
            <ul>
                <li>An xG model built from scratch (~0.81 AUC, close to StatsBomb's 0.85 using just 5 features).</li>
                <li>Rigorous evaluation — calibration, SHAP, ROC.</li>
                <li>A cross-league test of how xG transfers between leagues.</li>
                <li>Player analysis identifying clinical finishers vs under-performers.</li>
            </ul>
            <p><b>Tools:</b> Python, pandas, scikit-learn, SQLite, SHAP, Streamlit.</p>
        </div>
    """, unsafe_allow_html=True)