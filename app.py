# ---- IMPORT THE LIBRARIES ----
import streamlit as st        # streamlit: turns Python into a web app
import numpy as np            # numpy: for the distance/angle math
import pandas as pd           # pandas: to load the player stats CSV
import joblib                 # joblib: to load our saved trained model
import matplotlib.pyplot as plt  # matplotlib: to draw the player scatter chart

# ---- LOAD THE MODEL AND DATA (once, when the app starts) ----
# @st.cache_resource keeps the model in memory so it doesn't reload on every click
@st.cache_resource
def load_model():
    return joblib.load('xg_model.pkl')   # read the saved model from disk

# @st.cache_data keeps the player data in memory too
@st.cache_data
def load_player_stats():
    return pd.read_csv('player_stats.csv')   # read the player stats CSV

model = load_model()                 # load the trained model
player_stats = load_player_stats()   # load the player table

# ---- SIDEBAR: let the user pick which page to view ----
# st.sidebar puts things in a panel on the left.
# st.radio makes a set of round option buttons.
page = st.sidebar.radio("Choose a page:",
                        ["xG Calculator", "Player Analysis"])

# ============================================================
# PAGE 1: THE xG CALCULATOR
# ============================================================
if page == "xG Calculator":

    # Big title at the top
    st.title("⚽ Expected Goals (xG) Calculator")

    # Intro text
    st.write("Pick where a shot was taken and some details, and the model "
             "predicts the chance it becomes a goal.")

    # --- User inputs for the shot ---
    st.subheader("Where was the shot taken?")

    # Slider for position along the pitch (0-120, goal at 120)
    shot_x = st.slider("Distance along pitch (0 = own goal, 120 = attacking goal)",
                       0, 120, 105)

    # Slider for position across the pitch (0-80, center at 40)
    shot_y = st.slider("Position across pitch (0 = left, 40 = center, 80 = right)",
                       0, 80, 40)

    st.subheader("Shot details")

    # Checkboxes for the shot type (return True/False)
    is_header = st.checkbox("Was it a header?")
    is_penalty = st.checkbox("Was it a penalty?")
    is_open_play = st.checkbox("Was it from open play?", value=True)

    # --- Calculate the features from the inputs (same formulas as Phase 2) ---
    x_dist = 120 - shot_x                       # gap to goal line
    y_dist = abs(shot_y - 40)                    # gap from center
    distance = np.sqrt(x_dist**2 + y_dist**2)    # straight-line distance

    goal_width = 8                               # goal is 8 units wide
    angle = np.arctan2(goal_width * x_dist,
                       x_dist**2 + y_dist**2 - (goal_width / 2)**2)
    if angle < 0:                                # keep angle positive
        angle = angle + np.pi

    # --- Make the prediction ---
    # Build features in the same order the model expects
    features = [[distance, angle, int(is_header), int(is_penalty), int(is_open_play)]]
    xg_value = model.predict_proba(features)[0][1]   # probability of a goal

    # --- Show the result ---
    st.subheader("Predicted xG")
    st.metric("Chance of scoring", f"{xg_value:.1%}")   # show as a percentage

    # Context message based on how good the chance is
    if xg_value > 0.3:
        st.success("That's a great chance! 🔥")
    elif xg_value > 0.1:
        st.info("Decent opportunity. ⚽")
    else:
        st.warning("Low chance — tough shot. 😬")

    # Show the calculated numbers
    st.write(f"Distance to goal: {distance:.1f} units")
    st.write(f"Shooting angle: {angle:.2f} radians")

# ============================================================
# PAGE 2: THE PLAYER ANALYSIS
# ============================================================
elif page == "Player Analysis":

    # Title for this page
    st.title("📊 Player Analysis — Who Beats Their xG?")

    # Explanation
    st.write("Comparing each player's actual goals to their expected goals (xG) "
             "from La Liga 2015/16. Players **above** the line are clinical finishers; "
             "**below** the line means they under-performed their chances.")
    st.write("*(Only players with 20+ shots are shown.)*")

    # --- The scatter chart: xG vs actual goals ---
    fig, ax = plt.subplots(figsize=(10, 7))   # create a chart

    # Plot every player as a dot
    ax.scatter(player_stats['xg'], player_stats['goals'],
               alpha=0.5, s=40, color='steelblue')

    # The diagonal "average finishing" line
    max_val = player_stats['goals'].max() + 5
    ax.plot([0, max_val], [0, max_val], 'k--', label='Average finishing (goals = xG)')

    # Label the top 5 over-performers and top 3 under-performers
    players_to_label = pd.concat([
        player_stats.sort_values('goals_minus_xg', ascending=False).head(5),
        player_stats.sort_values('goals_minus_xg', ascending=True).head(3)
    ])
    for _, row in players_to_label.iterrows():
        short_name = row['player'].split()[0]   # first name only, to keep it clean
        ax.annotate(short_name, (row['xg'], row['goals']),
                    fontsize=9, xytext=(5, 5), textcoords='offset points')

    # Labels and title
    ax.set_xlabel('Expected Goals (xG)')
    ax.set_ylabel('Actual Goals')
    ax.set_title('Which players beat their xG? (La Liga 2015/16)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show the chart in the app
    st.pyplot(fig)

    # --- Tables of top over- and under-performers ---
    st.subheader("Top 10 over-performers (clinical finishers)")
    # Sort by goals_minus_xg (highest first) and show the top 10
    top_over = player_stats.sort_values('goals_minus_xg', ascending=False).head(10)
    st.dataframe(top_over[['player', 'shots', 'goals', 'xg', 'goals_minus_xg']])

    st.subheader("Top 10 under-performers")
    # Sort by goals_minus_xg (lowest first) and show the top 10
    top_under = player_stats.sort_values('goals_minus_xg', ascending=True).head(10)
    st.dataframe(top_under[['player', 'shots', 'goals', 'xg', 'goals_minus_xg']])