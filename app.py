# ---- IMPORT THE LIBRARIES ----
import streamlit as st        # streamlit: turns Python into a web app
import numpy as np            # numpy: for the distance/angle math
import joblib                 # joblib: to load our saved trained model

# ---- LOAD THE TRAINED MODEL ----
# Load the model file we saved earlier. This happens once when the app starts.
# @st.cache_resource tells Streamlit to keep the model in memory (don't reload every click)
@st.cache_resource
def load_model():
    return joblib.load('xg_model.pkl')   # read the saved model from disk

model = load_model()   # actually load it into a variable we can use

# ---- PAGE TITLE AND INTRO ----
# st.title puts a big heading at the top of the app
st.title("⚽ Expected Goals (xG) Calculator")

# st.write puts normal text on the page
st.write("Pick where a shot was taken and some details, and the model predicts "
         "the chance it becomes a goal.")

# ---- USER INPUTS ----
# We'll let the user describe a shot using sliders and checkboxes.

# A subheading for this section
st.subheader("Where was the shot taken?")

# Slider for how far along the pitch (x). Pitch is 0-120, goal at 120.
# st.slider(label, min, max, default) creates a draggable slider.
shot_x = st.slider("Distance along pitch (0 = own goal, 120 = attacking goal)",
                   0, 120, 105)   # default 105 = a decent attacking position

# Slider for how far across the pitch (y). Pitch is 0-80, center is 40.
shot_y = st.slider("Position across pitch (0 = left side, 40 = center, 80 = right side)",
                   0, 80, 40)     # default 40 = central

# A subheading for the shot-type inputs
st.subheader("Shot details")

# Checkboxes for the shot type. st.checkbox returns True if ticked, False if not.
is_header = st.checkbox("Was it a header?")          # True/False
is_penalty = st.checkbox("Was it a penalty?")        # True/False
is_open_play = st.checkbox("Was it from open play?", value=True)  # ticked by default

# ---- CALCULATE THE FEATURES FROM THE INPUTS ----
# We turn the user's inputs into the SAME features the model was trained on.

# Distance to goal center (120, 40), using Pythagoras — same formula as Phase 2
x_dist = 120 - shot_x                          # gap to goal line
y_dist = abs(shot_y - 40)                       # gap from center
distance = np.sqrt(x_dist**2 + y_dist**2)       # straight-line distance

# Angle to goal — same trig formula as Phase 2
goal_width = 8
angle = np.arctan2(goal_width * x_dist,
                   x_dist**2 + y_dist**2 - (goal_width / 2)**2)
if angle < 0:                                   # keep the angle positive
    angle = angle + np.pi

# ---- MAKE THE PREDICTION ----
# Build the feature list in the SAME order the model expects:
# [distance, angle, is_header, is_penalty, is_open_play]
# int(True) = 1, int(False) = 0, so the checkboxes become 1s and 0s.
features = [[distance, angle, int(is_header), int(is_penalty), int(is_open_play)]]

# Ask the model for the probability of a goal.
# predict_proba returns [[prob_no_goal, prob_goal]], so [0][1] grabs the goal probability.
xg_value = model.predict_proba(features)[0][1]

# ---- SHOW THE RESULT ----
# A subheading for the result
st.subheader("Predicted xG")

# st.metric shows a big number nicely. We show the xG as a percentage.
st.metric("Chance of scoring", f"{xg_value:.1%}")   # e.g. "23.4%"

# Add a little context message depending on how good the chance is
if xg_value > 0.3:
    st.success("That's a great chance! 🔥")          # green box for good chances
elif xg_value > 0.1:
    st.info("Decent opportunity. ⚽")                # blue box for medium
else:
    st.warning("Low chance — tough shot. 😬")        # yellow box for poor chances

# Show the calculated distance and angle too, so the user sees the maths
st.write(f"Distance to goal: {distance:.1f} units")
st.write(f"Shooting angle: {angle:.2f} radians")