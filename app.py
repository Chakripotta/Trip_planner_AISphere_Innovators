import streamlit as st
from Trip_Planner import TripPlanner, TripPlannerError
import os
from datetime import date, timedelta
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()

def main():
    st.title("🌍 AI Trip Planner")
    st.write("Plan your next adventure with a detailed, weather-aware itinerary!")

    load_env()
    # --- Configuration and Initialization ---
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

    if not GCP_PROJECT_ID or not OPENWEATHERMAP_API_KEY:
        st.error("Missing required environment variables: GCP_PROJECT_ID and/or OPENWEATHERMAP_API_KEY.")
        st.stop()

    try:
        # Use a cache to avoid re-initializing the planner on every interaction
        @st.cache_resource
        def get_planner():
            return TripPlanner(project_id=GCP_PROJECT_ID, location=GCP_LOCATION)
        
        planner = get_planner()
    except TripPlannerError as e:
        st.error(f"Failed to initialize Trip Planner: {e}")
        st.stop()

    # --- User Input Form ---
    with st.form("trip_form"):
        destination = st.text_input("📍 Destination (e.g., Goa, India or Northern Italy):", "Paris, France")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("🗓️ Start Date:", value=date.today() + timedelta(days=7))
        with col2:
            end_date = st.date_input("🗓️ End Date:", value=date.today() + timedelta(days=14))

        preference_options = {
            "Popular & Famous Places": "1",
            "Hidden Gems & Off-the-beaten-path": "2",
            "A Mix of Both": "3",
        }
        preference = st.selectbox("✈️ Travel Style:", list(preference_options.keys()))

        submitted = st.form_submit_button("Plan My Trip ✨")

    if submitted:
        with st.spinner("AI is crafting your personalized trip plan... This may take a minute..."):
            try:
                itinerary = planner.generate_plan(
                    region=destination,
                    start_date_str=start_date.strftime("%Y-%m-%d"),
                    end_date_str=end_date.strftime("%Y-%m-%d"),
                    preference_choice=preference_options[preference]
                )
                st.subheader(f"🗺️ Your Custom Itinerary for {destination}")
                st.markdown(itinerary)
                st.success("✨ Your travel plan has been generated! Happy travels! ✨")
            except TripPlannerError as e:
                st.error(f"An error occurred during planning: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
