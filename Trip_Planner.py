import os
import re
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    Part,
    FunctionDeclaration,
)

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TripPlannerError(Exception):
    """Custom exception for trip planner errors"""
    pass

class TripPlanner:
    """
    An AI-powered trip planner that uses Gemini 2.5 Pro, a weather tool,
    and adapts to user travel preferences.
    """

    def __init__(self, project_id: str, location: str, model_name: str = "gemini-2.5-pro"):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.weather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        
        if not self.weather_api_key:
            raise TripPlannerError("CRITICAL: OPENWEATHERMAP_API_KEY environment variable not set.")
        
        # Validate API key early to fail fast
        self._validate_api_key()
        
        # Validate inputs
        if not project_id or not location:
            raise TripPlannerError("Project ID and location must be provided")
        
        try:
            vertexai.init(project=self.project_id, location=self.location)
            logger.info(f"Initialized Vertex AI with project: {self.project_id}, location: {self.location}")
        except Exception as e:
            raise TripPlannerError(f"Failed to initialize Vertex AI: {e}")

        weather_function_declaration =  FunctionDeclaration(
                name="get_daily_weather_forecasts",
                description="Get the day-by-day weather forecast for a list of cities.",
                parameters={
                    "type": "object",
                    "properties": {
                        "city_date_ranges": {
                            "type": "array",
                            "description": "A list of cities and the date ranges to get weather for.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "The city name."},
                                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format."},
                                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format."}
                                },
                                "required": ["city", "start_date", "end_date"]
                            }
                        }
                    },
                    "required": ["city_date_ranges"]
                },
            )
        
        self.weather_tool = Tool(function_declarations=[weather_function_declaration])
        
        try:
            logger.info(f"Initializing model: {self.model_name}")
            self.model = GenerativeModel(self.model_name, tools=[self.weather_tool])
            # Add weather cache for performance
            self.weather_cache = {}
            # Tool handlers for extensibility
            self.tool_handlers = {
                "get_daily_weather_forecasts": self._handle_weather_tool
            }
        except Exception as e:
            raise TripPlannerError(f"Failed to initialize Gemini model: {e}")

    def _validate_date_range(self, start_date_str: str, end_date_str: str) -> Tuple[datetime, datetime, int]:
        """Validates date range and returns parsed dates and duration"""
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError as e:
            raise TripPlannerError(f"Invalid date format: {e}")
        
        if end_date < start_date:
            raise TripPlannerError("End date must be after start date")
        
        # Check if dates are too far in the future (weather API limitation)
        max_future_date = datetime.now() + timedelta(days=5)
        if start_date > max_future_date:
            logger.warning("Weather forecast may not be available for dates more than 5 days in the future")
        
        duration = (end_date - start_date).days + 1
        if duration > 30:
            logger.warning("Trip duration exceeds 30 days - this may affect performance")
        
        return start_date, end_date, duration

    def _get_daily_weather_forecasts(self, city_date_ranges: List[Dict]) -> str:
        """Gets day-by-day weather forecasts for multiple cities in parallel."""
        if not city_date_ranges:
            return "No cities provided for weather forecast."
        
        all_results = []
        # Limit concurrent requests to avoid API rate limits
        max_workers = min(len(city_date_ranges), 5)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_city = {
                executor.submit(self._fetch_weather_for_city, entry): entry['city'] 
                for entry in city_date_ranges
            }
            
            for future in as_completed(future_to_city):
                city = future_to_city[future]
                try:
                    result = future.result(timeout=15)  # Add timeout
                    all_results.append(result)
                except Exception as exc:
                    error_msg = f"Error fetching weather for {city}: {exc}"
                    logger.error(error_msg)
                    all_results.append(error_msg)
        
        return "\n".join(all_results)

    def _fetch_weather_for_city(self, city_data: Dict) -> str:
        """Fetches and processes weather data for a single city."""
        city = city_data['city']
        start_date = city_data.get('start_date')
        end_date = city_data.get('end_date')
        
        # Create cache key based on city and current date (weather changes daily)
        cache_key = f"{city.lower()}-{datetime.now().strftime('%Y-%m-%d')}"
        
        # Check cache first
        if cache_key in self.weather_cache:
            logger.info(f"Returning cached weather for {city}")
            return self.weather_cache[cache_key]
        
        logger.info(f"Fetching weather for {city} from {start_date} to {end_date}")
        
        base_url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": city, 
            "appid": self.weather_api_key, 
            "units": "metric",
            "cnt": 40  # Limit to reduce data transfer
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            return f"Weather for {city}: API request failed - {e}"
        except requests.exceptions.JSONDecodeError:
            return f"Weather for {city}: Invalid response format"

        if not data.get('list') or data.get('cod') != '200':
            return f"Weather for {city}: No forecast data available or city not found (code: {data.get('cod', 'unknown')})"

        # Filter forecasts within the requested date range
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
        except ValueError:
            logger.error(f"Invalid date format for {city}: {start_date} to {end_date}")
            start_dt = end_dt = None

        daily_summary = {}
        for forecast in data['list']:
            forecast_dt = datetime.fromtimestamp(forecast['dt'])
            date_str = forecast_dt.strftime('%Y-%m-%d')
            
            # Filter by date range if provided
            if start_dt and end_dt:
                if not (start_dt <= forecast_dt <= end_dt + timedelta(days=1)):
                    continue
            
            if date_str not in daily_summary:
                daily_summary[date_str] = {
                    'temps': [], 
                    'conditions': [], 
                    'humidity': [],
                    'wind_speed': []
                }
            
            daily_summary[date_str]['temps'].append(forecast['main']['temp'])
            daily_summary[date_str]['conditions'].append(forecast['weather'][0]['description'])
            daily_summary[date_str]['humidity'].append(forecast['main'].get('humidity', 0))
            daily_summary[date_str]['wind_speed'].append(forecast.get('wind', {}).get('speed', 0))

        if not daily_summary:
            return f"Weather for {city}: No forecast data available for the requested date range"

        # Create detailed summary
        output_str = f"Weather forecast for {city}:\n"
        for day, values in sorted(daily_summary.items()):
            avg_temp = sum(values['temps']) / len(values['temps'])
            min_temp = min(values['temps'])
            max_temp = max(values['temps'])
            common_condition = max(set(values['conditions']), key=values['conditions'].count)
            avg_humidity = sum(values['humidity']) / len(values['humidity']) if values['humidity'] else 0
            avg_wind = sum(values['wind_speed']) / len(values['wind_speed']) if values['wind_speed'] else 0
            
            output_str += (f"- {day}: {avg_temp:.1f}°C (min: {min_temp:.1f}°C, max: {max_temp:.1f}°C), "
                          f"{common_condition}, humidity: {avg_humidity:.0f}%, wind: {avg_wind:.1f} m/s\n")
        
        # Cache the result before returning (reuse existing cache_key)
        self.weather_cache[cache_key] = output_str
        
        return output_str

    def _handle_weather_tool(self, city_date_ranges: List[Dict]) -> str:
        """Handle weather tool calls with proper argument handling"""
        return self._get_daily_weather_forecasts(city_date_ranges)

    def _validate_api_key(self):
        """Makes a test call to the weather API to validate the key."""
        logger.info("Validating OpenWeatherMap API key...")
        test_url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {"q": "London", "appid": self.weather_api_key, "cnt": 1}  # Minimal request
        try:
            response = requests.get(test_url, params=params, timeout=5)
            if response.status_code == 401:
                raise TripPlannerError("OpenWeatherMap API key is invalid or expired.")
            response.raise_for_status()
            logger.info("OpenWeatherMap API key validation successful.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not validate API key (continuing anyway): {e}")
            # Don't fail hard on network issues during validation - just warn

    def _validate_date(self, date_str: str) -> bool:
        """Validates that the date is in YYYY-MM-DD format and is reasonable."""
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            # Check if date is not too far in the past or future
            today = datetime.now()
            if date_obj < today - timedelta(days=1):
                logger.warning("Date is in the past")
            if date_obj > today + timedelta(days=365):
                logger.warning("Date is more than a year in the future")
            return True
        except ValueError:
            return False

    def _get_season(self, date: datetime) -> str:
        """Determine the season based on the date (Northern Hemisphere)"""
        month = date.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def _get_user_input_with_validation(self, prompt: str, validator=None, max_attempts: int = 3) -> str:
        """Get user input with validation and retry logic"""
        for attempt in range(max_attempts):
            try:
                user_input = input(prompt).strip()
                if not user_input:
                    print("Input cannot be empty. Please try again.")
                    continue
                
                if validator and not validator(user_input):
                    if attempt < max_attempts - 1:
                        print("Invalid input. Please try again.")
                        continue
                    else:
                        raise TripPlannerError("Maximum attempts exceeded for input validation")
                
                return user_input
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                raise TripPlannerError("User cancelled operation")
            except Exception as e:
                logger.error(f"Error getting user input: {e}")
                if attempt == max_attempts - 1:
                    raise
        
        raise TripPlannerError("Failed to get valid user input")

    def generate_plan(self, region: str, start_date_str: str, end_date_str: str, preference_choice: str) -> str:
        """
        Generates a trip plan based on provided parameters.
        This is designed to be called from a web UI like Streamlit.
        """
        try:
            # Validate date range
            start_date, end_date, trip_duration = self._validate_date_range(start_date_str, end_date_str)

            # Check if weather forecast is available (within 5 days)
            today = datetime.now()
            forecast_available = start_date <= today + timedelta(days=5)

            note = ""
            if not forecast_available:
                note = (
                    f"\nNote: Your trip starts {(start_date - today).days} days from now. "
                    "Real-time weather forecasts are only available for the next 5 days. "
                    "The AI will use seasonal weather patterns for your destination instead.\n"
                )

            # Get travel preference instruction
            preference_map = {
                '1': "Focus on the most popular, well-known, and highly-rated tourist destinations.",
                '2': "Focus on less explored, off-the-beaten-path, and unique hidden gems.",
                '3': "Create a balanced mix of popular attractions and hidden gems."
            }
            preference_instruction = preference_map.get(preference_choice)
            if not preference_instruction:
                raise TripPlannerError("Invalid preference choice.")

            # Create chat session
            self.chat = self.model.start_chat()

            # Create conditional prompt based on weather availability
            if forecast_available:
                prompt = f"""
You are a world-class, expert travel agent specializing in creating detailed, weather-aware itineraries. Your client wants a trip plan.

**Client's Request:**
- **Region:** "{region}"
- **Dates:** {start_date_str} to {end_date_str} (A {trip_duration}-day trip)
- **Travel Preference:** "{preference_instruction}"

**Your Task (Follow these steps precisely):**

1. **Validate Region:** First, determine if "{region}" is a real, travelable region. If not, respond with a polite message explaining the issue.

2. **Determine Locations:** For a {trip_duration}-day trip, choose 2-4 appropriate locations within {region} based on:
   - Trip duration (longer trips can cover more locations)
   - Travel logistics (reasonable distances between locations)
   - The client's stated preference

3. **Get Weather Data:** Call the `get_daily_weather_forecasts` tool for ALL chosen locations. Use the full date range ({start_date_str} to {end_date_str}) for each location.

4. **Create Weather-Optimized Itineraries:** After receiving weather data, create TWO distinct alternative itineraries that:
   - Take weather conditions into account for activity planning
   - Include indoor alternatives for poor weather days
   - Maximize outdoor activities on good weather days
   - Follow the client's travel preference

5. **Format Output:** Present each itinerary in clear Markdown format with:
   - **Day X (Date) - Location Name**
   - **Weather:** Summary from the forecast data
   - **Recommended Activities:** 2-3 specific activities suitable for the weather and location
   - **Weather Backup:** Alternative indoor activities if weather is poor

**Important Guidelines:**
- Consider weather when recommending outdoor vs. indoor activities
- Provide specific activity names, not just categories
- Include practical weather-related advice (clothing, gear)
- Keep descriptions concise but informative
- Don't show raw weather tool output in the final response

Generate the itineraries now.
"""
            else:
                # Prompt for future dates without weather tool
                month = start_date.strftime('%B')
                
                prompt = f"""
You are a world-class, expert travel agent specializing in creating detailed itineraries. Your client wants a trip plan.

**Client's Request:**
- **Region:** "{region}"
- **Dates:** {start_date_str} to {end_date_str} (A {trip_duration}-day trip, taking place in the month of {month})
- **Travel Preference:** "{preference_instruction}"

**Important Note:** Real-time weather forecasts are not available for these future dates. Base your recommendations on the **correct season and typical weather patterns** for "{region}" during {month}. Consider the geographical location to determine the appropriate season.

**Your Task (Follow these steps precisely):**

1. **Validate Region:** First, determine if "{region}" is a real, travelable region. If not, respond with a polite message explaining the issue.

2. **Determine Locations:** For a {trip_duration}-day trip, choose 2-4 appropriate locations within {region} based on:
   - Trip duration (longer trips can cover more locations)
   - Travel logistics (reasonable distances between locations)
   - The client's stated preference
   - Seasonal considerations for {month}

                3. **Create Season-Appropriate Itineraries:** Create TWO distinct alternative itineraries that:
   - Consider the correct seasonal weather patterns for {region} in {month} (accounting for hemisphere)
   - Include seasonal activities and attractions appropriate to the location
   - Provide clothing/gear recommendations for {month} weather in {region}
   - Account for seasonal opening hours and availability
   - Follow the client's travel preference

4. **Format Output:** Present each itinerary in clear Markdown format with:
   - **Day X (Date) - Location Name**
   - **Expected Weather:** Typical {month} conditions for the location (correct season)
   - **Recommended Activities:** 2-3 specific seasonal activities
   - **Season Tips:** Clothing, gear, and seasonal considerations

**Important Guidelines:**
- Use your knowledge of typical weather patterns for {region} in {month}
- **Correctly determine the season** based on the location's hemisphere
- Mention seasonal highlights (festivals, blooming seasons, etc.)
- Include practical seasonal advice (what to pack, best times of day)
- Provide backup indoor options for typical seasonal weather challenges
- Keep descriptions concise but informative

Generate the itineraries now.
"""
            
            start_time = time.time()
            response = self.chat.send_message(prompt)
            
            # Handle function calls only if weather forecast is available
            if forecast_available:
                response = self._handle_tool_calls(response)

            elapsed_time = time.time() - start_time
            logger.info(f"Total processing time: {elapsed_time:.2f} seconds")

            if response.candidates and response.candidates[0].content.parts:
                final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                return note + final_text
            else:
                return "No response generated. Please try again."

        except TripPlannerError as e:
            logger.error(f"Trip planning error in generate_plan: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_plan(): {e}", exc_info=True)
            raise TripPlannerError(f"An unexpected error occurred: {e}")

    def _handle_tool_calls(self, response):
        """Helper to process tool calls in a loop."""
        max_tool_calls = 5
        tool_call_count = 0
        
        while (response.candidates and 
               response.candidates[0].content.parts and 
               response.candidates[0].content.parts[0].function_call and 
               tool_call_count < max_tool_calls):
            
            tool_call_count += 1
            function_call = response.candidates[0].content.parts[0].function_call
            logger.info(f"Tool call #{tool_call_count}: {function_call.name}")
            
            # Use the modular tool handler approach
            if function_call.name in self.tool_handlers:
                try:
                    handler = self.tool_handlers[function_call.name]
                    # Pass arguments from the model to the handler
                    result = handler(**function_call.args)
                    
                    response = self.chat.send_message(
                        Part.from_function_response(
                            name=function_call.name,
                            response={'content': result}
                        )
                    )
                except Exception as e:
                    logger.error(f"Error in {function_call.name} tool call: {e}")
                    error_response = f"Tool unavailable due to error: {e}"
                    response = self.chat.send_message(
                        Part.from_function_response(
                            name=function_call.name,
                            response={'content': error_response}
                        )
                    )
            else:
                logger.warning(f"Unknown function call: {function_call.name}")
                break

        if tool_call_count >= max_tool_calls:
            logger.warning("Maximum tool calls reached")
        
        return response

    def plan(self):
        """Main method to run the trip planning interaction."""
        try:
            print("\n--- Welcome to the Gemini 2.5 Pro AI Trip Planner ---")
            
            region = self._get_user_input_with_validation("Enter a region or state (e.g., Goa, Garhwal): ")
            
            start_date_str = self._get_user_input_with_validation(
                "Enter the start date (YYYY-MM-DD): ", 
                validator=self._validate_date
            )
            
            end_date_str = self._get_user_input_with_validation(
                "Enter the end date (YYYY-MM-DD): ", 
                validator=self._validate_date
            )
            
            # Validate date range
            start_date, end_date, trip_duration = self._validate_date_range(start_date_str, end_date_str)
            
            print(f"\nTrip duration: {trip_duration} days")
            
            # Get travel preference
            preference_instruction = ""
            while True:
                print("\nWhat is your travel style?")
                print("  1: I want to visit the most popular and famous places.")
                print("  2: I want to explore less-known, hidden gems.")
                print("  3: I want a mix of both popular and hidden places.")
                
                choice = input("Enter your choice (1, 2, or 3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")

            print(f"\n...AI is thinking... This may take a moment...\n")
            
            result = self.generate_plan(region, start_date_str, end_date_str, choice)

            print("--- Here are your suggested itineraries, tailored to your preference and weather conditions ---\n")
            print(result)

        except TripPlannerError as e:
            print(f"\nTrip planning error: {e}")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            logger.error(f"Unexpected error in plan(): {e}", exc_info=True)
            print(f"\nAn unexpected error occurred: {e}")
            print("Please check your configuration and try again.")
