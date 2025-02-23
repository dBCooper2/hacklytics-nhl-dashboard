import streamlit as st

st.title("Hockey Play by Play Data Dictionary!")

# Friendly text with a button right underneath
st.write("Feel free to ask GrittyLM any questions you have!")

# Create a clickable button that navigates to the specified URL in a new tab

if st.button('Talk to Gritty!'):
    st.switch_page('pages/gritty-llm.py')

# Present the data dictionary as Markdown
st.markdown(
    """
| Field Name         | Data Type | Description                                                                 | Example                     |
|:-------------------|:---------:|:----------------------------------------------------------------------------|:----------------------------|
| Game_Id            | Integer   | Unique identifier for the game.                                             | 20001                       |
| Date               | Date      | Date of the game (MM/DD/YYYY).                                              | 10/2/2019                   |
| Period             | Integer   | The game period (1=First, 2=Second, 3=Third, etc.).                         | 1                           |
| Event              | String    | The event type (e.g., PSTR = Period Start, FAC = Faceoff, GOAL, MISS, etc.).| "PSTR"                      |
| Description        | String    | More detailed description of the event.                                     | "Period Start- Local time…" |
| Time_Elapsed       | String    | Time within the period (MM:SS format) when the event occurred.              | "0:25"                      |
| Seconds_Elapsed    | Integer   | Time in seconds from the start of the period for the event.                 | 25                          |
| Strength           | String    | On-ice player strength (e.g., 5x5, 5x4).                                     | "5x5"                       |
| Ev_Zone            | String    | Zone where the event occurred (Off, Neu, Def).                              | "Neu"                       |
| Type               | String    | Specific shot/event type (TIP-IN, SLAP SHOT, etc.).                         | "TIP-IN"                    |
| Ev_Team            | String    | Team associated with the event (home or away abbreviation).                 | "OTT"                       |
| Home_Zone          | String    | Home team’s zone classification for the event (Off, Neu, Def).             | "Def"                       |
| Away_Team          | String    | Away team’s 3-letter abbreviation.                                          | "OTT"                       |
| Home_Team          | String    | Home team’s 3-letter abbreviation.                                          | "TOR"                       |
| p1_name            | String    | Primary player’s name involved in the event.                                | "BRADY TKACHUK"             |
| p1_ID              | Integer   | ID of the primary player.                                                   | 8480801                     |
| p2_name            | String    | Secondary player’s name (assist or faceoff counterpart).                    | "CONNOR BROWN"             |
| p2_ID              | Integer   | ID of the secondary player.                                                 | 8477015                     |
| p3_name            | String    | Tertiary player’s name (additional assist, faceoff counterpart).            | "COLIN WHITE"              |
| p3_ID              | Integer   | ID of the tertiary player.                                                  | 8478400                     |
| awayPlayer1        | String    | First away player on the ice.                               | "COLIN WHITE"              |
| awayPlayer1_id     | Integer   | ID of the first away player.                                               | 8478400                     |
| awayPlayer2        | String    | Second away player on the ice.                                             | "CONNOR BROWN"             |
| awayPlayer2_id     | Integer   | ID of the second away player.                                              | 8477015                     |
| awayPlayer3        | String    | Third away player on the ice.                                              | "BRADY TKACHUK"            |
| awayPlayer3_id     | Integer   | ID of the third away player.                                               | 8480801                     |
| awayPlayer4        | String    | Fourth away player on the ice.                                             | "ERIK BRANNSTROM"          |
| awayPlayer4_id     | Integer   | ID of the fourth away player.                                              | 8480073                     |
| awayPlayer5        | String    | Fifth away player on the ice.                                              | "RON HAINSEY"              |
| awayPlayer5_id     | Integer   | ID of the fifth away player.                                               | 8468493                     |
| awayPlayer6        | String    | Sixth away player on the ice (often the goalie).                            | "CRAIG ANDERSON"           |
| awayPlayer6_id     | Integer   | ID of the sixth away player.                                               | 8467950                     |
| homePlayer1        | String    | First home player on the ice.                                              | "MITCHELL MARNER"          |
| homePlayer1_id     | Integer   | ID of the first home player.                                               | 8478483                     |
| homePlayer2        | String    | Second home player on the ice.                                             | "JOHN TAVARES"             |
| homePlayer2_id     | Integer   | ID of the second home player.                                              | 8475166                     |
| homePlayer3        | String    | Third home player on the ice.                                              | "KASPERI KAPANEN"          |
| homePlayer3_id     | Integer   | ID of the third home player.                                               | 8477953                     |
| homePlayer4        | String    | Fourth home player on the ice.                                             | "MORGAN RIELLY"            |
| homePlayer4_id     | Integer   | ID of the fourth home player.                                              | 8476853                     |
| homePlayer5        | String    | Fifth home player on the ice.                                              | "CODY CECI"                |
| homePlayer5_id     | Integer   | ID of the fifth home player.                                               | 8476879                     |
| homePlayer6        | String    | Sixth home player on the ice (often the goalie).                            | "FREDERIK ANDERSEN"        |
| homePlayer6_id     | Integer   | ID of the sixth home player.                                               | 8475883                     |
| Away_Players       | Integer   | Number of away players on the ice.                                         | 6                           |
| Home_Players       | Integer   | Number of home players on the ice.                                         | 6                           |
| Away_Score         | Integer   | Running away team score at this event.                                     | 1                           |
| Home_Score         | Integer   | Running home team score at this event.                                     | 0                           |
| Away_Goalie        | String    | Name of the away goalie in net at that time.                               | "CRAIG ANDERSON"           |
| Away_Goalie_Id     | Integer   | ID of the away goalie.                                                     | 8467950                     |
| Home_Goalie        | String    | Name of the home goalie in net at that time.                               | "FREDERIK ANDERSEN"        |
| Home_Goalie_Id     | Integer   | ID of the home goalie.                                                     | 8475883                     |
| xC                 | Integer   | X-coordinate for event location.                                           | 85                          |
| yC                 | Integer   | Y-coordinate for event location.                                           | -1                          |
| Home_Coach         | String    | Name of the home team’s head coach.                                        | "MIKE BABCOCK"             |
| Away_Coach         | String    | Name of the away team’s head coach.                                        | "D.J. SMITH"               |
    """
)