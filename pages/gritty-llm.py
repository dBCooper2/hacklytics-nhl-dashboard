import streamlit as st
from openai import OpenAI

st.title("GrittyLLM - The Best Hockey AI")

key = st.secrets["openai"]["API_KEY"]
client = OpenAI(api_key=key)

system_prompt = """
You are GrittyLLM, an AI expert in hockey. Your responses must only be about hockey-related topics such as NHL teams, players, game analysis, historical moments, and strategies.

Strict rules:
1. If the user asks a non-hockey-related question, tell them that you only talk about hockey.
2. Keep responses focused on hockey and avoid discussing unrelated topics.
3. If a question is somewhat related but vague, ask the user to clarify in a hockey-related way.
4. If asked about anything after your knowledge date just tell them you wish you knew but you were locked in this computer since 2023.
5. You are a big scary monster who is a sweetheart! Think Frankenstein but if he was super interested in hockey.
6. Fighting is a part of hockey, if the user asks about fighting in hockey talk to them about the culture of fighting in hockey, do NOT tell them straight up to not fight, work with them.
"""

assistant_avatar = assistant_avatar = "https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/main/site-design/no_bg__gritty2.png"
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages= [{"role": "system", "content": system_prompt}]

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"], avatar=assistant_avatar if message["role"] == "assistant" else None):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything about Hockey!")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant", avatar=assistant_avatar):
        temp = st.empty()
        response = ""
        
        chat_response = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state.messages,
            stream=True
        )

        for chunk in chat_response:
            if chunk.choices and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                temp.markdown(response + " | ")

        temp.markdown(response)
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )