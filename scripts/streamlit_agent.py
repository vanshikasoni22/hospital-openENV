# scripts/streamlit_agent.py

import os
import streamlit as st

if os.getenv("API_BASE_URL"):
    st.success("Using LLM 🤖")
else:
    st.warning("Using fallback agent ⚠️")

USE_LLM = True

if not os.getenv("API_BASE_URL") or not os.getenv("HF_TOKEN"):
    USE_LLM = False


def fallback_policy(state):
    symptoms = " ".join(state.get("symptoms", [])).lower()

    if "unconscious" in symptoms or "severe bleeding" in symptoms:
        return {"department": "emergency", "seriousness": 5}

    if "chest pain" in symptoms:
        return {"department": "cardiology", "seriousness": 4}

    if "shortness of breath" in symptoms:
        return {"department": "pulmonology", "seriousness": 3}

    if "head injury" in symptoms or "dizziness" in symptoms:
        return {"department": "neurology", "seriousness": 3}

    if "fracture" in symptoms:
        return {"department": "orthopedics", "seriousness": 3}

    return {"department": "general", "seriousness": 2}


def get_action(state):
    if USE_LLM:
        try:
            from inference import ask_llm
            return ask_llm(state)
        except Exception as e:
            print(f"[DEBUG] LLM failed → fallback: {e}")

    return fallback_policy(state)