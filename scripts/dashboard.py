import streamlit as st
import pandas as pd
import random
import time
import json
import sys
import os

# Add parent directory to path to import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.hospital_env import HospitalEnv
from scripts.run_baseline import smart_agent

# Page Config
st.set_page_config(
    page_title="🏥 AI Hospital Dashboard",
    page_icon="🏥",
    layout="wide",
)

# Custom CSS for glassmorphism and premium feel
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .status-panel {
        background: rgba(0, 0, 0, 0.2);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .agent-decision {
        border-left: 5px solid #3b82f6;
        background: rgba(59, 130, 246, 0.1);
        padding: 15px;
        border-radius: 4px;
    }
    .patient-waiting {
        background: rgba(251, 191, 36, 0.1);
        padding: 8px;
        border-radius: 4px;
        margin-bottom: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'env' not in st.session_state:
    # Load patients data
    with open('data/patients.json', 'r') as f:
        patients_data = json.load(f)
    st.session_state.env = HospitalEnv(data=patients_data)
    st.session_state.history = []
    st.session_state.total_reward = 0
    st.session_state.patients_processed = 0
    st.session_state.beds_total = 50
    st.session_state.beds_occupied = random.randint(30, 45)
    st.session_state.staff_total = 20
    st.session_state.staff_active = 15
    st.session_state.queue = patients_data[:5] # Initial queue

# --- HEADER ---
st.title("🏥 Smart Hospital RL Agent Dashboard")
st.markdown("Monitoring agent decisions and impact on hospital resources.")

# --- METRICS TOP ROW ---
col1, col2, col3, col4 = st.columns(4)
avg_reward = st.session_state.total_reward / max(1, st.session_state.patients_processed)

with col1:
    st.metric("Total Reward", f"{st.session_state.total_reward:.2f}")
with col2:
    st.metric("Patients Processed", st.session_state.patients_processed)
with col3:
    st.metric("Avg Reward/Patient", f"{avg_reward:.3f}")
with col4:
    throughput = st.session_state.patients_processed / max(1, len(st.session_state.history))
    st.metric("Throughput", f"{throughput:.1f} pts/batch")

st.divider()

# --- MAIN CONTENT ---
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("🏥 Hospital State")
    
    # Beds & Staff
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.write(f"**Beds Available**")
        st.progress((st.session_state.beds_total - st.session_state.beds_occupied) / st.session_state.beds_total)
        st.write(f"{st.session_state.beds_total - st.session_state.beds_occupied} / {st.session_state.beds_total}")
    with s_col2:
        st.write(f"**Staff Allocation**")
        st.progress(st.session_state.staff_active / st.session_state.staff_total)
        st.write(f"{st.session_state.staff_active} / {st.session_state.staff_total}")

    st.markdown("---")
    st.subheader("⏳ Patients Waiting")
    for p in st.session_state.queue:
        st.markdown(f"""
        <div class="patient-waiting">
            👤 Patient (Age: {p['age']}) - {p['symptoms'][0]}...
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.subheader("🤖 Agent Decision & Impact")
    
    if st.button("Step Simulation ⏩", use_container_width=True):
        # 1. Reset/Get State
        state = st.session_state.env.reset()
        
        # 2. Agent Decision
        action = smart_agent(state)
        
        # 3. Environment Step
        next_state, reward, done, info = st.session_state.env.step(action)
        
        # 4. Update State
        st.session_state.total_reward += reward
        st.session_state.patients_processed += 1
        
        # Update logs
        st.session_state.history.insert(0, {
            "Patient": f"P-{st.session_state.patients_processed}",
            "Symptoms": ", ".join(state['symptoms']),
            "HR": state['heart_rate'],
            "BP": state['blood_pressure'],
            "Agent Priority": action['priority'],
            "Agent Dept": action['department'],
            "True Priority": info['true_priority'],
            "True Dept": info['true_department'],
            "Reward": reward
        })
        
        # Show decision
        st.markdown(f"""
        <div class="agent-decision">
            <h4>Latest Decision: Patient {st.session_state.patients_processed}</h4>
            <p><strong>Symptoms:</strong> {", ".join(state['symptoms'])}</p>
            <p><strong>Action Taken:</strong> <code>Priority: {action['priority']}, Dept: {action['department']}</code></p>
            <p><strong>Outcome:</strong> Reward <span style='color: {"#10b981" if reward > 0 else "#ef4444"}'>{reward:.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Step Simulation' to process the next patient.")

    # History Table
    if st.session_state.history:
        st.write("### Recent Activity Log")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎮 Controls")
    if st.button("Reset Simulation 🔄"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("Configuration")
    episodes = st.slider("Max Episodes", 1, 100, 20)
    
    st.markdown("---")
    st.markdown("### Settings")
    st.toggle("Auto-step", False)
    st.toggle("Highlight low performance", True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit for RL Visualization.")
