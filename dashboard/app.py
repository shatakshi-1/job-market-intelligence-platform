import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "plotly"], capture_output=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from collections import Counter

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Job Market Intelligence Platform",
    page_icon="💼",
    layout="wide"
)

import os
PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
# ══════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    analyst  = pd.read_csv(PATH + "data/processed/analyst_clean.csv")
    salaries = pd.read_csv(PATH + "data/processed/salaries_clean.csv")
    naukri   = pd.read_csv(PATH + "data/processed/naukri_clean.csv")
    return analyst, salaries, naukri

@st.cache_resource
def load_model():
    with open(PATH + "models/salary_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(PATH + "models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(PATH + "models/features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, encoders, features

analyst, salaries, naukri = load_data()
model, encoders, features = load_model()

# ══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════

st.sidebar.title("💼 Job Market Intel")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "💰 Salary Explorer",
    "🛠️ Skills Analysis",
    "🌍 India vs Global",
    "🤖 Salary Predictor"
])

st.sidebar.markdown("---")
st.sidebar.caption("Data: Kaggle — DataAnalyst Jobs, DS Salaries, Naukri")

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════

if page == "📊 Overview":

    st.title("📊 Job Market Intelligence Platform")
    st.markdown("Real-time insights from **3 datasets** covering US & India job markets.")
    st.markdown("---")

    # Metric cards
   
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs",       f"{len(analyst) + len(naukri):,}")
    col2.metric("Avg Salary (USD)", f"${salaries['salary_avg'].mean():,.0f}")
    col3.metric("Countries",        f"{salaries['company_location'].nunique()}")
    col4.metric("Job Titles",       f"{salaries['job_title'].nunique()}")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 10 Job Titles (US)")
        top_titles = analyst['job_title'].value_counts().head(10).reset_index()
        top_titles.columns = ['Job Title', 'Count']
        fig = px.bar(top_titles, x='Count', y='Job Title',
                     orientation='h', color='Count',
                     color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=400,
                          yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Work Mode Distribution (Global)")
        wm = salaries['work_mode'].value_counts().reset_index()
        wm.columns = ['Work Mode', 'Count']
        fig = px.pie(wm, names='Work Mode', values='Count',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Hiring Locations (US)")
    loc = analyst['location'].value_counts().head(10).reset_index()
    loc.columns = ['Location', 'Count']
    fig = px.bar(loc, x='Location', y='Count',
                 color='Count', color_continuous_scale='Teal')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — SALARY EXPLORER
# ══════════════════════════════════════════════════════════

elif page == "💰 Salary Explorer":

    st.title("💰 Salary Explorer")
    st.markdown("Explore how salary varies by experience, role, and company size.")
    st.markdown("---")

    exp_options = ['All'] + sorted(salaries['experience_level'].dropna().unique().tolist())
    selected_exp = st.selectbox("Filter by Experience Level", exp_options)

    filtered_salaries = salaries if selected_exp == 'All' \
                        else salaries[salaries['experience_level'] == selected_exp]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Salary Distribution")
        fig = px.histogram(filtered_salaries, x='salary_avg', nbins=40,
                           color_discrete_sequence=['#2196F3'],
                           labels={'salary_avg': 'Annual Salary (USD)'})
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Salary by Experience Level")
        exp_map = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
        salaries['exp_label'] = salaries['experience_level'].map(exp_map).fillna(salaries['experience_level'])
        fig = px.box(salaries, x='exp_label', y='salary_avg',
                 color='exp_label',
                 labels={'salary_avg': 'Salary (USD)',
                         'exp_label': 'Experience'},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 Highest Paying Job Titles")
    top_pay = (salaries.groupby('job_title')['salary_avg']
                       .mean()
                       .sort_values(ascending=False)
                       .head(15)
                       .reset_index())
    top_pay.columns = ['Job Title', 'Avg Salary (USD)']
    top_pay['Avg Salary (USD)'] = top_pay['Avg Salary (USD)'].round(0)
    fig = px.bar(top_pay, x='Avg Salary (USD)', y='Job Title',
                 orientation='h', color='Avg Salary (USD)',
                 color_continuous_scale='Viridis',
                 text='Avg Salary (USD)')
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Salary by Company Size")
    salaries['company_size_label'] = salaries['company_size'].map(
        {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    )
    size_order = ['Small', 'Medium', 'Large']
    fig = px.violin(salaries, x='company_size_label', y='salary_avg',
                    color='company_size_label', box=True,
                    labels={'salary_avg': 'Salary (USD)',
                            'company_size_label': 'Company Size'},
                    color_discrete_sequence=['#42A5F5', '#66BB6A', '#FFA726'])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — SKILLS ANALYSIS
# ══════════════════════════════════════════════════════════

elif page == "🛠️ Skills Analysis":

    st.title("🛠️ Skills Analysis")
    st.markdown("What skills do employers actually want?")
    st.markdown("---")

    skill_series = naukri['skills'].dropna().str.lower().str.split(',')
    all_skills = Counter(
        s.strip() for sub in skill_series
        for s in sub if s.strip() not in ['not listed', '']
    )

    top_n = st.slider("Show top N skills", 5, 30, 15)
    skill_df = pd.DataFrame(all_skills.most_common(top_n),
                            columns=['Skill', 'Count'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top {top_n} Skills — India (Naukri)")
        fig = px.bar(skill_df, x='Count', y='Skill',
                     orientation='h', color='Count',
                     color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Skills — Radar Chart")
        radar_df = skill_df.head(10)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_df['Count'].tolist() + [radar_df['Count'].iloc[0]],
            theta=radar_df['Skill'].tolist() + [radar_df['Skill'].iloc[0]],
            fill='toself',
            name='Skill demand',
            line_color='#2196F3'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Skill Gap Checker")
    st.markdown("Enter your skills to see what you're missing from the top required skills.")

    user_input = st.text_input(
        "Your skills (comma separated)",
        placeholder="python, sql, excel, power bi"
    )

    if user_input:
        user_skills   = set(s.strip().lower() for s in user_input.split(','))
        top_required  = set(skill_df.head(15)['Skill'].tolist())
        matched       = top_required & user_skills
        missing       = top_required - user_skills

        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"✅ You have {len(matched)} of top 15 skills")
            for s in sorted(matched):
                st.write(f"✅ {s}")
        with col_b:
            st.warning(f"⚠️ {len(missing)} skills to learn")
            for s in sorted(missing):
                st.write(f"❌ {s}")

# ══════════════════════════════════════════════════════════
# PAGE 4 — INDIA VS GLOBAL
# ══════════════════════════════════════════════════════════

elif page == "🌍 India vs Global":

    st.title("🌍 India vs Global Job Market")
    st.markdown("Comparing job trends between India (Naukri) and the global market.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇮🇳 Top 10 Job Titles — India")
        india_titles = naukri['job_title'].value_counts().head(10).reset_index()
        india_titles.columns = ['Job Title', 'Count']
        fig = px.bar(india_titles, x='Count', y='Job Title',
                     orientation='h',
                     color_discrete_sequence=['#FF9800'])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌐 Top 10 Job Titles — Global")
        global_titles = salaries['job_title'].value_counts().head(10).reset_index()
        global_titles.columns = ['Job Title', 'Count']
        fig = px.bar(global_titles, x='Count', y='Job Title',
                     orientation='h',
                     color_discrete_sequence=['#2196F3'])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🌐 Global Salary by Experience")
        order = ['Entry', 'Mid', 'Senior', 'Executive']
        exp_salary = (salaries.groupby('experience_level')['salary_avg']
                              .mean()
                              .reindex(order)
                              .reset_index())
        exp_salary.columns = ['Experience', 'Avg Salary (USD)']
        fig = px.bar(exp_salary, x='Experience', y='Avg Salary (USD)',
                     color='Experience',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("🇮🇳 Top Hiring Cities — India")
        india_cities = naukri['location'].value_counts().head(8).reset_index()
        india_cities.columns = ['City', 'Jobs']
        fig = px.pie(india_cities, names='City', values='Jobs',
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     hole=0.35)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🌐 Global Salary by Country (Top 15)")
    country_salary = (salaries.groupby('company_location')['salary_avg']
                              .mean()
                              .sort_values(ascending=False)
                              .head(15)
                              .reset_index())
    country_salary.columns = ['Country', 'Avg Salary (USD)']
    fig = px.bar(country_salary, x='Country', y='Avg Salary (USD)',
                 color='Avg Salary (USD)',
                 color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — SALARY PREDICTOR
# ══════════════════════════════════════════════════════════

elif page == "🤖 Salary Predictor":

    st.title("🤖 AI Salary Predictor")
    st.markdown("Enter your profile and get a predicted salary based on real market data.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox(
            "Job Title",
            sorted(encoders['job_title'].classes_.tolist())
        )
        experience_level = st.selectbox(
            "Experience Level",
            ['Entry', 'Mid', 'Senior', 'Executive']
        )
        employment_type = st.selectbox(
            "Employment Type",
            ['Full-Time', 'Part-Time', 'Contract', 'Freelance']
        )

    with col2:
        company_size = st.selectbox(
            "Company Size",
            ['S (Small)', 'M (Medium)', 'L (Large)']
        )
        company_location = st.selectbox(
            "Company Location",
            sorted(encoders['company_location'].classes_.tolist())
        )
        work_mode = st.selectbox(
            "Work Mode",
            ['On-site (0%)', 'Hybrid (50%)', 'Remote (100%)']
        )

    st.markdown("---")

    if st.button("🔮 Predict My Salary", type="primary"):

        size_map   = {'S (Small)': 'S', 'M (Medium)': 'M', 'L (Large)': 'L'}
        remote_map = {'On-site (0%)': 0, 'Hybrid (50%)': 50, 'Remote (100%)': 100}

        raw = {
            'job_title'        : job_title,
            'experience_level' : experience_level,
            'employment_type'  : employment_type,
            'company_size'     : size_map[company_size],
            'company_location' : company_location,
            'remote_ratio'     : remote_map[work_mode]
        }

        encoded = []
        for col in features:
            if col in encoders:
                val = raw[col]
                if val in encoders[col].classes_:
                    encoded.append(encoders[col].transform([val])[0])
                else:
                    encoded.append(0)
            else:
                encoded.append(raw[col])

        input_df  = pd.DataFrame([encoded], columns=features)
        predicted = model.predict(input_df)[0]

        st.success(f"### 💰 Predicted Salary: ${predicted:,.0f} USD / year")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Annual",  f"${predicted:,.0f}")
        col_b.metric("Monthly", f"${predicted / 12:,.0f}")
        col_c.metric("Weekly",  f"${predicted / 52:,.0f}")

        st.markdown("---")

        avg      = salaries['salary_avg'].mean()
        diff     = predicted - avg
        diff_pct = (diff / avg) * 100

        if diff > 0:
            st.info(f"📈 This is ${diff:,.0f} ({diff_pct:.1f}%) above the global average of ${avg:,.0f}")
        else:
            st.warning(f"📉 This is ${abs(diff):,.0f} ({abs(diff_pct):.1f}%) below the global average of ${avg:,.0f}")



