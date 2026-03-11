import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Configuration ---
st.set_page_config(page_title="CPPopt Analyzer", layout="wide")

# --- Helper Functions ---

def parse_timestamps(timestamp_series):
    """
    Parses timestamps in various formats (MM:SS, HH:MM:SS, Date Time)
    and unwraps them to handle hour/day rollovers.
    """
    def to_seconds(t_str):
        try:
            if pd.isna(t_str): return np.nan
            t_str = str(t_str).strip()
            
            # If there's a date and time (e.g. "2023-10-24 14:30:00"), just take the time part
            if ' ' in t_str:
                t_str = t_str.split(' ')[-1]
                
            parts = t_str.split(':')
            
            # HH:MM:SS
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            # MM:SS
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            # Already in seconds?
            elif len(parts) == 1:
                return float(parts[0])
            return np.nan
        except:
            return np.nan

    seconds_raw = timestamp_series.apply(to_seconds)
    
    # Unwrap (detect rollover like going from 23:59:59 to 00:00:00)
    diffs = seconds_raw.diff()
    # If time jumps backwards by a large amount (e.g., > 1000s), assume a rollover
    wraps = (diffs < -1000).cumsum().fillna(0)
    
    # We add 3600 seconds if the data is MM:SS and wrapped the hour. 
    # If it's HH:MM:SS, it wraps the day (86400 seconds).
    # We'll dynamically guess the wrap size based on the max value before wrap.
    wrap_size = 3600 if seconds_raw.max() <= 3605 else 86400
    
    seconds_unwrapped = seconds_raw + (wraps * wrap_size)
    
    return seconds_unwrapped

def format_duration(seconds):
    """Converts seconds to 'X hr Y min' format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours} hr {minutes} min"

def calculate_cppopt(df_prx, df_cpp, bin_size=5, min_bin_count=10):
    df_prx = df_prx.sort_values('time_sec')
    df_cpp = df_cpp.sort_values('time_sec')
    
    f_cpp = interp1d(df_cpp['time_sec'], df_cpp['CPP'], kind='linear', fill_value="extrapolate")
    df_prx['aligned_cpp'] = f_cpp(df_prx['time_sec'])

    min_c = int(df_prx['aligned_cpp'].min())
    max_c = int(df_prx['aligned_cpp'].max())
    
    if max_c <= min_c:
         return pd.DataFrame(), np.nan, np.nan, np.nan

    bins = np.arange(min_c, max_c + bin_size, bin_size)
    df_prx['cpp_bin'] = pd.cut(df_prx['aligned_cpp'], bins=bins)

    global_avg_prx = df_prx['prx'].mean()

    stats = df_prx.groupby('cpp_bin', observed=True)['prx'].agg(['mean', 'count', 'sem']).reset_index()
    stats['cpp_mid'] = stats['cpp_bin'].apply(lambda x: x.mid).astype(float)
    
    valid_stats = stats[stats['count'] >= min_bin_count].copy()
    
    if not valid_stats.empty:
        min_idx = valid_stats['mean'].idxmin()
        cpp_opt = valid_stats.loc[min_idx, 'cpp_mid']
        min_prx_val = valid_stats.loc[min_idx, 'mean']
    else:
        cpp_opt = np.nan
        min_prx_val = np.nan
        
    return valid_stats, cpp_opt, min_prx_val, global_avg_prx

# --- Main App Interface ---

st.title("🧠 CPPopt Analyzer")
st.markdown("Upload your separate **Prx** and **CPP** CSV files below.")

col1, col2 = st.columns(2)
with col1:
    prx_file = st.file_uploader("Upload Prx CSV", type=['csv'])
with col2:
    cpp_file = st.file_uploader("Upload CPP CSV", type=['csv'])

if prx_file and cpp_file:
    st.divider()
    try:
        df_prx_raw = pd.read_csv(prx_file)
        df_cpp_raw = pd.read_csv(cpp_file)
        
        with st.spinner('Processing data...'):
            prx_cols = [c for c in df_prx_raw.columns if 'prx' in c.lower()]
            prx_time_cols = [c for c in df_prx_raw.columns if 'time' in c.lower()]
            cpp_cols = [c for c in df_cpp_raw.columns if 'cpp' in c.lower()]
            cpp_time_cols = [c for c in df_cpp_raw.columns if 'time' in c.lower()]

            if not (prx_cols and prx_time_cols and cpp_cols and cpp_time_cols):
                 st.error("Could not automatically detect columns. Ensure headers like 'timestamp' and 'prx'/'CPP' exist.")
                 st.stop()

            df_prx = df_prx_raw.rename(columns={prx_time_cols[0]: 'timestamp', prx_cols[0]: 'prx'})
            df_cpp = df_cpp_raw.rename(columns={cpp_time_cols[0]: 'timestamp', cpp_cols[0]: 'CPP'})
            
            # Show the user what the raw timestamps look like to help debug
            with st.expander("Show Timestamp Debugging Info"):
                st.write("First 3 Prx Timestamps:", df_prx['timestamp'].head(3).tolist())
                st.write("First 3 CPP Timestamps:", df_cpp['timestamp'].head(3).tolist())

            df_prx['time_sec'] = parse_timestamps(df_prx['timestamp'])
            df_cpp['time_sec'] = parse_timestamps(df_cpp['timestamp'])
            
            df_prx = df_prx.dropna(subset=['time_sec', 'prx'])
            df_cpp = df_cpp.dropna(subset=['time_sec', 'CPP'])
            
            # --- SAFETY CHECK ---
            if len(df_prx) == 0 or len(df_cpp) == 0:
                st.error("🚨 Error: The app couldn't understand the timestamp format, resulting in 0 data points.")
                st.info("Check the 'Timestamp Debugging Info' above. What do the raw timestamps look like? (e.g. 14:30:00, or something else?)")
                st.stop()

            stats_df, cpp_opt, min_prx, global_avg_prx = calculate_cppopt(df_prx, df_cpp)
            
            total_duration_sec = df_prx['time_sec'].max() - df_prx['time_sec'].min()
            
        # Row 1: Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal CPP", f"{cpp_opt} mmHg" if not pd.isna(cpp_opt) else "N/A")
        m2.metric("Min Prx (at Opt)", f"{min_prx:.3f}" if not pd.isna(min_prx) else "N/A")
        m3.metric("Avg Prx (Total)", f"{global_avg_prx:.3f}" if not pd.isna(global_avg_prx) else "N/A")
        m4.metric("Duration", format_duration(total_duration_sec))
        
        # Plotting
        st.subheader("Pressure Reactivity Curve")
        
        if not stats_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.errorbar(stats_df['cpp_mid'], stats_df['mean'], yerr=stats_df['sem'], 
                        fmt='o-', capsize=4, color='#1f77b4', label='Prx ± SEM')
            
            if not pd.isna(cpp_opt):
                ax.plot(cpp_opt, min_prx, 'rx', markersize=12, markeredgewidth=3, label=f'Optimal CPP: {cpp_opt}')
            
            ax.axhline(global_avg_prx, color='orange', linestyle=':', linewidth=1.5, label=f'Avg Prx ({global_avg_prx:.2f})')
            ax.axhline(0.2, color='green', linestyle='--', alpha=0.5, label='Threshold (0.2)')
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
            
            ax.set_xlabel("Cerebral Perfusion Pressure (CPP) [mmHg]")
            ax.set_ylabel("Prx")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
            st.pyplot(fig)
            
            csv = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Analysis Data", data=csv, file_name="cppopt_analysis.csv", mime="text/csv")
            
        else:
            st.warning("Not enough overlapping data to generate a curve.")
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
