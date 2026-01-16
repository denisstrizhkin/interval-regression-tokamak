import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# Create images directory if it doesn't exist
os.makedirs('report/images', exist_ok=True)

# Common styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

def plot_fig_3_2():
    """Figure 3.2: Temperature profiles with corridors"""
    try:
        profiles = pd.read_csv('report/traces/profiles.csv', comment='#', header=None,
                              names=['type', 'shot', 'time', 'radius', 'temp', 'low', 'high'])
        # Reference uses shot 42154
        shot_id = 42154 if 42154 in profiles['shot'].values else profiles['shot'].iloc[0]
        p_before = profiles[(profiles['type'] == 'before') & (profiles['shot'] == shot_id)]
        p_after = profiles[(profiles['type'] == 'after') & (profiles['shot'] == shot_id)]
        c_before = profiles[(profiles['type'] == 'corridor_before') & (profiles['shot'] == shot_id)]
        c_after = profiles[(profiles['type'] == 'corridor_after') & (profiles['shot'] == shot_id)]
        
        plt.figure(figsize=(10, 6))
        # Use high quality markers and transparency to match doc_ref-18
        if not c_before.empty:
            plt.fill_between(c_before['radius'], c_before['low'], c_before['high'], color='#2ecc71', alpha=0.3)
            plt.plot(c_before['radius'], (c_before['low']+c_before['high'])/2, 'g^-', label=f'value_t1_{shot_id}', markersize=3, lw=0.8)
        if not c_after.empty:
            plt.fill_between(c_after['radius'], c_after['low'], c_after['high'], color='#3498db', alpha=0.3)
            plt.plot(c_after['radius'], (c_after['low']+c_after['high'])/2, 'bs--', label=f'value_t2_{shot_id}', markersize=3, lw=0.8)
            
        plt.title(f"Te, R shot {shot_id}")
        plt.xlabel("R")
        plt.ylabel("Te")
        plt.xlim(40, 60)
        plt.ylim(0.2, 2.7)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_2_profiles.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error Fig 3.2: {e}")

def plot_fig_3_3():
    """Figure 3.3: Jaccard Index"""
    try:
        jaccard = pd.read_csv('report/traces/jaccard_curves.csv', comment='#', header=None,
                             names=['shot', 'time', 'radius', 'ji'])
        shot_id = 42154 if 42154 in jaccard['shot'].values else jaccard['shot'].iloc[0]
        subset = jaccard[jaccard['shot'] == shot_id]
        
        plt.figure(figsize=(10, 6))
        plt.plot(subset['radius'], subset['ji'], color='#3498db', lw=1.5, label='Ji')
        
        # In doc_ref-19, threshold lines are segments. 
        # For Ji=0.5, only show where Ji is high
        peak_range = subset[subset['ji'] > 0.3]['radius']
        if not peak_range.empty:
            plt.hlines(0.5, peak_range.min(), peak_range.max(), color='black', lw=1.2, label='Внутренний Ji=0.5')
        plt.axhline(0, color='red', lw=1, label='Внешний Ji>0')
        
        plt.title("Индекс Жаккара")
        plt.xlabel("R")
        plt.ylabel("Ji")
        plt.xlim(40, 60)
        plt.ylim(-0.05, 1.1)
        plt.legend(loc='upper right', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_3_jaccard.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error Fig 3.3: {e}")

def plot_fig_3_4_3_5():
    """Figure 3.4 & 3.5: Compatibility Analysis"""
    try:
        # Col 0: shot, 1: time_b, 2: time_a, 3: bt_ip, 4: r_point, 5: low, 6: high, 7: ext_low, 8: ext_high, 9: max_ji, 10: cons, 11: cons_ext
        data = pd.read_csv('report/traces/plot_data.csv', comment='#', header=None)
        points = data.copy()
        for i in range(len(points.columns)):
            points[i] = pd.to_numeric(points[i], errors='coerce')
        points = points.dropna(subset=[3, 5, 6, 7, 8])
        
        x = points[3] # bt_ip_ratio
        
        # --- Fig 3.4 (Internal) ---
        y_low = points[5]
        y_high = points[6]
        y_mid = (y_low + y_high) / 2.0
        
        # Fit OLS through internal midpoints
        z = np.polyfit(x, y_mid, 1)
        p = np.poly1d(z)
        xp = np.linspace(0.0016, 0.0042, 100)
        
        # Intersection logic: red if line passes through [low, high]
        line_vals = p(x)
        is_red = (y_low <= line_vals) & (line_vals <= y_high)
        
        plt.figure(figsize=(10, 6))
        plt.plot(xp, p(xp), color='black', lw=1.2, alpha=0.8, label='OLS Regression')
        
        # Plot inconsistent (blue)
        blue_pts = points[~is_red]
        blue_mid = (blue_pts[5] + blue_pts[6]) / 2.0
        plt.errorbar(blue_pts[3], blue_mid, yerr=[blue_mid - blue_pts[5], blue_pts[6] - blue_mid],
                     fmt='none', ecolor='blue', elinewidth=0.8, capsize=1.5)
        plt.scatter(blue_pts[3], blue_mid, color='blue', s=8, marker='o', label=r'Inconsistent ($J_I < 0.5$)')
        
        # Plot consistent (red)
        red_pts = points[is_red]
        red_mid = (red_pts[5] + red_pts[6]) / 2.0
        plt.errorbar(red_pts[3], red_mid, yerr=[red_mid - red_pts[5], red_pts[6] - red_mid],
                     fmt='none', ecolor='red', elinewidth=1.0, capsize=1.5)
        plt.scatter(red_pts[3], red_mid, color='red', s=10, marker='o', zorder=5, label=r'Consistent ($J_I \geq 0.5$)')
        
        plt.title(r"R_inv, Bt/Ip, $J_I \geq 0.5$ (Internal Compatibility)")
        plt.xlabel("Bt/Ip")
        plt.ylabel("R_inv")
        plt.xlim(0.0016, 0.0042)
        plt.ylim(40, 60)
        plt.legend(loc='upper right', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_4_internal.png', dpi=300)
        plt.close()

        # --- Fig 3.5 (External) ---
        y_ext_low = points[7]
        y_ext_high = points[8]
        y_ext_mid = (y_ext_low + y_ext_high) / 2.0
        
        # Fit OLS through external midpoints
        ze = np.polyfit(x, y_ext_mid, 1)
        pe = np.poly1d(ze)
        
        line_vals_ext = pe(x)
        is_red_ext = (y_ext_low <= line_vals_ext) & (line_vals_ext <= y_ext_high)
        
        plt.figure(figsize=(10, 6))
        plt.plot(xp, pe(xp), color='black', lw=1.2, alpha=0.8, label='OLS Regression')
        
        # Plot inconsistent (blue)
        blue_pts_ext = points[~is_red_ext]
        blue_mid_ext = (blue_pts_ext[7] + blue_pts_ext[8]) / 2.0
        plt.errorbar(blue_pts_ext[3], blue_mid_ext, yerr=[blue_mid_ext - blue_pts_ext[7], blue_pts_ext[8] - blue_mid_ext],
                     fmt='none', ecolor='blue', elinewidth=0.8, capsize=1.5)
        plt.scatter(blue_pts_ext[3], blue_mid_ext, color='blue', s=8, marker='o', label=r'Inconsistent ($J_I = 0$)')
        
        # Plot consistent (red)
        red_pts_ext = points[is_red_ext]
        red_mid_ext = (red_pts_ext[7] + red_pts_ext[8]) / 2.0
        plt.errorbar(red_pts_ext[3], red_mid_ext, yerr=[red_mid_ext - red_pts_ext[7], red_pts_ext[8] - red_mid_ext],
                     fmt='none', ecolor='red', elinewidth=1.0, capsize=1.5)
        plt.scatter(red_pts_ext[3], red_mid_ext, color='red', s=10, marker='o', zorder=5, label=r'Consistent ($J_I > 0$)')
        
        plt.title("R_inv, Bt/Ip, ji>0 (External Compatibility)")
        plt.xlabel("Bt/Ip")
        plt.ylabel("R_inv")
        plt.xlim(0.0016, 0.0042)
        plt.ylim(40, 60)
        plt.legend(loc='upper right', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_5_external.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error Fig 3.4/3.5: {e}")

def plot_fig_3_6():
    """Figure 3.9 style: Joint Corridor with multiple corridors and individual points"""
    try:
        # Load internal corridor
        corridor_int = pd.read_csv('report/traces/joint_corridor_forecast_int.csv', comment='#', header=None)
        # Load external corridor
        corridor_ext = pd.read_csv('report/traces/joint_corridor_forecast_ext.csv', comment='#', header=None)
        
        # Load raw points
        data = pd.read_csv('report/traces/plot_data.csv', comment='#', header=None)
        # Col 0: shot, 1: time_b, 2: time_a, 3: bt_ip, 4: r_point, 5: low, 6: high, 7: ext_low, 8: ext_high
        points = data.copy()
        for i in range(len(points.columns)):
            points[i] = pd.to_numeric(points[i], errors='coerce')
        points = points.dropna(subset=[3, 5, 6, 7, 8])

        plt.figure(figsize=(10, 6))
        
        # 1. External Admissible Region (broad blue corridor)
        plt.fill_between(corridor_ext[0], corridor_ext[2], corridor_ext[3], color='#a29bfe', alpha=0.3, label='corridor (Ji > 0)')
        
        # 2. Internal Admissible Region (tighter blue corridor)
        plt.fill_between(corridor_int[0], corridor_int[2], corridor_int[3], color='#a29bfe', alpha=0.6, label='corridor (Ji >= 0.5)')
        
        # 3. Experimental data points (Blue markers with error bars)
        # We use the internal error bars [low, high] for the main points to show the core uncertainty
        plt.errorbar(points[3], (points[5]+points[6])/2.0, yerr=[(points[5]+points[6])/2.0 - points[5], points[6] - (points[5]+points[6])/2.0],
                     fmt='none', ecolor='blue', elinewidth=0.6, capsize=1.5, alpha=0.7)
        plt.scatter(points[3], (points[5]+points[6])/2.0, color='blue', s=6, label='R_inv', marker='o', alpha=0.9)

        # 4. Forecast markers at specific points (matching reference style)
        forecast_x = [0.005, 0.008, 0.010]
        
        # Prediction Ji > 0 (Red markers)
        for x in forecast_x:
            # Interpolate from corridor_ext
            row = corridor_ext.iloc[(corridor_ext[0] - x).abs().argmin()]
            plt.errorbar([row[0]], [row[1]], yerr=[[row[1]-row[2]], [row[3]-row[1]]], fmt='none', ecolor='red', elinewidth=1.0, capsize=3)
            if x == forecast_x[0]:
                plt.scatter([row[0]], [row[1]], color='red', s=12, marker='o', label=r'prediction ji>0')
            else:
                plt.scatter([row[0]], [row[1]], color='red', s=12, marker='o')

        # Prediction Ji >= 0.5 (Black markers/bars)
        for x in forecast_x:
            # Interpolate from corridor_int
            row = corridor_int.iloc[(corridor_int[0] - x).abs().argmin()]
            plt.errorbar([row[0]], [row[1]], yerr=[[row[1]-row[2]], [row[3]-row[1]]], fmt='none', ecolor='black', elinewidth=1.5, capsize=3)
            if x == forecast_x[0]:
                plt.scatter([row[0]], [row[1]], color='black', s=15, marker='_', label=r'Prediction Ji $\geq$ 0.5')
            else:
                plt.scatter([row[0]], [row[1]], color='black', s=15, marker='_')

        plt.title("R_inv, Bt/Ip, ji>0")
        plt.xlabel("Bt/Ip")
        plt.ylabel("R_inv")
        plt.xlim(0.0015, 0.0105)
        plt.ylim(40.8, 60.0) # Match reference scale
        plt.legend(frameon=True, loc='upper right', fontsize=8)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_6_corridor.png', dpi=300)
        plt.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error Fig 3.6: {e}")

def plot_fig_3_7_informational_set():
    """Figure 3.8 style: Informational Set in parameter space (a, b)"""
    try:
        from scipy.spatial import ConvexHull
        df = pd.read_csv('report/traces/informational_set.csv', comment='#', header=None, names=['a', 'b'])
        
        plt.figure(figsize=(10, 6))
        points = df[['a', 'b']].values
        
        if len(points) >= 3:
            hull = ConvexHull(points)
            # Shade the informational set
            plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='#add8e6', alpha=0.5, edgecolor='#4682b4', lw=1.5)
            # Plot the 'corners' (red dots as in doc_ref-21)
            plt.scatter(points[:, 0], points[:, 1], color='red', s=10, zorder=5)
        else:
            plt.scatter(df['a'], df['b'], color='red', s=10)

        plt.title("Внешнее информационное множество для R_inv")
        plt.xlabel("b1 (slope)")
        plt.ylabel("b2 (intercept)")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_7_info_set.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error Fig 3.7: {e}")

if __name__ == "__main__":
    plot_fig_3_2()
    plot_fig_3_3()
    plot_fig_3_4_3_5()
    plot_fig_3_6()
    plot_fig_3_7_informational_set()
    print("All reference plots generated in report/images/")
