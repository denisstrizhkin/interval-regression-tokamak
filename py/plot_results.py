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
        data = pd.read_csv('report/traces/plot_data.csv', comment='#', header=None)
        points = data[data[0].apply(lambda x: str(x).isdigit())].copy()
        # Cols: shot, time_b, time_a, bt_ip, r_point, low, high, ext_low, ext_high, max_ji, is_cons, is_cons_ext
        for col in [3, 4, 5, 6, 7, 8]: points[col] = pd.to_numeric(points[col])
        
        # Fit OLS regression matching doc_ref-20 style
        z = np.polyfit(points[3], points[4], 1)
        p = np.poly1d(z)
        xp = np.linspace(0.0016, 0.0042, 100)
        
        # Fig 3.4 (Internal)
        plt.figure(figsize=(10, 6))
        plt.plot(xp, p(xp), color='gray', lw=1.5, alpha=0.8) # Regression line
        consistent = points[points[10] == 1]
        others = points[points[10] == 0]
        plt.errorbar(consistent[3], consistent[4], yerr=[consistent[4]-consistent[5], consistent[6]-consistent[4]],
                     fmt='none', ecolor='red', elinewidth=0.8, capsize=1.5)
        plt.scatter(consistent[3], consistent[4], color='red', s=8, marker='o')
        plt.errorbar(others[3], others[4], yerr=[others[4]-others[5], others[6]-others[4]],
                     fmt='none', ecolor='blue', elinewidth=0.8, capsize=1.5)
        plt.scatter(others[3], others[4], color='blue', s=8, marker='o')
        plt.title("R_inv, Bt/Ip, ji=0.5")
        plt.xlabel("Bt/Ip")
        plt.ylabel("R_inv")
        plt.xlim(0.0016, 0.0042)
        plt.ylim(40, 60)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_4_internal.png', dpi=300)
        plt.close()

        # Fig 3.5 (External)
        plt.figure(figsize=(10, 6))
        plt.plot(xp, p(xp), color='gray', lw=1.5, alpha=0.8) # Regression line
        consistent_ext = points[points[11] == 1]
        others_ext = points[points[11] == 0]
        plt.errorbar(consistent_ext[3], consistent_ext[4], yerr=[consistent_ext[4]-consistent_ext[7], consistent_ext[8]-consistent_ext[4]],
                     fmt='none', ecolor='red', elinewidth=0.8, capsize=1.5)
        plt.scatter(consistent_ext[3], consistent_ext[4], color='red', s=8, marker='o')
        plt.errorbar(others_ext[3], others_ext[4], yerr=[others_ext[4]-others_ext[7], others_ext[8]-others_ext[4]],
                     fmt='none', ecolor='blue', elinewidth=0.8, capsize=1.5)
        plt.scatter(others_ext[3], others_ext[4], color='blue', s=8, marker='o')
        plt.title("R_inv, Bt/Ip, ji>0")
        plt.xlabel("Bt/Ip")
        plt.ylabel("R_inv")
        plt.xlim(0.0016, 0.0042)
        plt.ylim(40, 60)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_5_external.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error Fig 3.4/3.5: {e}")

def plot_fig_3_6():
    """Figure 3.9 style: Joint Corridor with multiple regression choices"""
    try:
        hist = pd.read_csv('report/traces/histogram_data.csv', comment='#', header=None,
                          names=['bt_ip', 'mean', 'low', 'high', 'ext_low', 'ext_high', 'count'])
        
        mass_slope, mass_int = 0, 0
        diag_slope, diag_int = 0, 0
        
        with open('report/traces/joint_corridor_forecast.csv', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Mass Center Line" in line:
                    parts = line.split(',')
                    mass_slope = float(parts[0].split('=')[1])
                    mass_int = float(parts[1].split('=')[1])
                elif "Diagonal Center Line" in line:
                    parts = line.split(',')
                    diag_slope = float(parts[0].split('=')[1])
                    diag_int = float(parts[1].split('=')[1])

        corridor_data = [[float(p) for p in l.split(',')] for l in lines if l and l[0].isdigit()]
        corridor = pd.DataFrame(corridor_data)
        
        plt.figure(figsize=(10, 6))
        
        # 1. Admissible Region (blue corridor)
        plt.fill_between(corridor[0], corridor[2], corridor[3], color='#a29bfe', alpha=0.6, label='corridor')
        
        # 2. Experimental data points (Blue circles with error bars, as per doc_ref)
        # Using ext_low/ext_high for blue dots to show the broad uncertainty
        plt.errorbar(hist['bt_ip'], hist['mean'], yerr=[hist['mean']-hist['ext_low'], hist['ext_high']-hist['mean']],
                     fmt='none', ecolor='blue', elinewidth=0.8, capsize=2, alpha=0.6)
        plt.scatter(hist['bt_ip'], hist['mean'], color='blue', s=8, label='R_inv', marker='o')

        # 3. Forecast lines
        x_full = np.linspace(0.0016, 0.010, 200)
        plt.plot(x_full, mass_slope * x_full + mass_int, color='black', lw=1.5, label='mass center regression')
        plt.plot(x_full, diag_slope * x_full + diag_int, color='#27ae60', lw=1.2, label='diagonal center regression')

        # 4. Prediction points (Red markers at the end of the range)
        forecast_pts = corridor[corridor[0] > 0.0035].iloc[::40] # Sparse points for visibility
        plt.errorbar(forecast_pts[0], forecast_pts[1], yerr=[forecast_pts[1]-forecast_pts[2], forecast_pts[3]-forecast_pts[1]],
                     fmt='none', ecolor='red', elinewidth=0.8, capsize=2)
        plt.scatter(forecast_pts[0], forecast_pts[1], color='red', s=10, marker='x', label='prediction ji>0')

        plt.title("R_inv, Bt/Ip, ji>0")
        plt.xlabel("Bt/Ip")
        plt.ylabel("R_inv")
        plt.xlim(0.0015, 0.0105)
        plt.ylim(40, 60)
        plt.legend(frameon=True, loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('report/images/fig_3_6_corridor.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error Fig 3.6/3.9: {e}")

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
