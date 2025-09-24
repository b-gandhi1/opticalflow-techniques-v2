import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

pitch_df = pd.read_csv("participant_data/part_pitch_pressure.csv",usecols=['pitch','Pressure (kPa)'])
pitch, pressure_pitch = pitch_df['pitch'], pitch_df['Pressure (kPa)']
pitch_pol_imu_df = pd.read_csv("participant_data/part5/imu_pol_pitch_2.csv", usecols=['IMU X', 'Polaris Rz'])
pitch_pol, pitch_imu = pitch_pol_imu_df['IMU X'], pitch_pol_imu_df['Polaris Rz']

roll_df = pd.read_csv("participant_data/part_roll_pressure.csv",usecols=['roll','Pressure (kPa)'])
roll, pressure_roll = roll_df['roll'], roll_df['Pressure (kPa)']
roll_pol_imu_df = pd.read_csv("participant_data/part9/imu_pol_roll_3.csv",usecols=['IMU Y', 'Polaris Ry'])
roll_pol, roll_imu = roll_pol_imu_df['IMU Y'], roll_pol_imu_df['Polaris Ry']

ts = np.linspace(0,60,len(pitch))
fig_pitch, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(ts, pitch, label='MCP')
ax1.plot(ts, pitch_imu, label='Gyro')
ax1.plot(ts, pitch_pol, label='Ground Truth')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pitch (Degrees)')
ax1.legend(loc='lower left')
ax1.set_ylim(-34,26)
ax2.plot(ts, pressure_pitch)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Pressure (kPa)')
plt.tight_layout()
print(f"Pitch participant pressure, mean: {np.mean(pressure_pitch)}, std: {np.std(pressure_pitch)}")

fig_roll, (ax3, ax4) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
ax3.plot(ts, roll, label='MCP')
ax3.plot(ts, roll_imu, label='Gyro')
ax3.plot(ts, roll_pol, label='Ground Truth')
ax3.legend(loc='lower left')
ax3.set_ylim(-10,9)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Roll (Degrees)')
ax4.plot(ts, pressure_roll)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Pressure (kPa)')
plt.tight_layout()
print(f"Roll participant pressure, mean: {np.mean(pressure_roll)}, std: {np.std(pressure_roll)}")

plt.show()

print("------------------------------------------------")
# MCP roll order: part10_1, part9_3, part9_2, part9_1
# MCP pitch order: part5_2, part5_3, part9_3
part_pitch = pd.read_csv("participant_data/part-pitch_scaled_mcp.csv")
pitch_52, pitch_53, pitch_93 = part_pitch['0'], part_pitch['1'], part_pitch['2']
part_roll = pd.read_csv("participant_data/part-roll_scaled_mcp.csv")
roll_101, roll_93, roll_92, roll_91 = part_roll['0'], part_roll['1'], part_roll['2'], part_roll['3']
pitch_pressure = pd.read_csv("participant_data/partPitch_allpressures.csv")
roll_pressure = pd.read_csv("participant_data/partRoll_allpressures.csv")
pressure_pitch_52, pressure_pitch_53, pressure_pitch_93 = pitch_pressure['part5_2'], pitch_pressure['part5_3'], pitch_pressure['part9_3']
pressure_roll_101, pressure_roll_93, pressure_roll_92, pressure_roll_91 = roll_pressure['part10_1'], roll_pressure['part9_3'], roll_pressure['part9_2'], roll_pressure['part9_1']

pol_pitch_52 = pd.read_csv("participant_data/part5/imu_pol_pitch_2.csv", usecols=['Polaris Rz'])
pol_pitch_53 = pd.read_csv("participant_data/part5/imu_pol_pitch_3.csv", usecols=['Polaris Rz'])
pol_pitch_93 = pd.read_csv("participant_data/part9/imu_pol_pitch_3.csv", usecols=['Polaris Rz'])
pol_roll_101 = pd.read_csv("participant_data/part10/imu_pol_roll_1.csv", usecols=['Polaris Ry'])
pol_roll_93 = pd.read_csv("participant_data/part9/imu_pol_roll_3.csv", usecols=['Polaris Ry'])
pol_roll_92 = pd.read_csv("participant_data/part9/imu_pol_roll_2.csv", usecols=['Polaris Ry'])
pol_roll_91 = pd.read_csv("participant_data/part9/imu_pol_roll_1.csv", usecols=['Polaris Ry'])


# spearman correlations, pressure to MCP
pitch_spear_52 = spearmanr(pressure_pitch_52, pitch_52, alternative='two-sided', nan_policy='propagate')
pitch_spear_53 = spearmanr(pressure_pitch_53, pitch_53, alternative='two-sided', nan_policy='propagate')
pitch_spear_93 = spearmanr(pressure_pitch_93, pitch_93, alternative='two-sided', nan_policy='propagate')
roll_spear_101 = spearmanr(pressure_roll_101, roll_101, alternative='two-sided', nan_policy='propagate')
roll_spear_93 = spearmanr(pressure_roll_93, roll_93, alternative='two-sided', nan_policy='propagate')
roll_spear_92 = spearmanr(pressure_roll_92, roll_92, alternative='two-sided', nan_policy='propagate')
roll_spear_91 = spearmanr(pressure_roll_91, roll_91, alternative='two-sided', nan_policy='propagate')
print(f"Spearman correlation pitch 5_2: {pitch_spear_52}")
print(f"Spearman correlation pitch 5_3: {pitch_spear_53}")
print(f"Spearman correlation pitch 9_3: {pitch_spear_93}")
print(f"Spearman correlation roll 10_1: {roll_spear_101}")
print(f"Spearman correlation roll 9_3: {roll_spear_93}")
print(f"Spearman correlation roll 9_2: {roll_spear_92}")
print(f"Spearman correlation roll 9_1: {roll_spear_91}")    

print("------------------------------------------------")
# spearman correlations, pressure to Polaris
pitch_spear_pol_52 = spearmanr(pressure_pitch_52, pol_pitch_52, alternative='two-sided', nan_policy='propagate')
pitch_spear_pol_53 = spearmanr(pressure_pitch_53, pol_pitch_53, alternative='two-sided', nan_policy='propagate')
pitch_spear_pol_93 = spearmanr(pressure_pitch_93, pol_pitch_93, alternative='two-sided', nan_policy='propagate')
roll_spear_pol_101 = spearmanr(pressure_roll_101, pol_roll_101, alternative='two-sided', nan_policy='propagate')
roll_spear_pol_93 = spearmanr(pressure_roll_93, pol_roll_93, alternative='two-sided', nan_policy='propagate')
roll_spear_pol_92 = spearmanr(pressure_roll_92, pol_roll_92, alternative='two-sided', nan_policy='propagate')
roll_spear_pol_91 = spearmanr(pressure_roll_91, pol_roll_91, alternative='two-sided', nan_policy='propagate')
print(f"Spearman correlation pitch 5_2 Polaris: {pitch_spear_pol_52}")
print(f"Spearman correlation pitch 5_3 Polaris: {pitch_spear_pol_53}")
print(f"Spearman correlation pitch 9_3 Polaris: {pitch_spear_pol_93}")
print(f"Spearman correlation roll 10_1 Polaris: {roll_spear_pol_101}")
print(f"Spearman correlation roll 9_3 Polaris: {roll_spear_pol_93}")
print(f"Spearman correlation roll 9_2 Polaris: {roll_spear_pol_92}")
print(f"Spearman correlation roll 9_1 Polaris: {roll_spear_pol_91}")
