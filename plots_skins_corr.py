import pandas as pd
import matplotlib.pyplot as plt

# load selected samples 
pitch5 = pd.read_csv('skins-test-outputs/hysteresis/cm0-5/pitch2-avg.csv', dtype=float)
roll5 = pd.read_csv('skins-test-outputs/hysteresis/cm0-5/roll4-avg.csv', dtype=float)
pitch10 = pd.read_csv('skins-test-outputs/hysteresis/cm1-0/pitch5-avg.csv', dtype=float)
roll10 = pd.read_csv('skins-test-outputs/hysteresis/cm1-0/roll4-avg.csv', dtype=float)
pitch15 = pd.read_csv('skins-test-outputs/hysteresis/cm1-5/pitch5-avg.csv', dtype=float)
roll15 = pd.read_csv('skins-test-outputs/hysteresis/cm1-5/roll4-avg.csv', dtype=float)

pitch5load, pitch5unload, pitch5load_g, pitch5unload_g = pitch5["Loading_avg"], pitch5["Unloading_avg"], pitch5["Gnd_load_avg"], pitch5["Gnd_unload_avg"]
roll5load, roll5unload, roll5load_g, roll5unload_g = roll5["Loading_avg"], roll5["Unloading_avg"], roll5["Gnd_load_avg"], roll5["Gnd_unload_avg"]
pitch10load, pitch10unload, pitch10load_g, pitch10unload_g = pitch10["Loading_avg"], pitch10["Unloading_avg"], pitch10["Gnd_load_avg"], pitch10["Gnd_unload_avg"]
roll10load, roll10unload, roll10load_g, roll10unload_g = roll10["Loading_avg"], roll10["Unloading_avg"], roll10["Gnd_load_avg"], roll10["Gnd_unload_avg"]
pitch15load, pitch15unload, pitch15load_g, pitch15unload_g = pitch15["Loading_avg"], pitch15["Unloading_avg"], pitch15["Gnd_load_avg"], pitch15["Gnd_unload_avg"]
roll15load, roll15unload, roll15load_g, roll15unload_g = roll15["Loading_avg"], roll15["Unloading_avg"], roll15["Gnd_load_avg"], roll15["Gnd_unload_avg"]

t_pitch = pitch5["Time"]
t_roll = roll5["Time"]
# fig, ax = plt.subplots(2, 2)
# 4 figs: pitchLoad, pitchUnload, rollLoad, rollUnload
fig1 = plt.figure(1, figsize=(5,3))
plt.plot(t_pitch, pitch5load, "b.", label="MCP 5mm")
plt.plot(t_pitch, pitch5load_g, "b--", label="Gnd 5mm")
plt.plot(t_pitch, pitch10load, "r.", label="MCP 10mm")
plt.plot(t_pitch, pitch10load_g, "r--", label="Gnd 10mm")
plt.plot(t_pitch, pitch15load, "g.", label="MCP 15mm")
plt.plot(t_pitch, pitch15load_g, "g--", label="Gnd 15mm")
plt.xlabel("Time (s)")
plt.ylabel("Loading Angle (deg)")
plt.legend()
fig1.canvas.manager.set_window_title("Pitch Loading")
plt.tight_layout()

fig2 = plt.figure(2, figsize=(5,3))
plt.plot(t_pitch, pitch5unload, "b.", label="MCP 5mm")
plt.plot(t_pitch, pitch5unload_g, "b--", label="Gnd 5mm")
plt.plot(t_pitch, pitch10unload, "r.", label="MCP 10mm")
plt.plot(t_pitch, pitch10unload_g, "r--", label="Gnd 10mm")
plt.plot(t_pitch, pitch15unload, "g.", label="MCP 15mm")
plt.plot(t_pitch, pitch15unload_g, "g--", label="Gnd 15mm")
plt.xlabel("Time (s)")
plt.ylabel("Unloading Angle (deg)")
plt.legend()
fig2.canvas.manager.set_window_title("Pitch Unloading")
plt.tight_layout()

fig3 = plt.figure(3, figsize=(5,3))
plt.plot(t_roll, roll5load, "b.", label="MCP 5mm")
plt.plot(t_roll, roll5load_g, "b--", label="Gnd 5mm")
plt.plot(t_roll, roll10load, "r.", label="MCP 10mm")  
plt.plot(t_roll, roll10load_g, "r--", label="Gnd 10mm")
plt.plot(t_roll, roll15load, "g.", label="MCP 15mm")
plt.plot(t_roll, roll15load_g, "g--", label="Gnd 15mm")
plt.xlabel("Time (s)")
plt.ylabel("Loading Angle (deg)")
plt.legend()
fig3.canvas.manager.set_window_title("Roll Loading")
plt.tight_layout()

fig4 = plt.figure(4, figsize=(5,3))
plt.plot(t_roll, roll5unload, "b.", label="MCP 5mm")
plt.plot(t_roll, roll5unload_g, "b--", label="Gnd 5mm")
plt.plot(t_roll, roll10unload, "r.", label="MCP 10mm") 
plt.plot(t_roll, roll10unload_g, "r--", label="Gnd 10mm")
plt.plot(t_roll, roll15unload, "g.", label="MCP 15mm")
plt.plot(t_roll, roll15unload_g, "g--", label="Gnd 15mm")
plt.xlabel("Time (s)")
plt.ylabel("Unloading Angle (deg)")
plt.legend()
fig4.canvas.manager.set_window_title("Roll Unloading")
plt.tight_layout()

plt.show()
