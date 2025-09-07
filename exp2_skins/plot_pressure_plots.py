import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob 
import sys

class PlotPressurePlots:
    def __init__(self,spacing):
        self.pitch_paths = sorted(glob.glob(f"data_collection_with_franka/B07LabTrials/skins-data/cm{spacing}/pitch/fibrescope-13*.csv"))
        self.roll_paths = sorted(glob.glob(f"data_collection_with_franka/B07LabTrials/skins-data/cm{spacing}/roll/fibrescope-13*.csv"))
        self.spacing = spacing
        # breakpoint()
        if not self.pitch_paths or not self.roll_paths:
            raise ValueError("No data files found. Please check the file paths and ensure the data files are present.")
        
    def pressure_extract_motion(self,motion_path): # extract pressure values and create a matrix for each motion (pitch and roll)
        pressure_data = []
        
        for path in motion_path:
            data = pd.read_csv(path, usecols=['Pressure (kPa)'])
            pressure_data.append(data['Pressure (kPa)'].values)
                
        return pressure_data
    def plot_pressure(self, save):
        trim = 2
        pitch_pressure = self.pressure_extract_motion(self.pitch_paths)
        roll_pressure = self.pressure_extract_motion(self.roll_paths)
        
        ts = np.linspace(0,60,len(pitch_pressure[0])-trim) # time vector for 60 seconds

        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(ts, pitch_pressure[0][trim:], ts, pitch_pressure[1][trim:], ts, pitch_pressure[2][trim:], ts, pitch_pressure[3][trim:], ts, pitch_pressure[4][trim:])
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (kPa), pitch')

        plt.subplot(2, 1, 2)
        plt.plot(ts, roll_pressure[0][trim:], ts, roll_pressure[1][trim:], ts, roll_pressure[2][trim:], ts, roll_pressure[3][trim:], ts, roll_pressure[4][trim:])
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (kPa), roll')

        plt.tight_layout()
        
        if save == True:
            plt.savefig(f'figs_sampl/data2_skins/pressure_plots/pressure_plots_cm{self.spacing}.png')
            plt.savefig(f'figs_sampl/data2_skins/pressure_plots/pressure_plots_cm{self.spacing}.svg')
            print("Plot saved")
        plt.show()

        setpoint = 1.7
        # breakpoint()
        print(f"control: overall mean pressure: pitch - {np.mean(np.mean(pitch_pressure, axis=0))}, roll - {np.mean(np.mean(roll_pressure, axis=0))}")
        print(f"Max deviation pitch: {np.abs(np.max(np.mean(pitch_pressure, axis=0)[trim:]) - setpoint)}")
        print(f"Min deviation pitch: {np.abs(np.min(np.mean(pitch_pressure, axis=0)[trim:]) - setpoint)}")
        print(f"Max deviation roll: {np.abs(np.max(np.mean(roll_pressure, axis=0)[trim:]) - setpoint)}")
        print(f"Min deviation roll: {np.abs(np.min(np.mean(roll_pressure, axis=0)[trim:]) - setpoint)}")
if __name__ == "__main__":
    save = sys.argv[1].lower() == 'save' if len(sys.argv) > 1 else False
    try:
        try:
            print("Plotting pressure plots for spacing 0-5 cm...")
            plotter = PlotPressurePlots(spacing="0-5")
            plotter.plot_pressure(save)
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            print("Plotting pressure plots for spacing 1-0 cm...")
            plotter = PlotPressurePlots(spacing="1-0")
            plotter.plot_pressure(save)
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            print("Plotting pressure plots for spacing 1-5 cm...")
            plotter = PlotPressurePlots(spacing="1-5")
            plotter.plot_pressure(save)
        except Exception as e:
            print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        SystemExit("Process interrupted by user.")
            