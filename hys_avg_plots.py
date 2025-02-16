import matplotlib.pyplot as plt
import pandas as pd
import glob 
import sys

def hys_avg_plot(t_load, loading_avg, gnd_load_avg, unloading_avg, gnd_unload_avg, fig_title):
    fig, ax = plt.subplots()
    # fig.subplot(2,1,1)
    ax.errorbar(t_load, loading_avg, fmt='ro', label='Loading')
    ax.errorbar(t_load, gnd_load_avg, fmt='r-', label='Ground Load')
    ax.errorbar(t_load, unloading_avg, fmt='bo', label='Unloading')
    ax.errorbar(t_load, gnd_unload_avg, fmt='b-', label='Ground Unload')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Degree of rotation')
    # ax.set_title('Hysteresis Test')
    ax.legend()
    fig.canvas.manager.set_window_title(fig_title)
    # plt.show()
    
def readfile(pitchrollN, spacing):

    pitchroll_list = glob.glob("skins-test-outputs/hysteresis/cm"+spacing+"/"+pitchrollN+"-errbar.csv")
    loading_df = pd.DataFrame(index=None)
    loading_gnd_df = pd.DataFrame(index=None)
    unloading_df = pd.DataFrame(index=None)
    unloading_gnd_df = pd.DataFrame(index=None)
    for file in pitchroll_list:
        df = pd.read_csv(file, usecols={"Loading_avg", "Gnd_load_avg", "Unloading_avg", "Gnd_unload_avg"})
        loading_df = pd.concat([loading_df, df["Loading_avg"]], ignore_index=True, axis=1)
        loading_gnd_df = pd.concat([loading_gnd_df, df["Gnd_load_avg"]], ignore_index=True, axis=1)
        unloading_df = pd.concat([unloading_df, df["Unloading_avg"]], ignore_index=True, axis=1)
        unloading_gnd_df = pd.concat([unloading_gnd_df, df["Gnd_unload_avg"]], ignore_index=True, axis=1)
    
    t_load = pd.read_csv(pitchroll_list[0], usecols={"Time"})
    fig_title = pitchrollN+" "+spacing+" cm - hysteresis"
    hys_avg_plot(t_load, loading_df.mean(axis=1), loading_gnd_df.mean(axis=1), unloading_df.mean(axis=1), unloading_gnd_df.mean(axis=1), fig_title)

def main():
    readfile("pitch[1|2|3|4|5]", "0-5")
    readfile("pitch*mod", "0-5")
    readfile("roll*", "0-5")
    readfile("pitch*", "1-0")
    readfile("roll*", "1-0")
    readfile("pitch*", "1-5")
    readfile("roll*", "1-5")
    plt.show()
    # hys_avg_plot()
    
if __name__ == "__main__":
    try: 
        main()
    except KeyboardInterrupt:
        plt.close()
        sys.exit(1)