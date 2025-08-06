import pandas as pd
import numpy as np
import glob
import sys
from opticalflow import main as klt_main
    # data_collection_with_franka/B07LabTrials/participant-data/participant1/fibrescope-pitch1-20-Nov-2024--13-58-53.mp4

class ParticipantDataSort:
    def __init__(self, ):
        self.participant_data_path = ...
        
    def main(self):
        ...
if __name__ == "__main__":
    participant_data_sort = ParticipantDataSort()
    try: 
        participant_data_sort.main()
    except KeyboardInterrupt:
        SystemExit("Exiting participant data sort script due to keyboard interrupt.")

