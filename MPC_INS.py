import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" 

import math, sys, gc, joblib, time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pyenergyplus.plugin import EnergyPlusPlugin


from pymoo.config import Config
Config.warnings['not_compiled'] = False
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# --- GLOBAL CONFIGURATION (EXPERT INTERFACE) ---
MPC_CONFIG = {
    "GA": {"POP": 10, "GEN": 20, "VARS": 23, "LB": 0.2, "UB": 1.0},
    "DECISION": {"W_HEAT": 1.0, "W_COOL": 1.0},
    "ARCH": {"LOOKBACK": 24, "OUTPUTS": 4},
    "PATHS": {"M": "../python/model.h5", "S": "../python/scaler.pkl", 
              "D": "../python/initial_data.csv", "O": "../python/python_output_file.csv",
              "LOGO": "../python/logo.png"}
}

# --- ASSET SERIALIZATION & BUFFER INIT ---
def custom_weighted_mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred) * tf.constant([1.0, 1.0, 2.0, 3.0]))

SCALER = joblib.load(MPC_CONFIG["PATHS"]["S"])
MODEL = load_model(MPC_CONFIG["PATHS"]["M"], custom_objects={'custom_weighted_mae_loss': custom_weighted_mae_loss})
DF_HIST = pd.read_csv(MPC_CONFIG["PATHS"]["D"]).assign(isFuture=False)

# Runtime Buffers
_shift, _opt_active, _opt_count = 0, False, 0
_DATA_RES, _SIS_H = None, 0.8

# --- CORE LOGIC ---
def get_features(df):
    df['Time'] = pd.to_datetime(df['Time'])
    t = df['Time'].dt
    return df.assign(hour=t.hour, dayofweek=t.dayofweek, quarter=t.quarter, month=t.month, 
                     dayofyear=t.dayofyear, dayofmonth=t.day, weekofyear=t.isocalendar().week.astype(int))

def run_surrogate(SIS_seq):
    global _opt_count, _opt_active, DF_HIST
    S_in = np.insert(SIS_seq, 0, _SIS_H)
    f_t = pd.Timestamp('2024-01-01 01:00:00') + pd.DateOffset(hours=_shift)
    f_df = get_features(pd.DataFrame({'Time': pd.date_range(f_t, periods=24, freq='1h'), 'SIS': S_in}))
    
    if not _opt_active:
        d_curr = pd.DataFrame([_DATA_RES], columns=['Temp', 'RH', 'Heating', 'Cooling', 'SIS'])
        res = pd.concat([d_curr, f_df.head(1).drop(columns=['Time', 'SIS'])], axis=1)
        for c in ['Temp', 'RH', 'Heating', 'Cooling']:
            res[f'{c}_lag1'] = DF_HIST.iloc[-24][c]
        res['isFuture'], res['Time'] = False, f_df.head(1)['Time'].values[0]
        DF_HIST = pd.concat([DF_HIST, res], ignore_index=True).tail(256)
        _opt_active = True
    
    full_df = pd.concat([DF_HIST, f_df.iloc[1:]], ignore_index=True)
    full_df.iloc[:, :16] = SCALER.transform(full_df.iloc[:, :16])
    for c in ['Temp', 'RH', 'Heating', 'Cooling']:
        full_df[f'{c}_lag1'] = full_df[c].shift(24)
    
    dp = full_df.drop(columns=['Temp', 'RH', 'Heating', 'Cooling', 'Time', 'isFuture']).dropna()
    X_input = dp.values.reshape(dp.shape[0], 1, dp.shape[1])
    
    preds = MODEL(tf.convert_to_tensor(X_input.astype('float64')), training=False).numpy()
    out_wide = np.zeros((preds.shape[0], 16))
    out_wide[:, :4] = preds
    inv = SCALER.inverse_transform(out_wide)[:, :4]
    
    _opt_count += 1
    return np.sum(np.maximum(0, inv[-24:, 2])), np.sum(np.maximum(0, inv[-24:, 3]))

class SurrogateProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=MPC_CONFIG["GA"]["VARS"], n_obj=2, xl=MPC_CONFIG["GA"]["LB"], xu=MPC_CONFIG["GA"]["UB"])
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = list(run_surrogate(np.round(x, 1)))
        gc.collect()

# --- CO-SIMULATION INTERFACE ---
class SetConstructionControlState(EnergyPlusPlugin):
    def __init__(self):
        super().__init__()
        self.h_set, self.c_state, self.itr, self.msg_shown = False, 0.8, 0, False

    def on_begin_timestep_before_predictor(self, state) -> int:
        global _shift, _opt_count, _opt_active, _DATA_RES, _SIS_H
        if not self.h_set:
            self.h_v = [self.api.exchange.get_variable_handle(state, v, "Thermal Zone 1") 
                        for v in ["Zone Air Temperature", "Zone Air Relative Humidity", 
                                  "Zone Air System Sensible Heating Rate", "Zone Air System Sensible Cooling Rate"]]
            self.h_a = self.api.exchange.get_actuator_handle(state, "Schedule:Constant", "Schedule Value", "SIS SCH")
            self.h_set = True

        hr = self.api.exchange.current_sim_time(state)
        v_curr = [self.api.exchange.get_variable_value(state, h) for h in self.h_v]
        _SIS_H = self.api.exchange.get_actuator_value(state, self.h_a)
        _DATA_RES = v_curr + [_SIS_H]

        if hr == 1: self.itr += 1
        
        if self.itr >= 15 and abs(hr - math.floor(hr)) < 1e-6:
            if not self.msg_shown:
                print('******** MPC Control initiated - Model: LSTM + NSGA-II ********')
                try:
                    import pygame
                    pygame.init()
                    logo = pygame.image.load(MPC_CONFIG["PATHS"]["LOGO"])
                    win = pygame.display.set_mode(logo.get_size(), pygame.NOFRAME)
                    t_end = time.time() + 5
                    while time.time() < t_end:
                        for e in pygame.event.get(): pass
                        win.blit(logo, (0, 0))
                        pygame.display.update()
                    pygame.display.quit(); pygame.quit()
                except: pass
                self.msg_shown = True

            res = minimize(SurrogateProblem(), NSGA2(pop_size=MPC_CONFIG["GA"]["POP"]), 
                           get_termination("n_gen", MPC_CONFIG["GA"]["GEN"]), seed=1, verbose=False)
            
            cost = (res.F[:, 0] * MPC_CONFIG["DECISION"]["W_HEAT"]) + (res.F[:, 1] * MPC_CONFIG["DECISION"]["W_COOL"])
            self.c_state = float(round(res.X[np.argmin(cost)][0], 1))
            
            print(f"******** HOUR: {hr:4.0f} | SIS assigned: {self.c_state} ********")
            
            _shift += 1
            _opt_active, _opt_count = False, 0
            gc.collect()

        self.api.exchange.set_actuator_value(state, self.h_a, self.c_state)
        return 0