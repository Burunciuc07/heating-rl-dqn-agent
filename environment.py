import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time
from vars import *
import random


class Building:
    """ This class represents the building that has to be controlled. Its dynamics are modelled based on an RC analogy.
    When instanciated, it initialises the inside temperature to 21°C, the envelope temperature to 20, and resets the done
    and time variables.
    """
    def __init__(self, dynamic=False, eval=False):
        self.eval = eval
        self.dynamic = dynamic

        # temperaturi inițiale
        self.inside_temperature = 21.0
        self.envelope_temperature = 20.0

        # încarcă datele o singură dată
        self.weather_df = pd.read_csv(
            "data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv",
            header=3
        )
        self.prices_df = pd.read_csv("data/environment/spot_ro_prices.csv", header=0)

        # inițializează atributele episodului (ca să existe ambient_temperatures/prices etc.)
        self.reset()

    def heat_pump_power(self, phi_e):
        """Takes an electrical power flow and converts it to a heat flow.

        :param phi_e: The electrical power
        :type phi_e: Float
        :return: Returns the heat flow as an integer
        """
        return phi_e * (0.0606 * self.ambient_temperature + 2.612)

    def step(self, action):
        """
        :param action: (a_hvac, a_bat)
          - a_hvac: index in ACTIONS
          - a_bat: index in BATTERY_ACTIONS_KW
        :return: Returns the new state after a step, the reward for the action and the done state
        """
        # --- decode action ---
        
        n_hvac = len(ACTIONS)
        n_bat = len(BATTERY_ACTIONS_KW)

        a = int(action)  # poate veni din action.item() (float/int)
        a = max(0, min(a, n_hvac * n_bat - 1))

        a_hvac = a // n_bat
        a_bat = a % n_bat
        u = ACTIONS[a_hvac]
        p_bat_kw = float(BATTERY_ACTIONS_KW[a_bat])  # +charge, -discharge

        # --- HVAC thermal dynamics ---
        delta = 1 / (R_IA * C_I) * (self.ambient_temperature - self.inside_temperature) + \
                self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER * u) / C_I + A_w * self.sun_power / C_I
        self.inside_temperature += delta * TIME_STEP_SIZE

        # --- Battery SOC update ---
        dt_h = TIME_STEP_SIZE / 3600.0
        e_bat_req_kwh = p_bat_kw * dt_h  # energy at terminals (kWh)

        if e_bat_req_kwh >= 0:
            e_soc_kwh = e_bat_req_kwh * ETA_CH
        else:
            e_soc_kwh = e_bat_req_kwh / ETA_DISCH  # negative

        self.soc += e_soc_kwh / E_BAT_KWH
        self.soc = float(np.clip(self.soc, SOC_MIN, SOC_MAX))  # keep SOC in bounds [web:879]

        # --- reward (HVAC + battery energy cost) ---
        r = self.reward(a_hvac, e_bat_req_kwh)

        self.time += 1

        if self.dynamic:
            idx = int((self.time * TIME_STEP_SIZE) // 3600)  # 0..NUM_HOURS
            self.ambient_temperature = float(self.ambient_temperatures.iloc[idx])
            self.sun_power = float(self.sun_powers.iloc[idx])
            self.price = float(self.prices.iloc[idx])

        if self.time >= NUM_TIME_STEPS:
            self.done = True

        return [self.inside_temperature, self.ambient_temperature, self.sun_power, self.price, self.soc], r, self.done

    def reward(self, a_hvac, e_bat_req_kwh):
        a_hvac = int(a_hvac)
        a_hvac = max(0, min(a_hvac, len(ACTIONS) - 1))
        u = ACTIONS[a_hvac]

        if self.ambient_temperature <= T_MAX:
            penalty = np.maximum(0, self.inside_temperature - T_MAX) + np.maximum(0, T_MIN - self.inside_temperature)
            penalty *= COMFORT_PENALTY
        else:
            penalty = 0.0

        dt_h = TIME_STEP_SIZE / 3600.0  # ore
        power_kw = (NOMINAL_HEAT_PUMP_POWER / 1000.0) * u   # W -> kW
        hvac_energy_kwh = power_kw * dt_h

        # grid energy includes battery charge (+) / discharge (-)
        grid_energy_kwh = hvac_energy_kwh + e_bat_req_kwh

        price_lei_per_kwh = self.price / 1000.0  # lei/MWh -> lei/kWh
        cost = grid_energy_kwh * price_lei_per_kwh

        reward = -PRICE_PENALTY * cost - penalty
        reward = float(np.clip(reward, -1.0, 1.0))
        return reward

    def reset(self):
        self.inside_temperature = 21.0

        # --- 1) Start pentru vreme (Ninja DK) ---
        if self.eval:
            weather_start = 0
        else:
            # Nov/Dec în anul de vreme (ca înainte)
            weather_start = random.randint(304, 365 - NUM_HOURS // 24 - 1) * 24

        weather_df = pd.read_csv(
            "data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv",
            header=3
        )

        self.ambient_temperatures = (
            weather_df.iloc[weather_start: weather_start + NUM_HOURS + 1, 2]
            .reset_index(drop=True)
        )
        self.sun_powers = (
            weather_df.iloc[weather_start: weather_start + NUM_HOURS + 1, 3]
            .reset_index(drop=True)
        )

        self.ambient_temperature = float(self.ambient_temperatures.iloc[0])
        self.sun_power = float(self.sun_powers.iloc[0])

        # --- 2) Start pentru preț RO (independent) ---
        if self.eval:
            price_start = 0
        else:
            max_start = len(self.prices_df) - (NUM_HOURS + 1)
            price_start = random.randint(0, max_start)

        self.prices = (
            self.prices_df.iloc[price_start: price_start + NUM_HOURS + 1, 0]
            .reset_index(drop=True)
        )
        self.price = float(self.prices.iloc[0])

        self.done = False
        self.time = 0

        # --- Battery init + add SOC in state ---
        self.soc = float(SOC_INIT)

        return [self.inside_temperature, self.ambient_temperature, self.sun_power, self.price, self.soc]
