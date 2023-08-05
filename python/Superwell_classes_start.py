# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import os

inputs_path = os.path.join(os.getcwd(), 'inputs')

output_path = 'outputs'
output_name = 'superwell_python_test'  # file name for output
output_file = os.path.join(output_path, f'{output_name}.csv')

# load data
grid_df = pd.read_csv(os.path.join(inputs_path, 'inputs.csv'), keep_default_na=False)
well_params = pd.read_csv(os.path.join(inputs_path, 'Well_Params.csv'), index_col=0)
electricity_rates = pd.read_csv(os.path.join(inputs_path, 'GCAM_Electrical_Rates.csv'), index_col='country')
W_lookup = pd.read_csv(os.path.join(inputs_path, 'Theis_well_function_table.csv'))

# define constants
SECS_IN_DAY = 24 * 60 * 60

INTEREST_MULTIPLIER = ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE / (
            (1 + INTEREST_RATE) ** WELL_LIFETIME - 1)
# INTEREST_MULTIPLIER = (1 + INTEREST_RATE) * INTEREST_RATE  # equivalent expression, or is (WELL_LIFETIME-1) not


# convert electricity rate dictionary
electricity_rate_dict = electricity_rates.electricity_cost_dollar_per_KWh.to_dict()

# filter by country, if desired
country = 'United States'
if country != 'all':
    grid_df = grid_df[grid_df.CNTRY_NAME == country]

# determine if grid cell is skipped
grid_df = grid_df[
    (grid_df.Area >= 10 ** 7) &  # skip grid areas less than 10x10 km
    (grid_df.Depth >= 1) &  # depth to water table should we at least 5 meters
    (grid_df.Permeability >= -15) &  # limit low permeability values
    (grid_df.Porosity >= 0.05)  # limit porosity to 5% voids at least
    ]

# correct aquifer thickness outliers, replace >1000m thickness with 1000m
grid_df.Thickness.loc[grid_df.Thickness > 1000] = 1000

grid_df = grid_df.reset_index(drop=True)  # so that it starts at 0


# define Theis function
def drawdown_theis(time, r, S, T, Q):
    u = r ** 2 * S / (4 * T * time)

    if u < 0.6:  # use approximation for small u values
        W = -np.euler_gamma - math.log(u) + u - u ** 2 / (2 * 2)

    elif 0.6 <= u <= 5.9:  # use W(u) lookup table for intermediate values where approximation is insufficient
        W = W_lookup.W[(W_lookup.u - u).abs().idxmin()]

    else:  # for large u values, W will be insignificant and drawdown (s) will ~= 0
        W = 0

    s = W * Q / (4 * np.pi * T)

    return s


class GridCell:
    def __init__(self, cell_area, thickness, porosity, permeability, depth_to_water,
                 well_radius, well_length, pumping_duration, IRR_DEPTH, WELL_DEEPENING_DEPTH,
                 max_s_frac, max_s_absolute, DEPLETION_LIMIT, Q_array):
        # constant physical properties of the cell
        self.cell_area = cell_area
        self.thickness = thickness
        self.porosity = porosity
        self.hydraulic_conductivity = (10 ** permeability) * 1e7  # m/s

        self.available_volume = (thickness - depth_to_water) * cell_area * porosity
        self.initial_available_volume = self.available_volume
        self.initial_depth_to_water = depth_to_water

        self.max_s_frac = max_s_frac
        self.max_s_absolute = max_s_absolute
        self.DEPLETION_LIMIT = DEPLETION_LIMIT
        self.WELL_DEEPENING_DEPTH = WELL_DEEPENING_DEPTH
        self.Q_array = Q_array

        self.IRR_DEPTH = IRR_DEPTH
        self.well_radius = well_radius

        # these 4 levers control the total volume of water pumped each year
        self.well_length = well_length
        self.number_of_wells = 0
        self.pumping_duration = pumping_duration
        self.pumping_rate = 0

    @property
    def depth_to_water(self):
        return self.thickness - self.available_volume / (self.cell_area * self.porosity)

    @property
    def saturated_thickness(self):
        return self.well_length - self.depth_to_water

    @property
    def transmissivity(self):
        return self.hydraulic_conductivity * self.saturated_thickness

    @property
    def well_area(self):
        return self.pumping_rate * self.pumping_duration / self.IRR_DEPTH

    @property
    def well_roi(self):
        return np.sqrt(self.well_area / np.pi)

    def total_drawdown(self, time):
        s = drawdown_theis(time, self.well_radius, self.porosity, self.transmissivity, self.pumping_rate)
        s_interference = drawdown_theis(time, self.well_roi, self.porosity, self.transmissivity, self.pumping_rate)
        return s + 4 * s_interference

    def simulate_drawdown(self, timestep=10 * SECS_IN_DAY):
        # simulate drawdown each timestep
        ts = [self.total_drawdown(time) for time in range(timestep, self.pumping_duration + 1, timestep)]
        s_theis = sum(ts) / len(ts)  # compute average

        # apply jacob correction
        s_jacob = self.saturated_thickness * (1 - np.sqrt(1 - 2 * s_theis / self.saturated_thickness))

        return s_jacob

    def calibrate_pumping_rate(self, time):
        for Q in sorted(self.Q_array, reverse=True):
            s = drawdown_theis(time, self.well_radius, self.porosity, self.transmissivity, Q)
            if s / self.saturated_thickness < self.max_s_frac and s < self.max_s_absolute:
                return Q
        return 0

    def run_year(self):
        # if needed, some more complex behavior could be implemented here,
        # like decisions to add wells or increase/decrease pumping duration,
        # or to not pump more than some specified demand

        s = self.total_drawdown(self.pumping_duration)
        if s > self.max_s_frac or s > self.max_s_absolute:
            # increase well length if possible, otherwise reduce pumping rate
            if self.well_length < self.thickness:
                self.well_length = min(self.well_length + self.WELL_DEEPENING_DEPTH, self.thickness)
            else:
                self.pumping_rate = self.calibrate_pumping_rate(self.pumping_duration)
                self.number_of_wells = 0 if self.well_area == 0 else self.cell_area / self.well_area

        self.drawdown = self.simulate_drawdown(timestep=10 * SECS_IN_DAY)
        # remove water from cell, affecting depth_to_water -> saturated_thickness -> transmissivity
        self.available_volume -= self.pumping_rate * self.pumping_duration * self.number_of_wells

    def run(self, years):
        data = []
        for i in range(years):
            self.run_year()
            data.append((self.pumping_rate, self.number_of_wells, self.well_length))

            # end the loop if well shutdown or the depletion limit has been reached
            if self.pumping_rate == 0 or self.available_volume / self.initial_available_volume < self.DEPLETION_LIMIT:
                break

        df = pd.DataFrame.from_records(data, columns=['pumping_rate', 'number_of_wells', 'well_length'])

        # compute derived columns

        df.volume_per_well = df.pumping_rate * df.pumping_time
        df.volume_all_wells = df.volume_per_well * df.number_of_wells
        df.cumulative_volume_per_well = df.volume_per_well.cumsum()
        df.cumulative_volume_all_wells = df.volume_all_wells.cumsum()
        df.depleted_volume_fraction = df.cumulative_volume_all_wells / self.initial_available_volume

        df.depth_to_water = self.initial_depth_to_water + df.cumulative_volume_all_wells / (
                    self.cell_area * self.porosity)  # at end of year
        df.total_head = df.drawdown + df.depth_to_water

        # equivalent behavior?
        # comment mentioned need to reduce unit cost for lower rate but not implemented?
        # need to adjust WHYClass as pumping rate decreases every well lifetime.
        df.annual_capital_cost = df.number_of_wells * df.well_length * well_unit_cost * INTEREST_MULTIPLIER
        df.maintenance_cost = df.number_of_wells * df.well_length * well_unit_cost * MAINTENANCE_RATE

        df.well_installation_cost = well_unit_cost * df.well_length
        df.nonenergy_cost = df.annual_capital_cost + df.maintenance_cost
        df.power = df.number_of_wells * (SPECIFIC_WEIGHT * df.total_head * df.pumping_rate / EFFICIENCY) / 1000  # kW
        df.energy = df.power * (DAYS * 24)  # kWh/year
        df.energy_cost_rate = ELECTRICITY_RATE  # $ per kWh
        df.energy_cost = df.energy * df.energy_cost_rate  # $/year
        df.total_cost_all_wells = df.nonenergy_cost + df.energy_cost
        df.total_cost_per_well = df.total_cost_all_wells / df.number_of_wells
        df.unit_cost = df.total_cost_all_wells / df.volume_all_wells  # $/m^3
        df.unit_cost_per_km3 = df.unit_cost * 10 ** 9  # $/km^3
        df.unit_cost_per_acreft = df.unit_cost * 1233.48  # $/acft

        return df


class Scenario:
    def __init__(self):
        self.IRR_DEPTH = 0
        self.IRR_DEPTH = 0.30  # annual irrigation depth target (m)
        self.NUM_YEARS = 100  # maximum years of pumping
        self.DAYS = 100  # days pumping per year
        self.MAXIMUM_INITIAL_SAT_THICKNESS = 200
        self.NUM_INTERFERENCE_WELLS = 2
        self.WELL_DEEPENING_DEPTH = 50  # m
        self.time_Q = 2 * 365 * SECS_IN_DAY  # time period used for determining initial well Q
        self.well_r = 0.5 * well_params.Val['Well_Diameter']
        self.max_s_frac = 0.40  # max drawdown as % of sat thickness
        self.max_s_absolute = 80  # max drawdown in m
        self.ELECTRICITY_RATE = well_params.Val['Energy_cost_rate']  # default electricity rate
        self.DEPLETION_LIMIT = well_params.Val['Depletion_Limit']  # depletion limit for this scenario
        self.EFFICIENCY = well_params.Val['Pump_Efficiency']  # well efficiency
        self.WELL_LIFETIME = well_params.Val['Max_Lifetime_in_Years']
        self.INTEREST_RATE = well_params.Val['Interest_Rate']
        self.MAINTENANCE_RATE = well_params.Val['Maintenance_factor']
        self.SPECIFIC_WEIGHT = well_params.Val['Specific_weight']  # specific weight of water
        Q_array_gpm = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300,
                       1400, 1500]
        self.Q_array = [i / (60 * 264.17) for i in Q_array_gpm]  # convert gpm to m^3/s

columns = ['DTW', 'sat_thickness', 'well_length', 'T', 'Well_Q', 'well_roi', 'well_area', 'depleted_volume_fraction',
           'drawdown', 'drawdown_interference', 'total_head', 'volume_per_well', 'num_wells', 'volume_all_wells',
           'cumulative_volume_per_well', 'cumulative_volume_all_wells', 'annual_capital_cost', 'maintenance_cost',
           'well_installation_cost', 'nonenergy_cost', 'power', 'energy', 'energy_cost_rate', 'energy_cost',
           'total_cost_per_well', 'total_cost_all_wells', 'unit_cost', 'unit_cost_per_km3', 'unit_cost_per_acreft']
