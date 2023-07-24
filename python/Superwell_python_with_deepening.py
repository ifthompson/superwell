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
lookup_idx = pd.Index(W_lookup.W)

# define constants
SECS_IN_DAY = 24 * 60 * 60

# user defined 
IRR_DEPTH = 0.30  # annual irrigation depth target (m)
NUM_YEARS = 100  # maximum years of pumping
DAYS = 100  # days pumping per year
MAXIMUM_INITIAL_SAT_THICKNESS = 200
NUM_INTERFERENCE_WELLS = 2
WELL_DEEPENING_DEPTH = 50  # m

# screening criteria
max_s_frac = 0.40  # max drawdown as % of sat thickness
max_s_absolute = 80  # max drawdown in m

# imported 
ELECTRICITY_RATE =  well_params.Val['Energy_cost_rate'] # defualt electricity rate
DEPLETION_LIMIT = well_params.Val['Depletion_Limit'] # depletion limit for this scenario 
EFFICIENCY = well_params.Val['Pump_Efficiency'] # well efficiency
WELL_LIFETIME = well_params.Val['Max_Lifetime_in_Years']
INTEREST_RATE = well_params.Val['Interest_Rate']
MAINTENANCE_RATE = well_params.Val['Maintenance_factor']
SPECIFIC_WEIGHT = well_params.Val['Specific_weight'] # specific weight of water

INTEREST_MULTIPLIER = ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME - 1)
#INTEREST_MULTIPLIER = (1 + INTEREST_RATE) * INTEREST_RATE  # equivalent expression, or is (WELL_LIFETIME-1) not

# time and well radius for Theis solution
time_Q = 2 * 365 * SECS_IN_DAY  # time period used for determining initial well Q
well_r = 0.5 * well_params.Val['Well_Diameter']

# convert electricity rate dictionary 
electricity_rate_dict = electricity_rates.electricity_cost_dollar_per_KWh.to_dict()
    
# filter by country, if desired
country = 'United States'
if country != 'all':
    grid_df = grid_df[grid_df.CNTRY_NAME == country]

################ determine if grid cell is skipped ########################
grid_df = grid_df[
    (grid_df.Area >= 10**7) &  # skip grid areas less than 10x10 km
    (grid_df.Depth >= 1) &  # depth to water table should we at least 5 meters
    (grid_df.Permeability >= -15) &  # limit low permeability values
    (grid_df.Porosity >= 0.05)  # limit porosity to 5% voids at least
]

# correct aquifer thickness outliers, replace >1000m thickness with 1000m
grid_df.Thickness.loc[grid_df.Thickness > 1000] = 1000

grid_df = grid_df.reset_index(drop=True)  # so that it starts at 0


# define Theis function
def drawdown_theis(time, r, S, T, Q):
    u = r**2 * S/(4 * T * time)

    if u < 0.6:  # use approximation for small u values
        W = -np.euler_gamma - math.log(u) + u - u**2/(2*2)
        
    elif 0.6 <= u <= 5.9:  # use W(u) lookup table for intermediate values where approximation is insufficient
        W = W_lookup.W[(W_lookup.u - u).abs().idxmin()]
        
    else:  # for large u values, W will be insignificant and drawdown (s) will ~= 0
        W = 0

    s = W * Q / (4 * np.pi * T)
    
    return s


def get_pumping_rate(time, r, S, T, Q_candidates, sat_thickness, max_s_frac, max_s_absolute):
    for Q in sorted(Q_candidates, reverse=True):
        s = drawdown_theis(time, r, S, T, Q)
        if s / sat_thickness < max_s_frac and s < max_s_absolute:
            return Q
    return 0


def simulate_pumping(r, roi, S, T, Q, sat_thickness, days=100, timestep=10):
    s_theis_ts = []
    s_theis_interference_ts = []

    for day in range(timestep, days + 1, timestep):
        s_theis_ts.append(drawdown_theis(day * SECS_IN_DAY, r, S, T, Q))
        s_theis_interference_ts.append(drawdown_theis(day * SECS_IN_DAY, roi * 2, S, T, Q))

    # average drawdown
    s_interference_avg = 4 * np.mean(s_theis_interference_ts)
    s_theis_avg = np.mean(s_theis_ts) + s_interference_avg

    # convert to Jacob - solve quadratic
    a = -1 / (2 * sat_thickness)
    b = 1
    c = -s_theis_avg

    s_jacob = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
    # s_jacob = (1 - np.sqrt(1 - 2 * s_theis_avg / sat_thickness)) * sat_thickness  # ?
    #root_2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

    return s_jacob, s_interference_avg

# candidate well pumping rates (gallons per minute)
Q_array_gpm = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1400, 1500]

Q_array = [i / (60 * 264.17) for i in Q_array_gpm]  # convert gpm to m^3/s
columns = ['DTW', 'sat_thickness', 'well_length', 'T', 'Well_Q', 'well_roi', 'well_area', 'depleted_volume_fraction',
           'drawdown', 'drawdown_interference', 'total_head', 'volume_per_well', 'num_wells', 'volume_all_wells',
           'cumulative_volume_per_well', 'cumulative_volume_all_wells', 'annual_capital_cost', 'maintenance_cost',
           'well_installation_cost', 'nonenergy_cost', 'power', 'energy', 'energy_cost_rate', 'energy_cost',
           'total_cost_per_well', 'total_cost_all_wells', 'unit_cost', 'unit_cost_per_km3', 'unit_cost_per_acreft']
pd.DataFrame(columns=columns).to_csv(output_file, mode='w')  # write headers to file

#%% superwell code block 
for grid_cell in grid_df.itertuples(name='Row'):

    # initialize an empty dataframe to store values for each year
    df = pd.DataFrame(data=0, index=range(NUM_YEARS), columns=columns)

    print('Percent complete = ' + str(100 * grid_cell.Index/len(grid_df)))  # TODO: progress bar package?

    ELECTRICITY_RATE = electricity_rate_dict[grid_cell.CNTRY_NAME]
    # assign well unit cost based on WHY Class
    well_unit_cost = well_params.Val[f'Well_Install_{grid_cell.WHYClass}']

    # depth to water 
    DTW = grid_cell.Depth  # initial depth to water

    sat_thickness = min(grid_cell.Thickness - grid_cell.Depth, MAXIMUM_INITIAL_SAT_THICKNESS)

    well_length = sat_thickness + grid_cell.Depth

    # available volume
    available_volume = (grid_cell.Thickness - grid_cell.Depth) * grid_cell.Area * grid_cell.Porosity
    initial_available_volume = available_volume
    
    # aquifer properties for Theis 
    K = (10 ** grid_cell.Permeability) * 1e7  # m/s
    T = K * sat_thickness  # m/s
    
    #################### determine initial well Q #############################
    
    # find largest Q that meets screening criteria
    Well_Q = get_pumping_rate(time_Q, well_r, grid_cell.Porosity, T, Q_array, sat_thickness, max_s_frac, max_s_absolute)

    # skip grid cell if no pumping rates are viable
    if Well_Q == 0:
        continue

    well_area = Well_Q * DAYS * SECS_IN_DAY / IRR_DEPTH  # m^2
    well_roi = np.sqrt(well_area / np.pi)  # m

    ####################### annual pumping simulation loop ####################

    for year in range(NUM_YEARS):
            
        # test viability for current year (simulate drawdown at t = 100 days of pumping)
        s_theis = drawdown_theis(DAYS * SECS_IN_DAY, well_r, grid_cell.Porosity, T, Well_Q)
        s_theis_interference = drawdown_theis(DAYS * SECS_IN_DAY, well_roi * NUM_INTERFERENCE_WELLS, grid_cell.Porosity, T, Well_Q)
        s_total = s_theis + 4 * s_theis_interference  # total drawdown (well + interference)

        # check if drawdown constraints are violated by end of 100 day pumping period
        # if constraints violated: (1) first deepen well, (2) then reduce well pumping rate 
        if s_total / sat_thickness > max_s_frac or s_total > max_s_absolute:

            # 1) first preference deepen well
            if well_length < grid_cell.Thickness:
                # deepen well length by WELL_DEEPENING_DEPTH, but do not exceed aquifer thickness
                well_length = min(well_length + WELL_DEEPENING_DEPTH, grid_cell.Thickness)

                # update saturated thickness and T 
                sat_thickness = well_length - DTW
                T = sat_thickness * K
                
            # 2) once well cannot be deepened, reduce well pumping rate 
            else:
                # find largest Q that meets screening criteria
                Well_Q = get_pumping_rate(time_Q, well_r, grid_cell.Porosity, T, Q_array, sat_thickness, max_s_frac, max_s_absolute)

                # exit pumping code block if no pumping rates are viable
                if Well_Q == 0:
                    break

                # update roi
                well_area = Well_Q * DAYS * SECS_IN_DAY / IRR_DEPTH
                well_roi = np.sqrt(well_area / np.pi)
           
        # if constraints aren't violated, proceed to calculate output for pumping year  
        # simulate 100 days of pumping, with drawdown calculated every 10 days
        s_jacob, s_interference_avg = simulate_pumping(well_r, well_roi, grid_cell.Porosity, T, Well_Q, sat_thickness)

        # write fundamental variables to df
        # pumping rate, well length, number of wells, (and pumping time) are the quantities users can control,
        # and other variables can be simply derived from these and the grid cell properties with column operations
        df.Well_Q[year] = Well_Q
        df.well_length[year] = well_length
        df.num_wells[year] = grid_cell.Area / well_area  # possibly have num_wells calculated a different way

        # save annual drawdown values to df
        df.drawdown[year] = s_jacob
        df.drawdown_interference[year] = s_interference_avg

        # update for next iteration
        volume_all_wells = Well_Q * SECS_IN_DAY * DAYS * (grid_cell.Area / well_area)
        available_volume -= volume_all_wells

        DTW += (volume_all_wells / grid_cell.Area) / grid_cell.Porosity
        sat_thickness = well_length - DTW
        T = K * sat_thickness

        # end the loop if the depletion limit has been reached
        if available_volume / initial_available_volume < DEPLETION_LIMIT:
            break

    # compute derived columns

    df.volume_per_well = df.Well_Q * SECS_IN_DAY * DAYS
    df.volume_all_wells = df.volume_per_well * df.num_wells
    df.cumulative_volume_per_well = df.volume_per_well.cumsum()
    df.cumulative_volume_all_wells = df.volume_all_wells.cumsum()
    df.depleted_volume_fraction = df.cumulative_volume_all_wells / initial_available_volume

    df.DTW = grid_cell.Depth + (df.cumulative_volume_all_wells / grid_cell.Area) / grid_cell.Porosity  # at end of year
    df.total_head = df.drawdown + df.DTW

    # equivalent behavior?
    # comment mentioned need to reduce unit cost for lower rate but not implemented?
    # need to adjust WHYClass as pumping rate decreases every well lifetime.
    df.annual_capital_cost = df.num_wells * df.well_length * well_unit_cost * INTEREST_MULTIPLIER
    df.maintenance_cost = df.num_wells * df.well_length * well_unit_cost * MAINTENANCE_RATE

    df.well_installation_cost = well_unit_cost * df.well_length
    df.nonenergy_cost = df.annual_capital_cost + df.maintenance_cost
    df.power = df.num_wells * (SPECIFIC_WEIGHT * df.total_head * df.Well_Q / EFFICIENCY) / 1000  # kW
    df.energy = df.power * (DAYS * 24)  # kWh/year
    df.energy_cost_rate = ELECTRICITY_RATE  # $ per kWh
    df.energy_cost = df.energy * df.energy_cost_rate  # $/year
    df.total_cost_all_wells = df.nonenergy_cost + df.energy_cost
    df.total_cost_per_well = df.total_cost_all_wells / df.num_wells
    df.unit_cost = df.total_cost_all_wells / df.volume_all_wells  # $/m^3
    df.unit_cost_per_km3 = df.unit_cost * 10 ** 9  # $/km^3
    df.unit_cost_per_acreft = df.unit_cost * 1233.48  # $/acft

    df.to_csv(output_file, mode='a', header=False)  # append to file

    ######################## save grid cell results ###########################
    """
    ['iteration', 'year_number', 'DEPLETION_LIMIT', 'continent', 'country_name', 
    'gcam_basin_id', 'gcam_basin_name', 'well_id', 'grid_area', 'permeability', 'storativity', 
    'total_thickness', 'depth_to_piez_surface', 'orig_aqfr_sat_thickness', 'aqfr_sat_thickness', 
    'hydraulic_conductivity', 'transmissivity', 'radius_of_influence', 'areal_extent', 
    'max_drawdown', 'drawdown', 'drawdown_interference', 'total_head', 'well_yield', 'volume_produced_perwell', 
    'cumulative_vol_produced_perwell', 'number_of_wells', 'volume_produced_allwells', 
    'cumulative_vol_produced_allwells', 'available_volume', 'depleted_vol_fraction', 
    'well_installation_cost', 'annual_capital_cost', 'maintenance_cost', 'nonenergy_cost', 
    'power', 'energy', 'energy_cost_rate', 'energy_cost', 'total_cost_perwell', 'total_cost_allwells', 
    'unit_cost', 'unit_cost_per_km3', 'unit_cost_per_acreft', 'whyclass', 'total_well_length']
    """
